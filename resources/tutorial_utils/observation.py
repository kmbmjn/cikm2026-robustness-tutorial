"""M1 presentation helpers; retrieval, cache contents, and metric APIs stay shared.

Each observation is recomputed from its explicit query/severity arguments. This
keeps a participant's out-of-order cell reruns independent of earlier rankings.
"""
from html import escape
import textwrap

import pandas as pd

from .config import ValidationError
from .metrics import ndcg, recall, rbo
from .presentation import interpretation_html, show_interpretation
from .retrieval import query_text, retrieve, rerank

TIE = 1e-12  # Score differences this small count as equal.


def _join(items):
    """'A', 'A and B', or 'A, B, and C'."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else \
        ", ".join(items[:-1]) + ", and " + items[-1]


def _positives(ctx, query_id):
    return {d for d, grade in ctx.qrels[query_id].items() if grade > 0}


def _relevant_ranks(ranking, positives, depth=10):
    return [rank for rank, (doc_id, _) in enumerate(ranking[:depth], 1) if doc_id in positives]


def _edit_phrase(edits):
    """'one character edit (muscle → musckle)', in reading order of the claim."""
    words = ", ".join(f'{e["before"]} → {e["after"]}' for e in sorted(edits, key=lambda e: e["start"]))
    count = {1: "one character edit", 2: "two character edits",
             3: "three character edits"}.get(len(edits), f"{len(edits)} character edits")
    return f"{count} ({words})"


def observe(ctx, query_id, severity=1):
    """Return both conditions for three systems, each retrieving 20 candidates."""
    if ctx.mode not in {"cached", "live"}:
        raise ValidationError("M1 needs SciFact assets. Use MODE=\"cached\" or restore the supplied resources folder.")
    if query_id not in ctx.test_ids:
        raise ValidationError('Choose an official test query ID as a string, e.g. "1303", "1012", or "628".')
    if type(severity) is not int or severity not in range(4):
        raise ValidationError("M1 severity must be an integer: 0, 1, 2, or 3.")
    runs = {}
    for condition, level in [("clean", 0), ("shifted", severity)]:
        if condition == "shifted" and severity == 0:
            runs[condition] = runs["clean"]
            continue
        dense = retrieve(ctx, [query_id], method="dense", top_k=20, severity=level)
        runs[condition] = {
            "BM25": retrieve(ctx, [query_id], method="bm25", top_k=20, severity=level)[query_id],
            "Dense": dense[query_id],
            "Dense → reranker": rerank(ctx, dense, top_k=20, severity=level)[query_id],
        }
    rel = ctx.qrels[query_id]
    rows = []
    for system in runs["clean"]:
        clean, shifted = runs["clean"][system], runs["shifted"][system]
        a, b = ndcg(clean, rel), ndcg(shifted, rel)
        rows.append({"System": system, "Clean nDCG@10": a, "Shifted nDCG@10": b,
                     "ΔnDCG@10": b-a, "Clean Recall@10": recall(clean, rel),
                     "Shifted Recall@10": recall(shifted, rel), "Finite RBO@10": rbo(clean, shifted)})
    edits = [] if severity == 0 else ctx.variants[str(severity)][query_id]["edits"]
    return {"query_id": query_id, "severity": severity, "edits": edits,
            "clean_text": ctx.queries[query_id], "shifted_text": query_text(ctx, query_id, severity),
            "runs": runs, "metrics": pd.DataFrame(rows),
            "positive_count": sum(grade > 0 for grade in rel.values()),
            "candidate_recall": {c: recall(r["Dense"], rel, 20) for c, r in runs.items()},
            "label_validity": "identity" if not edits else "same intended claim assumed; human review: none"}


def _table_style(frame, caption):
    """One wrapping style for visible results and reference tables."""
    return (frame.style.hide(axis="index").format(precision=3, escape="html")
            .set_caption(caption)
            .set_table_styles([
                {"selector": "caption", "props": [("text-align", "left"), ("font-size", "17px"), ("font-weight", "600")]},
                {"selector": "th", "props": [("font-size", "15px"), ("text-align", "left")]},
                {"selector": "td", "props": [("font-size", "16px"), ("text-align", "left"),
                                                  ("white-space", "normal"), ("max-width", "540px")]},
            ]))


def show_table(frame, caption, interpretation=None):
    """Compact, wrapping tables readable in Jupyter."""
    from IPython.display import HTML, display
    display(_table_style(frame, caption))
    if interpretation is not None:
        display(HTML(interpretation_html(interpretation)))


def show_baseline(ctx, result):
    from IPython.display import HTML, display
    q = result["query_id"]
    display(HTML(f'<p><b>Original query:</b> {escape(result["clean_text"])}</p>'))
    ranking = result["runs"]["clean"]["Dense → reranker"]
    rows = [{"Rank": i+1, "Document ID": d, "Logit": score,
             "Judgment": ("judged relevant" if ctx.qrels[q].get(d, 0) > 0 else
                          "judged nonrelevant" if d in ctx.qrels[q] else "unjudged"),
             "Title": ctx.corpus[d].split("\n")[0]} for i, (d, score) in enumerate(ranking[:5])]
    relevant = [row for row in rows if row["Judgment"] == "judged relevant"]
    total = sum(grade > 0 for grade in ctx.qrels[q].values())
    noun = "paper" if total == 1 else "papers"
    where = ", at rank " + str(relevant[0]["Rank"]) if len(relevant) == 1 else ""
    show_table(pd.DataFrame(rows), "Original query: first five reranked papers (one query, full collection)",
               f"These five references include {len(relevant)} of {total} known relevant {noun}{where}.")
    display(HTML(
        "<p>Rank 1 is first after reranking. Document ID identifies the paper. "
        "Logit is the model score for the query–paper pair. The cross encoder reads the claim and the "
        "paper's title and abstract together as one input of up to 512 tokens, and its final layer "
        "outputs one raw number. Higher ranks earlier. It is not a probability: the number is never "
        "converted to a 0–1 scale, so it can be negative, and only its order among the 20 candidates matters. "
        "Judgment is the qrel label. Title is the full paper title.</p>"
        "<p>Unjudged means this query–paper pair has no relevance label. "
        "It gets zero credit in evaluation; its relevance is unknown.</p>"))
    ranks = [row["Rank"] for row in relevant]
    if not total:
        reading = "This claim has no known relevant paper, so the table cannot show whether evidence reaches the LLM."
    elif len(relevant) == total == 1:
        reading = ("With the original query, the reranker places the only known relevant paper first, "
                   "so the LLM would read it as Reference 1." if ranks == [1] else
                   f"With the original query, the only known relevant paper is at rank {ranks[0]}, "
                   "so it would still reach a five-reference LLM prompt.")
    elif len(relevant) == total:
        reading = (f"With the original query, all {total} known relevant papers are among the first five, "
                   "so a five-reference LLM prompt would include all of the evidence.")
    elif relevant:
        reading = (f"With the original query, only {len(relevant)} of the {total} known relevant papers are "
                   "among the first five, so a five-reference LLM prompt would miss part of the evidence.")
    else:
        reading = ("Even with the original query, no known relevant paper is among the first five, "
                   "so a five-reference LLM prompt would miss the evidence.")
    show_interpretation(reading)


def show_prompt_interpretation(ctx, query_id, reference_ids):
    """Say which numbered references are known evidence; the qrels themselves stay out of the prompt."""
    positives = _positives(ctx, query_id)
    numbers = [n for n, doc_id in enumerate(reference_ids, 1) if doc_id in positives]
    papers = "this paper" if len(reference_ids) == 1 else f"these {len(reference_ids)} papers"
    if not positives:
        text = "This claim has no known relevant paper, so the prompt cannot be checked for evidence."
    elif not numbers:
        text = (f"The LLM sees only the claim, the instructions, and {papers}, and "
                + ("it is not" if len(reference_ids) == 1 else "none of them is")
                + " a known relevant paper, so the LLM would have to answer without the evidence.")
    else:
        where = (f"Reference {numbers[0]}" if len(numbers) == 1 else "References " + _join(numbers))
        who = ("the known relevant paper is" if len(positives) == 1 else
               f"all {len(positives)} known relevant papers are" if len(numbers) == len(positives) else
               f"{len(numbers)} of the {len(positives)} known relevant papers {'is' if len(numbers) == 1 else 'are'}")
        share = "the evidence is" if len(numbers) == len(positives) else "only part of the evidence is"
        text = (f"The LLM sees only the claim, the instructions, and {papers}; {who} {where}, "
                f"so {share} in its input.")
    show_interpretation(text)


def show_inputs(result):
    from IPython.display import HTML, display
    clean, shifted, start = [], [], 0
    for edit in sorted(result["edits"], key=lambda e: e["start"]):
        prefix = escape(result["clean_text"][start:edit["start"]])
        clean += [prefix, "<mark>"+escape(edit["before"])+"</mark>"]
        shifted += [prefix, "<mark>"+escape(edit["after"])+"</mark>"]
        start = edit["end"]
    suffix = escape(result["clean_text"][start:])
    clean.append(suffix); shifted.append(suffix)
    operations = "; ".join(f'{e["before"]} → {e["after"]} ({e["operation"]})' for e in result["edits"]) or "no edits; input unchanged"
    display(HTML(f'<div style="font-size:18px;line-height:1.5;overflow-wrap:anywhere">'
                 f'<p><b>Original:</b> {"".join(clean)}<br><b>After typos:</b> {"".join(shifted)}</p>'
                 f'<p>Maximum character edits: <b>{result["severity"]}</b>, actual edits: <b>{len(result["edits"])}</b><br>'
                 f'{escape(operations)}</p></div>'))


def show_comparison(result, clean_only=False):
    from IPython.display import HTML, display
    columns = (["System", "Clean nDCG@10", "Clean Recall@10"] if clean_only else
               ["System", "Clean nDCG@10", "Shifted nDCG@10", "ΔnDCG@10", "Finite RBO@10"])
    metrics = result["metrics"]
    if clean_only:
        if (metrics[["Clean nDCG@10", "Clean Recall@10"]] == 1).all().all():
            interpretation = "All three systems score 1 on nDCG@10 and Recall@10 for this original query."
        else:
            best = metrics["Clean nDCG@10"].max()
            leaders = ", ".join(metrics.loc[metrics["Clean nDCG@10"] == best, "System"])
            interpretation = f"{leaders} has the highest original-query nDCG@10 ({best:.3f})."
    else:
        changes = "; ".join(
            f"{row['System']} {row['Clean nDCG@10']:.3f} → {row['Shifted nDCG@10']:.3f}"
            for row in metrics.to_dict("records"))
        interpretation = f"Original → typo nDCG@10 for each system: {changes}."
    show_table(metrics[columns].rename(columns={
                   "Clean nDCG@10": "Original nDCG@10", "Shifted nDCG@10": "After-typo nDCG@10",
                   "Clean Recall@10": "Original Recall@10"}),
               f'Query: {result["clean_text"]} — one query, full collection, top 10 scored'
               + (", original input" if clean_only else f', RBO p=0.9, maximum/actual edits: {result["severity"]}/{len(result["edits"])}'),
               interpretation)
    ceilings = result["candidate_recall"]
    ceiling = f'original {ceilings["clean"]:.3f}'
    reranked = result["metrics"].set_index("System").loc["Dense → reranker"]
    achieved = f'original {reranked["Clean Recall@10"]:.3f}'
    if not clean_only:
        ceiling += f' → after typos {ceilings["shifted"]:.3f}'
        achieved += f' → after typos {reranked["Shifted Recall@10"]:.3f}'
    display(HTML(f'<p>Dense candidate Recall@20: {ceiling}; '
                 f'reranked Recall@10: {achieved} (both higher is better).</p>'))
    if not clean_only:
        clean = result["runs"]["clean"]["Dense"][:10]
        shifted = result["runs"]["shifted"]["Dense"][:10]
        changes = [(i, a[0], b[0]) for i, (a, b) in enumerate(zip(clean, shifted), 1) if a[0] != b[0]]
        trace = (f'rank {changes[0][0]}: {escape(changes[0][1])} → {escape(changes[0][2])} (original → after typos)'
                 if changes else "none; the ordered top-ten document IDs are identical")
        display(HTML(f'<p><b>First changed dense rank:</b> {trace}. '
                     'The paper table shows five results; the scores use ten.</p>'))
    show_interpretation(comparison_interpretation(result, clean_only))


def comparison_interpretation(result, clean_only=False):
    """Plain reading of the three-system table: where the typo hurt, and at which step."""
    scores = result["metrics"].set_index("System")
    if clean_only:
        if (scores["Clean nDCG@10"] >= 1-TIE).all():
            return ("Every system already ranks the known evidence as well as possible, "
                    "so any later loss on this claim comes from the typo.")
        best = scores["Clean nDCG@10"].max()
        leaders = [s for s in scores.index if scores.loc[s, "Clean nDCG@10"] >= best-TIE]
        missing = [s for s in scores.index if scores.loc[s, "Clean Recall@10"] < 1-TIE]
        what = "the known evidence" if result["positive_count"] == 1 else "some known evidence"
        found = (f"{_join(missing)} already miss{'es' if len(missing) == 1 else ''} {what} in the top ten"
                 if missing else "every system finds the known evidence in its top ten")
        return (f"Even without a typo, {found}, and {_join(leaders)} rank{'s' if len(leaders) == 1 else ''} it best "
                f"(nDCG@10 {best:.3f}), so compare each system's typo result with its own original score.")
    if not result["edits"]:
        return "With no edits, the typo query is the original query, so every score stays the same."
    delta = scores["ΔnDCG@10"]
    lower = [s for s in scores.index if delta[s] < -TIE]
    higher = [s for s in scores.index if delta[s] > TIE]
    same = [s for s in scores.index if s not in lower + higher]
    if not lower and not higher:
        if (scores["Clean nDCG@10"] <= TIE).all():
            return "No system finds the known evidence in its top ten, before or after the typo."
        return "The typo reshuffles some results, but every system keeps its nDCG@10, so this typo does not hurt this claim."
    cause = ""
    if "Dense → reranker" in lower:
        before, after = result["candidate_recall"]["clean"], result["candidate_recall"]["shifted"]
        reranked = scores.loc["Dense → reranker", "Shifted Recall@10"]
        if after <= TIE < before:
            cause = (", and because no known relevant paper is left among the 20 dense candidates, "
                     "the reranker has nothing to move up")
        elif after < before-TIE:
            cause = ", partly because some known relevant papers left the 20 dense candidates before reranking"
        elif reranked < after-TIE:
            cause = (", even though the evidence is still among the 20 candidates: "
                     "reranking places some of it below rank 10")
    if lower and not higher:
        start = ("The typo lowers nDCG@10 for all three systems" if not same else
                 f"The typo lowers nDCG@10 for {_join(lower)} and leaves {_join(same)} unchanged")
        return start + cause + "."
    if higher and not lower:
        return (f"The typo raises nDCG@10 for {_join(higher)}"
                + (f" and leaves {_join(same)} unchanged" if same else "")
                + ", so a typo does not always hurt.")
    return (f"The typo lowers nDCG@10 for {_join(lower)} but raises it for {_join(higher)}"
            + (f" and leaves {_join(same)} unchanged" if same else "")
            + (cause + "." if cause else ", so the same edit can hurt one system and help another."))


def reference_cases(ctx):
    """Fixed dense/BM25 snapshots, independent of current attendee controls.

    Read the original cache even if the optional live path is active. These are
    explicitly reference results, never presented as new live inference.
    """
    rows = []
    for q in ctx.demo_ids:
        rel = ctx.qrels[q]
        clean = ctx.runs["dense_s0"][q]
        for severity in range(4):
            shifted = ctx.runs[f"dense_s{severity}"][q]
            a, b = ndcg(clean, rel), ndcg(shifted, rel)
            rows.append({"Query": q, "Requested": severity,
                         "Actual": 0 if severity == 0 else len(ctx.variants[str(severity)][q]["edits"]),
                         "Clean nDCG@10": a, "Shifted nDCG@10": b, "ΔnDCG@10": b-a,
                         "Finite RBO@10": rbo(clean, shifted),
                         "Candidate Recall@20": recall(shifted, rel, 20),
                         "Clean BM25 nDCG@10": ndcg(ctx.runs["bm25_s0"][q], rel),
                         "Shifted BM25 nDCG@10": ndcg(ctx.runs[f"bm25_s{severity}"][q], rel)})
    return pd.DataFrame(rows)


def show_reference_cases(ctx):
    """Laptop-free fixed results, always visible."""
    from IPython.display import HTML, display
    frame = reference_cases(ctx)
    blocks = ['<section><h4>Reference results (fixed cached results)</h4>',
              '<p>Dense metrics plus a labeled BM25 column, full corpus, n=1 per row, evaluation cutoff=10, finite RBO p=0.9. '
              'The queries were selected to illustrate a loss, an unchanged score, and a gain after one synthetic typo. '
              'This reference stays fixed after edits to Experiment Settings or MODE.</p>']
    for q in ctx.demo_ids:
        subset = frame[frame["Query"] == q].drop(columns=["Query", "Clean nDCG@10", "Clean BM25 nDCG@10"])
        subset = subset.rename(columns={"Requested": "Maximum edits", "Actual": "Actual edits",
                                        "Shifted nDCG@10": "Dense nDCG@10",
                                        "ΔnDCG@10": "Dense ΔnDCG@10", "Shifted BM25 nDCG@10": "BM25 nDCG@10"})
        clean_score = frame.loc[frame["Query"] == q, "Clean nDCG@10"].iloc[0]
        bm25_score = frame.loc[frame["Query"] == q, "Clean BM25 nDCG@10"].iloc[0]
        blocks.append(f'<p><b>Test query:</b> {escape(ctx.queries[q])}</p>')
        blocks.append(_table_style(subset, f'Clean nDCG@10: dense {clean_score:.3f}, BM25 {bm25_score:.3f}; all rows below are shifted').to_html())
        single = frame[(frame["Query"] == q) & (frame["Requested"] == 1)].iloc[0]
        last = frame[(frame["Query"] == q) & (frame["Requested"] == 3)].iloc[0]
        blocks.append(interpretation_html(
            f"Dense nDCG@10 changes from {clean_score:.3f} to {single['Shifted nDCG@10']:.3f} "
            f"at one edit and to {last['Shifted nDCG@10']:.3f} at the maximum-three-edit setting."))
        blocks.append('<p><b>Query text at each edit setting:</b><br>' + '<br>'.join(
            f'severity={s}: {escape(query_text(ctx, q, s))}' for s in range(1, 4)) + '</p>')
    blocks.append('</section>')
    display(HTML(''.join(blocks)))
    return frame


def show_rankings(ctx, result, depth=5):
    """Dense-only clean and shifted top five side by side; never infer a mechanism from score movement."""
    from IPython.display import HTML, display
    q = result["query_id"]
    metric = result["metrics"].set_index("System").loc["Dense"]
    edits = len(result["edits"])

    def document(doc_id):
        status = ("judged relevant" if ctx.qrels[q].get(doc_id, 0) > 0 else
                  "judged nonrelevant" if doc_id in ctx.qrels[q] else "unjudged")
        title = textwrap.shorten(ctx.corpus[doc_id].split("\n")[0], width=60, placeholder="…")
        return f"{doc_id} ({status}): {title}"

    clean, shifted = (result["runs"][c]["Dense"][:depth] for c in ("clean", "shifted"))
    rows = [{"Rank": i, "Clean query": document(a), "Clean cosine": a_score,
             "Shifted query": document(b), "Shifted cosine": b_score}
            for i, ((a, a_score), (b, b_score)) in enumerate(zip(clean, shifted), 1)]
    heading = ("Ranked papers before and after one character edit" if edits == 1 else
               "Identity control: no character edited" if edits == 0 else
               f"Ranking after {edits} character edits")
    relevant_counts = [sum(ctx.qrels[q].get(doc_id, 0) > 0 for doc_id, _ in ranking)
                       for ranking in (clean, shifted)]
    show_table(pd.DataFrame(rows).rename(columns={
                   "Clean query": "Original query", "Shifted query": "After typos",
                   "Clean cosine": "Original cosine", "Shifted cosine": "Typo cosine"}),
               f"{heading}: {result['clean_text']} — dense top five, one query",
               f"Known relevant papers in these five results: "
               f"{relevant_counts[0]} before edits → {relevant_counts[1]} after edits.")
    display(HTML(f'<p>Dense nDCG@10: original {metric["Clean nDCG@10"]:.3f} → after typos {metric["Shifted nDCG@10"]:.3f} '
                 f'(Δ {metric["ΔnDCG@10"]:+.3f}); Recall@10: original {metric["Clean Recall@10"]:.3f} → '
                 f'after typos {metric["Shifted Recall@10"]:.3f}; finite RBO@10 (p=0.9): {metric["Finite RBO@10"]:.3f}<br>'
                 f'<small>{len(ctx.corpus):,} documents, {ctx.mode} mode, original SciFact relevance judgments</small></p>'))
    show_interpretation(rankings_interpretation(ctx, result))


def rankings_interpretation(ctx, result):
    """Plain reading of the dense before/after lists: how much changed, and what happened to the evidence."""
    if not result["edits"]:
        return "With no edits, both inputs are the same, so the dense results and their scores are identical."
    metric = result["metrics"].set_index("System").loc["Dense"]
    before, after = metric["Clean nDCG@10"], metric["Shifted nDCG@10"]
    clean, shifted = (result["runs"][c]["Dense"] for c in ("clean", "shifted"))
    kept = len({d for d, _ in clean[:10]} & {d for d, _ in shifted[:10]})
    positives = _positives(ctx, result["query_id"])
    ranks_before, ranks_after = _relevant_ranks(clean, positives), _relevant_ranks(shifted, positives)
    one = len(positives) == 1
    paper, s, is_ = (("the known relevant paper", "s", "is") if one else ("the known relevant papers", "", "are"))
    start = f"After {_edit_phrase(result['edits'])}, "
    scores = (f"nDCG@10 falls from {before:.3f} to {after:.3f}" if after < before-TIE else
              f"nDCG@10 rises from {before:.3f} to {after:.3f}" if after > before+TIE else
              f"nDCG@10 stays at {before:.3f}")
    if not kept:
        lost = ranks_before and not ranks_after
        return (start + "none of the 10 original papers stays in the dense top ten"
                + (f", including {paper}" if lost else "") + f", so {scores}.")
    start += (f"{'only ' if kept <= 5 else ''}{kept} of the 10 original papers "
              f"{'stays' if kept == 1 else 'stay'} in the dense top ten")
    if ranks_before and not ranks_after:
        return f"{start}, and {paper} {is_} not among them, so {scores}."
    if not ranks_before and not ranks_after:
        return f"{start}, but {paper} {is_} missing from the top ten both before and after the typo, so {scores}."
    if not ranks_before:
        return f"{start}, and {paper} now enter{s} the top ten, so {scores}."
    if one and ranks_before[0] != ranks_after[0]:
        direction = "down" if ranks_after[0] > ranks_before[0] else "up"
        joiner = "but" if direction == "down" else "and"
        return (f"{start}, {joiner} {paper} moves {direction} from rank {ranks_before[0]} "
                f"to rank {ranks_after[0]}, so {scores}.")
    if one:
        return f"{start}, and {paper} keeps rank {ranks_before[0]}, so {scores}."
    return f"{start}, and {scores}."


def run_lab(ctx, query_id, severity):
    """One rerunnable cell: explicit controls, fresh results, no earlier demo state."""
    result = observe(ctx, query_id, severity)
    show_inputs(result)
    show_rankings(ctx, result)
    show_comparison(result)
    return result
