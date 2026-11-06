"""Small M2 teaching views over the existing verified SciFact cache.

No model inference or corpus mutation. Every runner derives its results from
explicit controls, so settings can be changed without rerunning earlier
lecture cells.
"""
from html import escape
import math

import numpy as np
import pandas as pd

from .config import ValidationError
from .metrics import ndcg
from .plotting import COLORS, square_axes, titles
from .retrieval import query_text
from .presentation import interpretation_html, plain_html, show_interpretation

EXPERIMENTS = ("typo", "synthetic_spam")
NAMES = {"1303": "Tirasemtiv", "1012": "Radioiodine", "628": "HTLV-1"}
TIE = 1e-12  # Differences this small count as equal.


def validate_settings(ctx, query_id, perturbation="typo", severity=1):
    if ctx.mode != "cached":
        raise ValidationError("M2 uses verified cached CPU results. Keep MODE='cached'.")
    if query_id not in ctx.demo_ids:
        raise ValidationError(f"M2 query_id must be one of these cached strings: {ctx.demo_ids}.")
    if perturbation not in EXPERIMENTS:
        raise ValidationError("M2 perturbation must be 'typo' or 'synthetic_spam'.")
    if type(severity) is not int or severity not in range(4):
        raise ValidationError("M2 severity must be an integer from 0 to 3; it affects typos only.")


def _ids(ranking, k=20):
    return [d for d, _ in ranking[:k]]


def _style(frame, caption):
    return (frame.style.hide(axis="index").format(precision=3, escape="html", na_rep="—")
            .set_caption(caption).set_table_styles([
                {"selector": "caption", "props": [("text-align", "left"), ("font-size", "18px"), ("font-weight", "600")]},
                {"selector": "th", "props": [("text-align", "left"), ("font-size", "16px"), ("white-space", "normal")]},
                {"selector": "td", "props": [("text-align", "left"), ("font-size", "17px"), ("padding", "5px 10px")]},
            ]))


def _table(frame, caption, interpretation):
    from IPython.display import HTML, display
    styled = _style(frame, caption)
    display(HTML(plain_html(styled.to_html()) + interpretation_html(interpretation)))


def _note(text):
    from IPython.display import HTML, display
    display(HTML(plain_html(f'<p>{text}</p>')))


def ranking_table(ctx, query_id, conditions, injected_id=None):
    """Show complete top-ten ID order; keep long titles out of the main view."""
    def label(doc_id):
        if doc_id == injected_id:
            return "Inserted"
        return doc_id + (" [relevant]" if ctx.qrels[query_id].get(doc_id, 0) > 0 else "")
    return pd.DataFrame({"Rank": range(1, 11), **{
        name: [label(d) for d in _ids(ranking, 10)] for name, ranking in conditions.items()
    }})


def query_shift(ctx, query_id, severity=1):
    """Compare cached query vectors only; no retrieval or relevance evaluation."""
    validate_settings(ctx, query_id, "typo", severity)
    i = ctx.indices([query_id])[0]
    a, b = (ctx.embeddings[f"q_s{s}"][i].astype(float) for s in (0, severity))
    cosine = float(np.clip(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
    return {"query_id": query_id, "family": "typo", "severity": severity,
            "clean_text": ctx.queries[query_id], "shifted_text": query_text(ctx, query_id, severity),
            "edits": [] if severity == 0 else ctx.variants[str(severity)][query_id]["edits"],
            "query_distance": max(0., 1-cosine)}


def show_query_shift(ctx, query_id, severity=1):
    result = query_shift(ctx, query_id, severity)
    original, typo, start = [], [], 0
    for edit in sorted(result["edits"], key=lambda e: e["start"]):
        prefix = escape(result["clean_text"][start:edit["start"]])
        original += [prefix, escape(edit["before"])]
        typo += [prefix, escape(edit["after"])]
        start = edit["end"]
    suffix = escape(result["clean_text"][start:])
    original.append(suffix)
    typo.append(suffix)
    interpretation = ("The typo changed the query vector's direction."
                      if result["query_distance"] > 1e-12 else
                      "The query vector's direction is unchanged.")
    changes = "; ".join(f'{escape(e["before"])} → {escape(e["after"])}' for e in result["edits"]) or "none"
    _note(f'<b>Selected test query, maximum character edits: {severity}, actual edits: {len(result["edits"])}</b><br>'
          f'<b>Original Query:</b> {"".join(original)}<br>'
          f'<b>Typo Query:</b> {"".join(typo)}<br>'
          f'Edited words: {changes}<br>'
          f'<b>Embedding distance (1 − cosine similarity): {result["query_distance"]:.3f}</b><br>'
          f'{interpretation}')
    count = len(result["edits"])
    if result["query_distance"] <= TIE:
        show_interpretation("With no edits, the query vector does not move, so dense search sees exactly the original query.")
    else:
        edits = ("Even one character edit" if count == 1 else
                 f"{ {2: 'Two', 3: 'Three'}.get(count, count)} character edits")
        show_interpretation(f"{edits} turn{'s' if count == 1 else ''} the query vector away from the original "
                            f"(distance {result['query_distance']:.3f}), which can change which papers are closest "
                            "to it in dense search.")
    return result


def embedding_figure(ctx, query_id, severity=1):
    """M2 distance-only view: 1 − cosine similarity for each teaching query."""
    validate_settings(ctx, query_id, "typo", severity)
    distances = [query_shift(ctx, q, severity)["query_distance"] for q in ctx.demo_ids]
    # One y-scale for every cached edit setting, so bars stay comparable after a settings change.
    largest = max(query_shift(ctx, q, s)["query_distance"] for q in ctx.demo_ids for s in range(4))
    fig, ax = square_axes()
    x = np.arange(len(distances))
    bars = ax.bar(x, distances, width=.5, color=COLORS["clean"])
    ax.bar_label(bars, labels=[f"{d:.3f}" for d in distances], padding=4)
    labels = {"1303": "Tirasemtiv", "1012": "Radioiodine", "628": "HTLV-1"}
    ax.set(xticks=x, xticklabels=[labels[q] for q in ctx.demo_ids], xlim=(-.6, len(x)-.4),
           ylim=(0, math.ceil(largest*1.15/.05)*.05), ylabel="1 − cosine similarity")
    titles(ax, "Query vector change after typos")
    return fig


def embedding_interpretation(ctx, query_id, severity=1):
    """Set each bar beside the query's dense nDCG@10 change: distance is not search quality."""
    validate_settings(ctx, query_id, "typo", severity)
    if severity == 0:
        return "With no edits, every distance is zero, because each typo query is identical to its original query."
    distance = {q: query_shift(ctx, q, severity)["query_distance"] for q in ctx.demo_ids}
    scores = {q: (ndcg(ctx.runs["dense_s0"][q], ctx.qrels[q]), ndcg(ctx.runs[f"dense_s{severity}"][q], ctx.qrels[q]))
              for q in ctx.demo_ids}

    def outcome(q):
        before, after = scores[q]
        if after < before-TIE:
            return ("loses every known relevant paper from its dense top ten" if after <= TIE else
                    f"loses {before-after:.3f} dense nDCG@10")
        return f"gains {after-before:.3f} dense nDCG@10" if after > before+TIE else "keeps its dense nDCG@10"

    far = max(ctx.demo_ids, key=lambda q: distance[q])
    worst = min(ctx.demo_ids, key=lambda q: scores[q][1]-scores[q][0])
    if scores[worst][1] >= scores[worst][0]-TIE:
        return ("All three vectors move, yet none of these claims loses dense nDCG@10, "
                "so a moved vector does not by itself mean lost evidence.")
    if far != worst:
        return (f"{NAMES[far]} moves the most ({distance[far]:.3f}) yet {outcome(far)}, while {NAMES[worst]} "
                f"moves less ({distance[worst]:.3f}) and {outcome(worst)}, so a longer bar does not mean a larger loss.")
    safe = [q for q in ctx.demo_ids if scores[q][1] >= scores[q][0]-TIE]
    if not safe:
        return (f"{NAMES[far]} moves the most ({distance[far]:.3f}) and also loses the most, but every claim loses "
                "some score here, so check qrels rather than the bar length.")
    other = max(safe, key=lambda q: distance[q])
    return (f"{NAMES[far]} moves the most ({distance[far]:.3f}) and {outcome(far)}, but {NAMES[other]} moves "
            f"{distance[other]:.3f} and {outcome(other)}, so the bar length alone cannot tell which query fails.")


def show_embedding_interpretation(ctx, query_id, severity=1):
    show_interpretation(embedding_interpretation(ctx, query_id, severity))


def show_candidate_check():
    """A labeled toy example of set membership, not a SciFact experiment."""
    result = {"corpus": ("Relevant_Evidence", "Background_Overview", "Related_Experiment", "General_Review"),
              "relevant_id": "Relevant_Evidence",
              "candidates": ("Background_Overview", "Related_Experiment", "General_Review"),
              "reranked": ("General_Review", "Background_Overview", "Related_Experiment")}
    _note('<b>Toy example: illustrative document labels and order</b><br>'
          'Full corpus: Relevant Evidence (relevant), Background Overview, Related Experiment, General Review<br>'
          'Dense Retrieval selects: Background Overview, Related Experiment, General Review<br>'
          'Cross-Encoder reranks: General Review, Background Overview, Related Experiment<br>'
          'Relevant Evidence is absent from the candidates, so the reranker cannot include it in the final ranking.')
    return result


def exposure_metrics(clean, shifted, injected_id, k=10):
    """One-source exposure fraction, hit, rank, and original-ID displacement."""
    clean_ids = [d for d, _ in clean[:k]]
    ids = [d for d, _ in shifted[:k]]
    rank = next((i for i, (d, _) in enumerate(shifted, 1) if d == injected_id), None)
    return {"injected_rank": rank, "hit": int(injected_id in ids),
            "exposure": ids.count(injected_id)/k,
            "displaced_original_ids": len(set(clean_ids)-set(ids))}


def document_insertion(ctx, query_id):
    """Compare original and enlarged-corpus rankings from the verified cache."""
    validate_settings(ctx, query_id, "synthetic_spam")
    scenario = ctx.scenarios[query_id]["synthetic_spam"]
    rankings = {
        "Dense": {"Original": scenario["clean"], "After insertion": scenario["shifted"]},
        "Reranker": {"Original": ctx.runs["rerank_s0"][query_id], "After insertion": scenario["reranked"]},
    }
    injected_id = scenario["injected_id"]
    rows, displaced = [], {}
    for stage, conditions in rankings.items():
        original = conditions["Original"]
        inserted = conditions["After insertion"]
        displaced[stage] = [d for d in _ids(original, 10) if d not in _ids(inserted, 10)]
        for label, ranking in conditions.items():
            metrics = exposure_metrics(original, ranking, injected_id)
            rows.append({"Stage": stage, "Corpus": label,
                         "Inserted rank / 20": str(metrics["injected_rank"] or "absent"), "Hit@10": metrics["hit"]})
    return {"query_id": query_id, "family": "synthetic_spam", "severity": None,
            "injected_id": injected_id, "injected_text": scenario["injected_text"],
            "rankings": rankings, "insertion_metrics": pd.DataFrame(rows),
            "displaced_ids": displaced,
            "survives_reranking": injected_id in _ids(rankings["Reranker"]["After insertion"], 10)}


def show_insertion(ctx, query_id):
    result = document_insertion(ctx, query_id)
    _note('<b>Selected test query, one added document</b><br>'
          f'{escape(ctx.queries[query_id])}<br><b>Inserted document (cached experiment input):</b> '
          f'<code>{escape(result["injected_text"])}</code>')
    inserted = result["insertion_metrics"].query("Corpus == 'After insertion'").set_index("Stage")
    survived = "yes" if result["survives_reranking"] else "no"
    _table(result["insertion_metrics"], "Inserted document rank and attack Hit@10 (0 is safer)",
           f"The attack ranks {inserted.loc['Dense', 'Inserted rank / 20']} in dense search "
           f"and {inserted.loc['Reranker', 'Inserted rank / 20']} after reranking. "
           f"Still in the reranked top ten: {survived}.")
    show_interpretation(_insertion_interpretation(inserted))
    for stage, conditions in result["rankings"].items():
        relevant_ranks = {
            condition: ", ".join(str(rank) for rank, (doc_id, _) in enumerate(ranking[:10], 1)
                                 if ctx.qrels[query_id].get(doc_id, 0) > 0) or "none"
            for condition, ranking in conditions.items()}
        _table(ranking_table(ctx, query_id, conditions, result["injected_id"]),
               f"{stage} top-10 (Inserted = attack; [relevant] = positive qrel; rank 1 is highest)",
               f"Known relevant papers rank {relevant_ranks['Original']} before insertion and "
               f"{relevant_ranks['After insertion']} after insertion. "
               f"Original papers pushed out of the top ten: {len(result['displaced_ids'][stage])}.")
        show_interpretation(_top_ten_interpretation(ctx, query_id, stage, result))
    displaced = "; ".join(f'{s}: {", ".join(ids) or "none"}' for s, ids in result["displaced_ids"].items())
    _note(f'<b>Original papers pushed out of the top ten (IDs):</b> {escape(displaced)}')
    return result


def _rank(value):
    return None if value == "absent" else int(value)


def _insertion_interpretation(inserted):
    """Did reranking catch the inserted passage?"""
    dense, reranked = (_rank(inserted.loc[stage, "Inserted rank / 20"]) for stage in ("Dense", "Reranker"))
    if dense is None:
        return "The attack does not reach the 20 dense candidates, so it can reach neither the reranker nor the LLM."
    if reranked == 1:
        return ("A passage that only repeats the claim outranks every real paper in dense search and after "
                "reranking, so reranking does not stop this attack." if dense == 1 else
                f"The attack enters at dense rank {dense} and reranking moves it to first place, "
                "so reranking makes this attack stronger.")
    if reranked <= 10:
        return (f"Reranking moves the attack from dense rank {dense} to rank {reranked}, but it stays in the "
                "top ten, so reranking alone does not remove it.")
    return f"Reranking pushes the attack from dense rank {dense} down to rank {reranked}, out of the top ten."


def _evidence_move(before, after):
    if not before and not after:
        return "no known relevant paper is in the top ten before or after insertion"
    if before and not after:
        return "the known evidence drops out of the top ten"
    if len(before) == len(after) == 1:
        if after[0] > before[0]:
            return f"the relevant paper slips from rank {before[0]} to {after[0]}"
        if after[0] < before[0]:
            return f"the relevant paper moves up from rank {before[0]} to {after[0]}"
        return f"the relevant paper keeps rank {before[0]}"
    return (f"known relevant papers move from ranks {', '.join(map(str, before)) or 'none'} "
            f"to {', '.join(map(str, after)) or 'none'}")


def _top_ten_interpretation(ctx, query_id, stage, result):
    """What the inserted passage did to one stage's top ten."""
    conditions = result["rankings"][stage]
    ranks = {condition: [rank for rank, (d, _) in enumerate(ranking[:10], 1) if ctx.qrels[query_id].get(d, 0) > 0]
             for condition, ranking in conditions.items()}
    attack = next((rank for rank, (d, _) in enumerate(conditions["After insertion"][:10], 1)
                   if d == result["injected_id"]), None)
    evidence = _evidence_move(ranks["Original"], ranks["After insertion"])
    if stage == "Dense":
        pushed = len(result["displaced_ids"][stage])
        lead = (f"In dense search, the attack takes rank {attack}" if attack else
                "In dense search, the attack stays out of the top ten")
        out = ("no original paper leaves the top ten" if not pushed else
               "one original paper is pushed out of the top ten" if pushed == 1 else
               f"{pushed} original papers are pushed out of the top ten")
        return f"{lead} and {evidence}; {out}."
    dense = _rank(result["insertion_metrics"].query("Corpus == 'After insertion'")
                  .set_index("Stage").loc["Dense", "Inserted rank / 20"])
    if attack == 1:
        lead = "Reranking keeps the attack at rank 1" if dense == 1 else "After reranking, the attack takes rank 1"
        return f"{lead} and {evidence}, so an LLM would read the attack before any real paper."
    if attack:
        return f"After reranking, the attack takes rank {attack} and {evidence}, so it still reaches the top ten."
    return f"After reranking, the attack is out of the top ten and {evidence}."


def run_experiment(ctx, query_id, perturbation="typo", severity=1):
    """Single hands-on runner; earlier demo variables are never read."""
    validate_settings(ctx, query_id, perturbation, severity)
    if perturbation == "synthetic_spam":
        return show_insertion(ctx, query_id)
    return show_query_shift(ctx, query_id, severity)
