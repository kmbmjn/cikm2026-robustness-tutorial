"""M4's validation-first experiment and presentation over immutable CPU assets.

This module keeps training noise separate from evaluation noise and compares
methods on paired query IDs, with explicit guards against stale interactive
selection state.
"""
import html

import numpy as np
import pandas as pd

from .config import ValidationError, fingerprint
from .interventions import (build_vocabulary, input_normalization, select_adapter,
                            vocab_spell_correction)
from .metrics import evaluate_suite, ndcg, paired_bootstrap
from .perturbations import TOKEN
from .plotting import COLORS, legend_below, show, square_axes, titles
from .presentation import interpretation_html, show_interpretation
from .retrieval import BM25, query_text, retrieve

NAMES = {"none": "Dense baseline", "input_normalization": "Normalize",
         "vocab_spell_correction": "Vocabulary", "repair_adapter": "Adapter"}
LESSON_NAMES = {"none": "Original search", "input_normalization": "Input normalization",
                "vocab_spell_correction": "Vocabulary correction", "repair_adapter": "Repair adapter"}
SUBJECTS = {"input_normalization": "normalization", "vocab_spell_correction": "correction",
            "repair_adapter": "the adapter"}
IN_SENTENCE = {"none": "original search", "input_normalization": "input normalization",
               "vocab_spell_correction": "vocabulary correction", "repair_adapter": "the repair adapter"}
TIE = 1e-12  # nDCG@10 differences this small count as equal scores.


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else \
        ", ".join(items[:-1]) + ", and " + items[-1]


def _rank_move(before, after):
    """How the known relevant papers moved within the 20 results."""
    if before and after:
        if len(before) == len(after) == 1:
            if after[0] > before[0]:
                return f"moves the relevant paper down from rank {before[0]} to rank {after[0]}"
            if after[0] < before[0]:
                return f"moves the relevant paper up from rank {before[0]} to rank {after[0]}"
            return f"keeps the relevant paper at rank {before[0]}"
        return (f"moves the known relevant papers from ranks {', '.join(map(str, before))} "
                f"to ranks {', '.join(map(str, after))}")
    if before:
        return "pushes the known evidence out of the top 20"
    return "brings known evidence into the top 20" if after else "leaves the known evidence outside the top 20"


def split_audit(ctx):
    """Purge cross-split duplicate text without reading qrels or changing assets.

    Test has priority over validation, which has priority over train. The key
    uses NFKC, casefold and collapsed whitespace; it is not semantic deduplication.
    """
    kept, excluded, higher = {}, [], {}
    for split, ids in [("test", ctx.test_ids), ("validation", ctx.val_ids), ("train", ctx.train_ids)]:
        kept[split] = []
        for q in ids:
            key = input_normalization(ctx.queries[q])
            if key in higher:
                other_split, other_id = higher[key]
                excluded.append({"Split": split, "Excluded ID": q,
                                 "Matches split": other_split, "Retained ID": other_id})
            else:
                kept[split].append(q)
        # Do not change the official test denominator or silently deduplicate
        # within a split. This audit addresses overlap BETWEEN splits only.
        higher.update({input_normalization(ctx.queries[q]): (split, q) for q in kept[split]})
    return {"policy": "NFKC/casefold/whitespace; test > validation > train",
            "ids": kept, "excluded": excluded}


def candidate_signature(ctx, config, audit):
    return fingerprint({**config, "source_splits": ctx.manifest["splits"], "split_audit": audit})


def configuration_label(candidate):
    c, s = candidate["config"], candidate["selection"]
    return (f"train n={len(s['fit_query_ids'])}, noise={c['train_noise_severity']}, seed={c['seed']}; "
            f"selected λ={s['lambda']:g}; validation/test noise={c['severity']}, seed={c['validation_seed']}")


def controls(ctx, severity=1, lambdas=(.1, 1., 10., 100.), train_noise_severity=1, seed=42):
    if ctx.mode != "cached":
        raise ValidationError("M4 uses verified cached embeddings. Keep MODE='cached'; see README for optional live APIs.")
    if type(severity) is not int or severity not in range(4):
        raise ValidationError("M4 severity must be 0, 1, 2, or 3; zero is the identity control.")
    if type(train_noise_severity) is not int or train_noise_severity not in (1, 2, 3):
        raise ValidationError("train_noise_severity must be 1, 2, or 3.")
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValidationError("training_seed must be 42, 43, or 44.")
    try:
        grid = tuple(float(v) for v in lambdas)
    except (TypeError, ValueError) as error:
        raise ValidationError("lambda_grid must contain distinct positive finite numbers.") from error
    if not grid or len(set(grid)) != len(grid) or not all(np.isfinite(v) and v > 0 for v in grid):
        raise ValidationError("lambda_grid must contain distinct positive finite numbers.")
    return {"severity": severity, "lambda_grid": grid, "train_noise_severity": train_noise_severity,
            "seed": seed, "validation_seed": ctx.seed}


def select_candidate(ctx, severity=1, lambdas=(.1, 1., 10., 100.), train_noise_severity=1, seed=42):
    config = controls(ctx, severity, lambdas, train_noise_severity, seed)
    audit = split_audit(ctx)
    selection = select_adapter(ctx, lambdas=config["lambda_grid"], train_noise_severity=train_noise_severity,
                               seed=seed, validation_severity=severity,
                               train_ids=audit["ids"]["train"], validation_ids=audit["ids"]["validation"])
    losses = {}
    for split, key in [("train", f"train_s{train_noise_severity}_seed{seed}"),
                       ("validation", f"q_s{severity}")]:
        indices = ctx.indices(audit["ids"][split])
        diff = ctx.embeddings[key][indices].astype(float) - ctx.embeddings["q_s0"][indices]
        losses[f"{split}_mse"] = float(np.mean(diff**2))
    return {"config": config, "selection": selection, "split_audit": audit, "identity_losses": losses,
            "signature": candidate_signature(ctx, config, audit)}


def table(frame, caption, precision=4, interpretation=None):
    from IPython.display import HTML, display
    display(frame.style.hide(axis="index").format(precision=precision, na_rep="not measured", escape="html")
            .set_caption(caption).set_table_styles([
                {"selector": "caption", "props": [("text-align", "left"), ("font-size", "18px"), ("font-weight", "600")]},
                {"selector": "th, td", "props": [("font-size", "16px"), ("text-align", "left"),
                                                   ("white-space", "normal"), ("max-width", "220px")]},
            ]))
    if interpretation is not None:
        display(HTML(interpretation_html(interpretation)))


def show_inputs(ctx, severity=1, query_id="1303"):
    controls(ctx, severity)
    from IPython.display import display, Markdown
    vocab = build_vocabulary(ctx.corpus)
    text = query_text(ctx, query_id, severity)
    rows = [("Original query", ctx.queries[query_id]), ("After typos", text),
            ("After normalization", input_normalization(text)),
            ("After vocabulary correction", vocab_spell_correction(text, vocab))]
    display(Markdown(f"**Selected test query, maximum edits: {severity}**\n\n" +
                     "\n\n".join(f"**{label}:** {html.escape(value)}" for label, value in rows)))
    original, normalized, corrected = rows[0][1], rows[2][1], rows[3][1]
    how = ("only lowercases the text" if normalized == text.casefold() else
           "only standardizes letter forms, case, and spacing")
    if text == original:
        show_interpretation("With no typos there is nothing to repair: normalization " + how.replace("text", "claim")
                            + ", and vocabulary correction "
                            + ("leaves the claim unchanged." if corrected == original else
                               "still changes it, which can alter a correctly written claim."))
    else:
        tokens = TOKEN.findall(normalized.casefold())
        kept = [e["after"] for e in sorted(ctx.variants[str(severity)][query_id]["edits"], key=lambda e: e["start"])
                if e["after"].casefold() in tokens]
        lead = ("Only vocabulary correction restores the original claim" if corrected == original else
                "Vocabulary correction changes the typo text but does not restore the original claim"
                if corrected != text else "Neither repair restores the original claim")
        show_interpretation(f"{lead}; normalization {how}" + (f" and keeps {_join(kept)}." if kept else "."))
    return dict(rows)


def selection_figure(candidate):
    f = candidate["selection"]["validation"].sort_values("lambda")
    fig, axes = square_axes(2)
    selected = candidate["selection"]["lambda"]
    for ax, split in zip(axes, ("train", "validation")):
        # Plot in units of 1e-4 so no axis multiplier sits above the plot.
        ax.semilogx(f["lambda"], f[f"{split}_mse"]*1e4, "o-", color=COLORS["clean"], label="Fitted adapter",
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.axhline(candidate["identity_losses"][f"{split}_mse"]*1e4, color=COLORS["muted"],
                   linestyle="--", linewidth=1.5, label="A=0 (no repair)")
        ax.axvline(selected, color=COLORS["axis"], linewidth=1, zorder=1, label="λ selected on validation")
        ax.set(xlabel="Ridge λ (sum-of-squares convention)")
        titles(ax, f"{split.title()} alignment MSE")
    axes[0].set_ylabel("Squared error per coordinate (×10⁻⁴)")
    legend_below(fig, axes[0])
    return fig


def show_selection(ctx, candidate):
    from IPython.display import display, Markdown
    s, c = candidate["selection"], candidate["config"]
    display(Markdown(f"**After duplicate-text exclusion — train / validation / test:** "
                     f"{len(s['fit_query_ids'])} / {len(s['validation_query_ids'])} / {len(ctx.test_ids)}. "
                     f"**Train noise / seed:** {c['train_noise_severity']} / {c['seed']}. "
                     f"**Fixed validation noise / seed:** {c['severity']} / {ctx.seed}. "
                     f"**Selected λ = {s['lambda']:g}**, by shifted nDCG, then clean nDCG, then smaller λ."))
    table(s["validation"].rename(columns={"lambda": "λ", "validation_clean_ndcg": "Val clean nDCG@10",
          "validation_shifted_ndcg": "Val shifted nDCG@10", "train_mse": "Train MSE",
          "validation_mse": "Val MSE", "regularized_objective": "Train SSE + penalty"}),
          "Validation-only selection: " + configuration_label(candidate) +
          "; higher nDCG is better; lower MSE means closer vector alignment", precision=5,
          interpretation=f"Validation selects λ={s['lambda']:g} by retrieval quality, "
                         "not by the smallest alignment error.")
    loss = candidate["identity_losses"]
    display(Markdown(f"**A=0 alignment control:** train MSE {loss['train_mse']:.6f}; "
                     f"validation MSE {loss['validation_mse']:.6f}. Choosing the best adapter λ does not require beating no repair."))
    show(selection_figure(candidate))


def compare(ctx, candidate, split="test"):
    if split not in {"validation", "test"}:
        raise ValidationError("Use the validation or test split.")
    c, s = candidate["config"], candidate["selection"]
    controls(ctx, c["severity"], c["lambda_grid"], c["train_noise_severity"], c["seed"])
    ids = candidate["selection"]["validation_query_ids"] if split == "validation" else ctx.test_ids
    evaluations, rows = {}, []
    for name in NAMES:
        runs = [retrieve(ctx, queries=ids, severity=level, intervention=name, adapter=s["adapter"])
                for level in (0, c["severity"])]
        ev = evaluate_suite(ctx, *runs, query_ids=ids)
        evaluations[name] = ev
        v = ev["summary"]
        rows.append({"Method": NAMES[name], "Clean nDCG@10": v["clean_ndcg"], "Shifted nDCG@10": v["shifted_ndcg"],
                     "Bottom shifted 10%": v["bottom_shifted_ndcg"], "Worst 10% Δ": v["worst_paired_delta"],
                     "Finite RBO@10": v["rbo"]})
    # BM25 is context, not a fifth dense intervention. Validation runs are computed
    # from the same fixed texts; test rankings are the supplied exact lexical runs.
    if split == "test":
        bm_runs = [retrieve(ctx, method="bm25", severity=level) for level in (0, c["severity"])]
    else:
        bm = BM25(ctx.corpus)
        bm_runs = [{q: bm.retrieve(query_text(ctx, q, level), 20) for q in ids} for level in (0, c["severity"])]
    bm_ev = evaluate_suite(ctx, *bm_runs, query_ids=ids)
    return {"candidate": candidate, "split": split, "query_ids": list(ids), "evaluations": evaluations,
            "table": pd.DataFrame(rows), "bm25": bm_ev}


def prepare_lesson(ctx, severity=1):
    """Recompute the four methods with fixed training settings for the short lesson."""
    candidate = select_candidate(ctx, severity=severity)
    return compare(ctx, candidate, split="test")


def show_adapter_example(ctx, result, query_id="1303"):
    """Apply the lesson's selected adapter to one unchanged input string."""
    from IPython.display import display, Markdown
    candidate = result["candidate"]
    severity = candidate["config"]["severity"]
    text = query_text(ctx, query_id, severity)
    positives = {d for d, grade in ctx.qrels[query_id].items() if grade > 0}
    rankings, rows = {}, []
    for method, label in [("none", "Without adapter"), ("repair_adapter", "With adapter")]:
        ranking = retrieve(ctx, queries=[query_id], severity=severity, intervention=method,
                           adapter=candidate["selection"]["adapter"])[query_id]
        rankings[method] = ranking
        found = len(positives & {d for d, _ in ranking})
        rows.append({"Search": label, "First paper": ctx.corpus[ranking[0][0]].split("\n")[0],
                     "Known relevant papers in top 20": f"{found} / {len(positives)}"})
    frame = pd.DataFrame(rows)
    paragraphs = [f"**Selected test query, maximum edits: {severity}**",
                  f"Input to both searches: {html.escape(text)}",
                  "The text stays unchanged. The adapter adjusts the 384-number query vector."]
    for row in rows:
        paragraphs.append(f"**{row['Search']}**\n\n"
                          f"First paper: {html.escape(row['First paper'])}\n\n"
                          f"Known relevant papers in top 20: {row['Known relevant papers in top 20']}.")
    if positives and all(not positives & {d for d, _ in ranking} for ranking in rankings.values()):
        paragraphs.append("Both searches miss the known evidence in this example. "
                          "The next table compares all 300 queries.")
    display(Markdown("\n\n".join(paragraphs)))
    if positives:
        ranks = {method: [rank for rank, (d, _) in enumerate(ranking, 1) if d in positives]
                 for method, ranking in rankings.items()}
        without, adapted = ranks["none"], ranks["repair_adapter"]
        first = ("changes the first paper" if rankings["none"][0][0] != rankings["repair_adapter"][0][0]
                 else "keeps the same first paper")
        if not without and not adapted:
            reading = (f"The adapter {first} but does not bring the known evidence back into the top 20, "
                       "so it does not repair this query.")
        elif len(adapted) > len(without):
            reading = "The adapter brings known evidence into the top 20 that the search without it misses."
        elif len(adapted) < len(without):
            reading = "The adapter loses known evidence that the search without it finds, so it harms this query."
        elif adapted == without:
            where = f"rank {adapted[0]}" if len(adapted) == 1 else "ranks " + ", ".join(map(str, adapted))
            reading = f"Both searches find the known evidence at {where}, so the adapter does not change the outcome here."
        else:
            reading = f"Both searches find the known evidence, but the adapter {_rank_move(without, adapted)}."
        show_interpretation(reading)
    return {"query_id": query_id, "severity": severity, "input_text": text,
            "rankings": rankings, "table": frame}


def show_lesson_comparison(result):
    """Display original/typo quality and a short reading of the current results."""
    frame = result["table"][["Method", "Clean nDCG@10", "Shifted nDCG@10"]].copy()
    frame["Method"] = frame["Method"].replace({NAMES[name]: label for name, label in LESSON_NAMES.items()})
    frame = frame.rename(columns={"Clean nDCG@10": "Original input",
                                  "Shifted nDCG@10": "After typos"})
    severity = result["candidate"]["config"]["severity"]
    best = frame["After typos"].max()
    leaders = ", ".join(frame.loc[np.isclose(frame["After typos"], best, rtol=0, atol=TIE), "Method"])
    baseline = result["evaluations"]["none"]["summary"]["shifted_ndcg"]
    table(frame, f"nDCG@10 (higher is better), {len(result['query_ids'])} {result['split']} queries, "
          f"maximum edits: {severity}",
          interpretation=f"{leaders} scores highest on the after-typo inputs ({best:.4f}), "
                         f"compared with {baseline:.4f} for original search.")
    show_interpretation(comparison_interpretation(result))
    return frame


def comparison_interpretation(result):
    """Which repair recovers the typo loss, at what cost on correctly written queries."""
    summary = {name: ev["summary"] for name, ev in result["evaluations"].items()}
    clean = {name: s["clean_ndcg"] for name, s in summary.items()}
    typo = {name: s["shifted_ndcg"] for name, s in summary.items()}
    repairs = [name for name in NAMES if name != "none"]
    if not result["candidate"]["config"]["severity"]:
        top, bottom = max(repairs, key=clean.get), min(repairs, key=clean.get)
        return ("Without typos the two columns match, so this shows only how each repair treats correctly "
                f"written queries: {IN_SENTENCE[top]} scores highest ({clean[top]:.4f}) and "
                f"{IN_SENTENCE[bottom]} lowest ({clean[bottom]:.4f}), against {clean['none']:.4f} for original search.")
    base_clean, base_typo = clean["none"], typo["none"]
    best = max(repairs, key=typo.get)
    if typo[best] <= base_typo+TIE:
        return "No repair beats original search after typos, so none of them recovers the typo loss on average."
    loss = base_clean-base_typo
    share = (typo[best]-base_typo)/loss if loss > TIE else 1.
    amount = "almost all" if share >= .9 else "most" if share >= .5 else "part"
    reached = (f"{typo[best]:.4f} after typos, close to original search's {base_clean:.4f} before typos"
               if share >= .9 else
               f"{base_typo:.4f} → {typo[best]:.4f} after typos, against original search's {base_clean:.4f} before typos")
    cost = clean[best]-base_clean
    cost = (f"at a {'small ' if abs(cost) < .01 else ''}cost on correctly written queries "
            f"({base_clean:.4f} → {clean[best]:.4f})" if cost < -TIE else
            "without lowering scores on correctly written queries")
    others = []
    for name in repairs:
        if name == best:
            continue
        typo_gain, clean_gain = typo[name]-base_typo, clean[name]-base_clean
        others.append(f"{IN_SENTENCE[name]} changes no score" if abs(typo_gain) <= TIE and abs(clean_gain) <= TIE else
                      f"{IN_SENTENCE[name]} mainly raises original-query scores" if clean_gain > max(typo_gain, TIE) else
                      f"{IN_SENTENCE[name]} recovers less" if typo_gain > TIE else
                      f"{IN_SENTENCE[name]} does not help after typos")
    return (f"{IN_SENTENCE[best].capitalize()} recovers {amount} of the typo loss ({reached}) {cost}, "
            f"while {_join(others)}.")


def effect_intervals(result, seed=42, n_boot=500):
    """Between-method paired differences; not within-method typo degradation."""
    base = result["evaluations"]["none"]["per_query"]
    rows = []
    for name, ev in result["evaluations"].items():
        f = ev["per_query"]
        if not f.index.equals(base.index):
            raise ValidationError("Align method query IDs before computing an intervention effect.")
        for condition in ("clean", "shifted"):
            ci = paired_bootstrap(base[f"{condition}_ndcg"], f[f"{condition}_ndcg"], seed=seed, n_boot=n_boot)
            rows.append({"method": name, "condition": condition, **ci})
    return pd.DataFrame(rows)


def show_effects(result):
    effects = effect_intervals(result)
    rows = []
    for name in NAMES:
        f = effects[effects.method == name].set_index("condition")
        rows.append({"Method": NAMES[name], **{
            f"{condition.title()} gain [95% CI]": f"{f.loc[condition, 'estimate']:+.4f} [{f.loc[condition, 'low']:+.4f}, {f.loc[condition, 'high']:+.4f}]"
            for condition in ("clean", "shifted")}})
    shifted = effects[effects.condition == "shifted"]
    best = shifted.loc[shifted.estimate.idxmax()]
    table(pd.DataFrame(rows), f"{result['split'].title()} n={len(result['query_ids'])}; " +
          configuration_label(result["candidate"]) + "; method − dense (higher gain is better); 500 paired resamples, seed 42",
          interpretation=f"{NAMES[best['method']]} has the largest measured after-typo gain "
                         f"({best['estimate']:+.4f}); the original-input intervals show its clean-query cost.")
    return effects


def search_summary(ctx, query_id, ranking):
    """First paper, ranks of known relevant papers among the 20 candidates, and nDCG@10."""
    positives = {d for d, grade in ctx.qrels[query_id].items() if grade > 0}
    return {"first_paper": ctx.corpus[ranking[0][0]].split("\n")[0],
            "relevant_ranks": [rank for rank, (d, _) in enumerate(ranking, 1) if d in positives],
            "relevant_count": len(positives), "ndcg": ndcg(ranking, ctx.qrels[query_id])}


def count_queries(n):
    return f"{n} {'query' if n == 1 else 'queries'}"


def failure_case(ctx, result, method):
    """Select one method's largest failure from the lesson run; never refit or reselect.

    The case is the largest nDCG@10 drop against the original search on the same
    input; ties prefer the original input, then test order. A method that lowers
    no score fails by leaving typo damage in place, so its case is instead the
    largest drop from the original query to the method's search after typos.
    """
    if method not in SUBJECTS:
        raise ValidationError("Choose input_normalization, vocab_spell_correction, or repair_adapter.")
    severity = result["candidate"]["config"]["severity"]
    ids = result["query_ids"]
    base = result["evaluations"]["none"]["per_query"]
    runs = result["evaluations"][method]["per_query"]
    damaged = base["clean_ndcg"] - base["shifted_ndcg"] > TIE
    still_lower = base["clean_ndcg"] - runs["shifted_ndcg"] > TIE
    case = {"method": method, "severity": severity, "query_count": len(ids),
            "counts": {c: {"lower": int((base[f"{c}_ndcg"] - runs[f"{c}_ndcg"] > TIE).sum()),
                           "higher": int((runs[f"{c}_ndcg"] - base[f"{c}_ndcg"] > TIE).sum())}
                       for c in ("clean", "shifted")},
            "typo_losses": int(damaged.sum()), "typo_losses_repaired": int((damaged & ~still_lower).sum())}
    vocabulary = build_vocabulary(ctx.corpus) if method == "vocab_spell_correction" else None
    if vocabulary is not None:
        # A score can fall even when correction restores the original wording exactly.
        lowered = base["shifted_ndcg"] - runs["shifted_ndcg"] > TIE
        case["lowered_after_typos_restored"] = sum(
            vocab_spell_correction(query_text(ctx, q, severity), vocabulary) == ctx.queries[q]
            for q in lowered[lowered].index)
        outcomes = {"restored": [], "unchanged": [], "different": []}
        for q in ids:
            for edit in (ctx.variants[str(severity)][q]["edits"] if severity else []):
                corrected = vocab_spell_correction(edit["after"], vocabulary)
                key = ("restored" if corrected == edit["before"] else
                       "unchanged" if corrected == edit["after"] else "different")
                outcomes[key].append({"query_id": q, "original": edit["before"],
                                      "typo": edit["after"], "corrected": corrected})
        case["typo_word_outcomes"] = outcomes
    drops = pd.concat({c: base[f"{c}_ndcg"] - runs[f"{c}_ndcg"] for c in ("clean", "shifted")})
    kind = "worse"
    if not drops.max() > TIE:
        kind = "unrepaired"
        drops = pd.concat({"shifted": base["clean_ndcg"] - runs["shifted_ndcg"]})
    if not drops.max() > TIE:
        return {**case, "kind": None, "query_id": None}
    condition, query_id = drops.idxmax()  # First maximum: original input, then test order.
    level = severity if condition == "shifted" else 0
    text = query_text(ctx, query_id, level)
    output = (input_normalization(text) if method == "input_normalization" else
              vocab_spell_correction(text, vocabulary) if vocabulary is not None else text)
    changed = ([(a, b) for a, b in zip(TOKEN.findall(text), TOKEN.findall(output)) if a != b]
               if vocabulary is not None else [])
    rankings = {"none": retrieve(ctx, queries=[query_id], severity=0 if kind == "unrepaired" else level)[query_id],
                method: retrieve(ctx, queries=[query_id], severity=level, intervention=method,
                                 adapter=result["candidate"]["selection"]["adapter"])[query_id]}
    return {**case, "kind": kind, "query_id": query_id, "input": condition, "drop": float(drops.max()),
            "original_text": ctx.queries[query_id], "input_text": text, "output_text": output,
            "typo_words": [(e["before"], e["after"]) for e in sorted(
                ctx.variants[str(level)][query_id]["edits"] if level else [], key=lambda e: e["start"])],
            "changed_words": changed, "word_counts": {w: vocabulary[w] for pair in changed for w in pair},
            "rankings": rankings,
            "searches": {name: search_summary(ctx, query_id, ranking) for name, ranking in rankings.items()}}


def show_failure_case(ctx, result, method):
    """Display one method's largest failure from the run behind the comparison table."""
    from IPython.display import display, Markdown
    case = failure_case(ctx, result, method)
    name, subject, severity, counts = LESSON_NAMES[method], SUBJECTS[method], case["severity"], case["counts"]
    unchanged = not any(value for c in counts.values() for value in c.values())
    if case["query_id"] is None:
        display(Markdown(f"**{name}, maximum edits: {severity}**\n\n" +
            (f"{subject.capitalize()} gave the same nDCG@10 as the original search for all "
             f"{case['query_count']} queries" if unchanged else f"{subject.capitalize()} lowered no nDCG@10 score") +
            (", and severity 0 adds no typos. Set severity to 1 or more and use Run All Below to see "
             "a typo pass through." if severity == 0 else ", and no typo lowered nDCG@10.")))
        show_interpretation(failure_interpretation(case))
        return case
    if case["kind"] == "unrepaired":
        header = f"maximum edits: {severity}"
        texts = [("Original query", case["original_text"]), ("After typos", case["input_text"]),
                 (f"After {subject}", case["output_text"])]
        titles = {"none": "Original search, original query", method: f"{name}, after typos"}
    else:
        header = "original input" if case["input"] == "clean" else f"after typos, maximum edits: {severity}"
        texts = [] if case["input"] == "clean" else [("Original query", case["original_text"])]
        texts += ([("Input to both searches", case["input_text"])] if method == "repair_adapter" else
                  [("Input", case["input_text"]), (f"After {subject}", case["output_text"])])
        titles = {"none": "Original search", method: name}
    paragraphs = [f"**{name}, {header}**"]
    paragraphs += [f"{label}: {html.escape(text)}" for label, text in texts]
    if case["typo_words"]:
        typos, plural = ", ".join(f"{a} → {b}" for a, b in case["typo_words"]), len(case["typo_words"]) > 1
        kept = all(b.casefold() in TOKEN.findall(case["output_text"].casefold()) for _, b in case["typo_words"])
        paragraphs.append(f"Typo{'s' if plural else ''}: {typos}." +
                          (f" {subject.capitalize()} keeps the misspelling{'s' if plural else ''}."
                           if case["kind"] == "unrepaired" and kept else ""))
    if case["changed_words"]:
        n = case["word_counts"]
        paragraphs.append(f"Replaced word{'s' if len(case['changed_words']) > 1 else ''}: " +
                          ", ".join(f"{a} → {b}" for a, b in case["changed_words"]) + ". In the documents, " +
                          "; ".join(f"{a} appears {n[a]} time{'' if n[a] == 1 else 's'} and "
                                    f"{b} {n[b]} time{'' if n[b] == 1 else 's'}" for a, b in case["changed_words"]) + ".")
    if method == "repair_adapter":
        paragraphs.append("The text stays unchanged. The adapter adjusts the 384-number query vector.")
    for key in ("none", method):
        s = case["searches"][key]; ranks = s["relevant_ranks"]
        where = (f", at rank {ranks[0]}" if len(ranks) == 1 else
                 f", at ranks {', '.join(map(str, ranks))}" if ranks else "")
        paragraphs.append(f"**{titles[key]}**\n\nFirst paper: {html.escape(s['first_paper'])}\n\n"
                          f"Known relevant papers in top 20: {len(ranks)} / {s['relevant_count']}{where}. "
                          f"nDCG@10: {s['ndcg']:.4f}.")
    if unchanged:
        run = (f"{subject.capitalize()} gave the same nDCG@10 as the original search for all "
               f"{case['query_count']} queries.")
    elif severity == 0:
        run = (f"Compared with the original search, {subject} lowered nDCG@10 for "
               f"{count_queries(counts['clean']['lower'])} and raised it for {counts['clean']['higher']}.")
    else:
        run = (f"On original input, {subject} lowered nDCG@10 for "
               f"{count_queries(counts['clean']['lower'])} and raised it for {counts['clean']['higher']} "
               f"compared with original search. After typos, it lowered nDCG@10 for "
               f"{count_queries(counts['shifted']['lower'])} and raised it for {counts['shifted']['higher']}.")
    if case["kind"] == "unrepaired":
        run += (f" Typos lowered nDCG@10 for {count_queries(case['typo_losses'])}; {subject} returned "
                f"{case['typo_losses_repaired'] or 'none'} of them to the original-query score.")
    if case.get("lowered_after_typos_restored") and severity:
        run += (f" In {case['lowered_after_typos_restored']} of those {counts['shifted']['lower']} queries, "
                "correction restored the original wording exactly. The typo versions had scored higher than the originals.")
    paragraphs.append(f"**In this run:** {run}")
    if case.get("typo_word_outcomes") and severity:
        words = case["typo_word_outcomes"]
        sentence = (f"Of the {sum(map(len, words.values()))} typo words, correction restored "
                    f"{len(words['restored'])}, left {len(words['unchanged'])} unchanged, and replaced "
                    f"{len(words['different'])} with a different word")
        if words["different"]:
            e = words["different"][0]
            sentence += (f". For example, {e['typo']} → {e['corrected']}, "
                         f"but the original word was {e['original']}")
        paragraphs.append(sentence + ".")
    display(Markdown("\n\n".join(paragraphs)))
    show_interpretation(failure_interpretation(case))
    return case


def failure_interpretation(case):
    """One sentence on why this repair failed on the displayed query."""
    method, subject = case["method"], SUBJECTS[case["method"]]
    if case["query_id"] is None:
        return (f"Without typos there is nothing to repair, and {subject} lowers no score." if case["severity"] == 0 else
                f"{subject.capitalize()} lowers no score at this setting.")
    base, repaired = case["searches"]["none"], case["searches"][method]
    move = _rank_move(base["relevant_ranks"], repaired["relevant_ranks"])
    if case["kind"] == "unrepaired":
        typos = [typo for _, typo in case["typo_words"]]
        reach = (f"{_join(typos)} still reach{'es' if len(typos) == 1 else ''} the encoder" if typos else
                 "the typo damage remains")
        harmless = (f"{subject.capitalize()} is harmless but does not help against these typos"
                    if not any(value for c in case["counts"].values() for value in c.values()) else
                    f"{subject.capitalize()} does not repair this typo")
        return (f"{harmless}: {reach}, so nDCG@10 stays at {repaired['ndcg']:.4f} "
                f"instead of {base['ndcg']:.4f}.")
    which = "correctly written" if case["input"] == "clean" else "misspelled"
    if method == "vocab_spell_correction" and case["changed_words"]:
        before, after = case["changed_words"][0]
        count = case["word_counts"][before]
        absent = "never appears" if count == 0 else f"appears only {count} time{'s' if count != 1 else ''}"
        return (f"Because {before} {absent} in the papers, correction treats it as a typo and changes it to "
                f"{after}, which {move} on this {which} claim.")
    if method == "repair_adapter":
        return (f"The adapter changes only the vector of this {which} query, not its text, "
                f"and that shift {move}.")
    return f"{subject.capitalize()} changes this {which} query, which {move}."
