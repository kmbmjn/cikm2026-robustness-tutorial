"""M3 measurement and presentation over verified, immutable full-test runs.

Every lab call takes all controls explicitly, including the estimand: relevance
deltas or finite RBO minus identity stability.
"""
import math

import numpy as np
import pandas as pd

from .config import ValidationError
from .metrics import checked_ranking, evaluate_suite, paired_bootstrap, recall
from .perturbations import make_query_typos, perturbation_accounting
from .plotting import COLORS, legend_below, show, square_axes, titles
from .presentation import interpretation_html, show_interpretation

TIE = 1e-12  # Score differences this small count as equal.


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else \
        ", ".join(items[:-1]) + ", and " + items[-1]


def validate_controls(ctx, severity=1, k=10, metric="ndcg", p=.9, tail_fraction=.1, n_boot=500):
    if ctx.mode != "cached":
        raise ValidationError("M3 measures verified cached runs. Keep MODE='cached'; see README for live APIs.")
    if type(severity) is not int or severity not in range(4):
        raise ValidationError("M3 severity must be an integer: 0, 1, 2, or 3.")
    if type(k) is not int or k not in (5, 10, 20):
        raise ValidationError("M3 K must be 5, 10, or 20; the stored runs contain 20 results.")
    if metric not in {"ndcg", "recall", "rbo"}:
        raise ValidationError("M3 metric must be ndcg, recall, or rbo.")
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValidationError("rbo_p must be strictly between 0 and 1.")
    if not np.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValidationError("tail_fraction must be in (0, 1].")
    if type(n_boot) is not int or not 1 <= n_boot <= 10000:
        raise ValidationError("bootstrap_repetitions must be an integer from 1 to 10000.")


def metric_label(metric, k):
    return f"{dict(ndcg='nDCG', recall='Recall', rbo='RBO')[metric]}@{k}"


def select_measure(ev, metric, seed=42, n_boot=500):
    """Select the estimand without changing the shared evaluation's nDCG fields."""
    frame = ev["per_query"]
    if metric == "rbo":
        # RBO is a property of a ranking pair, not a relevance score of one list.
        scores = pd.DataFrame({"clean": np.ones(len(frame)), "shifted": frame.rbo}, index=frame.index)
    elif metric in {"ndcg", "recall"}:
        scores = frame[[f"clean_{metric}", f"shifted_{metric}"]].copy()
        scores.columns = ["clean", "shifted"]
    else:
        raise ValidationError("metric must be ndcg, recall, or rbo.")
    scores = scores.loc[np.isfinite(scores).all(axis=1)].copy()
    scores["delta"] = scores.shifted - scores.clean
    count = math.ceil(len(scores) * ev["summary"]["tail_fraction"])
    # Stable tie breaking gives reproducible examples; tied tail members are arbitrary.
    bottom = scores.sort_values("shifted", kind="stable").iloc[:count]
    drops = scores.sort_values("delta", kind="stable").iloc[:count]
    ci = paired_bootstrap(scores.clean, scores.shifted, seed=seed, n_boot=n_boot)
    return {"scores": scores, "metric": metric, "n": len(scores), "tail_count": count,
            "clean": float(scores.clean.mean()), "shifted": float(scores.shifted.mean()),
            "delta": float(scores.delta.mean()), "bottom": float(bottom.shifted.mean()),
            "worst_delta": float(drops.delta.mean()), "bottom_ids": list(bottom.index),
            "worst_ids": list(drops.index), "ci": {**ci, "seed": seed, "n_boot": n_boot}}


def evaluate_cached(ctx, system, severity, k=10, p=.9, tail_fraction=.1, n_boot=500):
    runs = []
    for level in (0, severity):
        key = f"{system}_s{level}"
        if key not in ctx.runs or set(ctx.runs[key]) != set(ctx.test_ids):
            raise ValidationError(f"Missing full-test cache {key}; restore the supplied assets.")
        run = ctx.runs[key]
        if any(len(checked_ranking(run[q])) != 20 for q in ctx.test_ids):
            raise ValidationError("M3 requires exactly 20 distinct results per query.")
        if system == "rerank" and any(
            set(checked_ranking(run[q])) != set(checked_ranking(ctx.runs[f"dense_s{level}"][q]))
            for q in ctx.test_ids
        ):
            raise ValidationError("Reranker cache must use that condition's dense top-20 candidates.")
        runs.append(run)
    return evaluate_suite(ctx, *runs, k=k, p=p, tail_fraction=tail_fraction, n_boot=n_boot)


def generation_audit(ctx):
    rows = []
    for severity in range(4):
        records = (make_query_typos({q: ctx.queries[q] for q in ctx.test_ids}, 0, ctx.seed)
                   if severity == 0 else {q: ctx.variants[str(severity)][q] for q in ctx.test_ids})
        a = perturbation_accounting(records)
        rows.append({"Maximum edits": severity, "Queries": a["queries"], "Edited queries": a["modified"],
                     "Unchanged": a["unchanged"], "Queries at maximum edits": a["requested_edits_achieved"],
                     "Reviewed flags": a["human_reviewed"]})
    return pd.DataFrame(rows)


def table_style(frame, caption):
    return (frame.style.hide(axis="index").format(precision=4, na_rep="undefined", escape="html")
            .set_caption(caption).set_table_styles([
                {"selector": "caption", "props": [("text-align", "left"), ("font-size", "17px"), ("font-weight", "600")]},
                {"selector": "th, td", "props": [("font-size", "17px"), ("text-align", "left"),
                                                   ("white-space", "normal"), ("max-width", "230px")]},
            ]))


def show_table(frame, caption, interpretation):
    from IPython.display import HTML, display
    display(table_style(frame, caption))
    display(HTML(interpretation_html(interpretation)))


SYSTEMS = {"BM25": "bm25", "Dense": "dense", "Dense → reranker": "rerank"}


def compare_changes(first, second, seed=42, n_boot=500):
    """Bootstrap first-minus-second changes, pairing systems by query ID."""
    pairs = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    return paired_bootstrap(pairs["second"], pairs["first"], seed, n_boot)


def prepare_questions(ctx, severity=1, k=10, metric="ndcg", p=.9, tail_fraction=.1, n_boot=500):
    validate_controls(ctx, severity, k, metric, p, tail_fraction, n_boot)
    if metric not in {"ndcg", "recall"}:
        raise ValidationError("Choose ndcg or recall for quality comparisons; RBO is reported separately.")
    systems, stability, coverage = {}, {}, []
    for name, key in SYSTEMS.items():
        systems[name] = {}
        for level in range(4):
            ev = evaluate_cached(ctx, key, level, k, p, tail_fraction, n_boot)
            systems[name][level] = select_measure(ev, metric, ctx.seed, n_boot)
            if level == severity:
                stability[name] = float(ev["per_query"].rbo.mean())
                before_coverage = np.mean([
                    sum(d in ctx.qrels[q] for d in checked_ranking(ctx.runs[f"{key}_s0"][q])[:k])/k
                    for q in ctx.test_ids])
                coverage.append({"System": name, "Before judged@K": before_coverage,
                                 "After judged@K": ev["summary"]["judged_coverage"]})
    comparisons = {}
    # Prespecified contrasts, independent of whichever system leads the sample.
    for first, second in [("BM25", "Dense"), ("BM25", "Dense → reranker"),
                          ("Dense → reranker", "Dense")]:
        comparisons[f"{first} − {second}"] = compare_changes(
            systems[first][severity]["scores"].delta,
            systems[second][severity]["scores"].delta, ctx.seed, n_boot)
    cutoffs = []
    for name, key in SYSTEMS.items():
        for cutoff in (5, 20):
            before, after = [np.array([recall(ctx.runs[f"{key}_s{level}"][q], ctx.qrels[q], cutoff)
                                      for q in ctx.test_ids]) for level in (0, severity)]
            valid = np.isfinite(before) & np.isfinite(after)
            cutoffs.append({"System": name, "K": cutoff, "Before": float(before[valid].mean()),
                            "After": float(after[valid].mean()),
                            "Recall difference (after − before)": float((after[valid]-before[valid]).mean())})
    return {"config": {"severity": severity, "k": k, "metric": metric, "p": p,
                       "tail_fraction": tail_fraction, "n_boot": n_boot, "seed": ctx.seed},
            "systems": systems, "stability": stability, "comparisons": comparisons,
            "cutoffs": pd.DataFrame(cutoffs), "audit": generation_audit(ctx),
            "coverage": pd.DataFrame(coverage),
            "n_requested": len(ctx.test_ids), "corpus_count": len(ctx.corpus)}


def observed_leaders(result, field="delta", severity=None):
    level = result["config"]["severity"] if severity is None else severity
    values = {name: levels[level][field] for name, levels in result["systems"].items()}
    best = max(values.values())
    return [name for name, value in values.items() if np.isclose(value, best, atol=1e-12, rtol=0)]


def question_table(result, question):
    c = result["config"]
    label = metric_label(c["metric"], c["k"])
    current = {name: levels[c["severity"]] for name, levels in result["systems"].items()}
    if question == 1:
        return pd.DataFrame([{"System": name, "Before typos": v["clean"], "After typos": v["shifted"],
                              f"Mean Δ{label}": v["delta"]} for name, v in current.items()])
    if question == 3:
        return pd.DataFrame([{"System": name, "Lowest-group score": v["bottom"],
                              f"Largest-drop group Δ{label}": v["worst_delta"]} for name, v in current.items()])
    if question == 4:
        return pd.DataFrame([{"System": name, f"Mean RBO@{c['k']}": value}
                             for name, value in result["stability"].items()])
    if question == 5:
        intervals = {"Dense: after − before": current["Dense"]["ci"],
                     f"BM25 Δ{label} − Dense Δ{label}": result["comparisons"]["BM25 − Dense"]}
        return pd.DataFrame([{"Comparison": name, "Estimate": ci["estimate"],
                              "95% interval": f"[{ci['low']:+.4f}, {ci['high']:+.4f}]"}
                             for name, ci in intervals.items()])
    if question == 6:
        return pd.DataFrame([{"Maximum edits": s,
                              **{name: levels[s]["delta"] for name, levels in result["systems"].items()},
                              "Observed leader(s)": ", ".join(observed_leaders(result, severity=s))}
                             for s in range(4)])
    if question == 7:
        return result["cutoffs"].copy()
    raise ValidationError("This question has no table.")


def question_figure(result, question):
    c = result["config"]
    label = metric_label(c["metric"], c["k"])
    fig, ax = square_axes()
    if question == 2:
        dense = result["systems"]["Dense"][c["severity"]]
        ax.hist(dense["scores"].delta, bins=np.linspace(-1, 1, 26), color=COLORS["shifted"],
                edgecolor="white", linewidth=.6)
        ax.axvline(0, color=COLORS["ink"], lw=1)
        ax.axvline(dense["delta"], color=COLORS["clean"], label=f"Mean {dense['delta']:+.4f}")
        ax.set(xlabel=f"Δ{label} (after − before)", ylabel="Queries")
        titles(ax, "Dense: score gains and losses per query")
        legend_below(fig, ax)
    elif question == 6:
        # The shaded column marks the current setting; the curves always show all four edit settings.
        ax.axvspan(c["severity"]-.22, c["severity"]+.22, color=COLORS["grid"], zorder=0, lw=0)
        ax.axhline(0, color=COLORS["ink"], lw=1)
        for (name, levels), color, marker in zip(result["systems"].items(),
                                                [COLORS["clean"], COLORS["shifted"], COLORS["repaired"]], ["o", "s", "^"]):
            ax.plot(range(4), [levels[s]["delta"] for s in range(4)], marker=marker, color=color, label=name,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set(xlabel="Maximum character edits per query",
               ylabel=f"Mean Δ{label}\n(after − before; lower = larger loss)",
               xticks=range(4), xlim=(-.4, 3.4))
        titles(ax, f"{label} loss as typos increase")
        legend_below(fig, ax)
    else:
        raise ValidationError("This question has no figure.")
    return fig


# The main lesson has three questions and one editable control: severity.
EXAMPLE_QUERY = "1303"  # M1's selected failure, revisited in Individual Query Outcomes.


def comparison_reading(result):
    """Answer the score-after-typos and smallest-drop questions separately."""
    c = result["config"]
    severity = c["severity"]
    label = metric_label(c["metric"], c["k"])
    leaders = observed_leaders(result)
    quality_leaders = observed_leaders(result, "shifted")
    best = result["systems"][leaders[0]][severity]
    quality = result["systems"][quality_leaders[0]][severity]["shifted"]
    if severity == 0:
        return (f"Without typos, every query keeps its original {label}. "
                f"{', '.join(quality_leaders)} has the highest score ({label}: {quality:.4f}).")
    reading = (f"Highest score after typos: {', '.join(quality_leaders)} ({label}: {quality:.4f}).\n\n"
               f"Smallest drop from the original score: {', '.join(leaders)} "
               f"(mean Δ{label}: {best['delta']:+.4f}, after − before).\n\n")
    if set(leaders) == set(quality_leaders):
        return reading + "The system with the highest score after typos also loses the fewest points here."
    return reading + "The system that loses the fewest points does not have the highest score after typos."


def show_comparison(result):
    c = result["config"]
    label = metric_label(c["metric"], c["k"])
    quality_leaders = observed_leaders(result, "shifted")
    loss_leaders = observed_leaders(result)
    quality = result["systems"][quality_leaders[0]][c["severity"]]["shifted"]
    change = result["systems"][loss_leaders[0]][c["severity"]]["delta"]
    effect = (f"loses the fewest points ({-change:.4f})" if change < -1e-12 else
              f"gains the most points ({change:.4f})" if change > 1e-12 else "has zero average loss")
    interpretation = (
        f"{', '.join(quality_leaders)} scores highest after typos ({quality:.4f}), while "
        f"{', '.join(loss_leaders)} {effect}."
        if c["severity"] else
        f"No edits change any score; {', '.join(quality_leaders)} has the highest {label} ({quality:.4f}).")
    show_table(question_table(result, 1),
               f"{label}, {result['n_requested']} queries, maximum edits: {c['severity']}; "
               "higher scores and higher mean changes are better", interpretation)
    if not c["severity"]:
        show_interpretation(f"With no edits, every score stays the same, so this table shows only original "
                            f"quality, where {_join(quality_leaders)} lead{'s' if len(quality_leaders) == 1 else ''}.")
        return
    losers = [name for name, levels in result["systems"].items() if levels[c["severity"]]["delta"] < -TIE]
    first = ("Every system loses quality after typos" if len(losers) == len(result["systems"]) else
             f"{_join(losers)} lose{'s' if len(losers) == 1 else ''} quality after typos" if losers else
             "No system loses average quality after typos")
    second = (f"the best final score ({_join(quality_leaders)}) and the smallest loss ({_join(loss_leaders)}) "
              "come from different systems, so report both columns"
              if set(quality_leaders) != set(loss_leaders) else
              f"{_join(quality_leaders)} has both the best final score and the smallest loss")
    show_interpretation(f"{first}, and {second}.")


def outcome_counts(result):
    scores = result["systems"]["Dense"][result["config"]["severity"]]["scores"]
    return pd.Series({"Lower score": int((scores.delta < -1e-12).sum()),
                      "Same score": int((scores.delta.abs() <= 1e-12).sum()),
                      "Higher score": int((scores.delta > 1e-12).sum())}, name="Queries")


def outcome_figure(result):
    c = result["config"]
    counts = outcome_counts(result)
    fig, ax = square_axes()
    bars = ax.bar(counts.index, counts.values, width=.5,
                  color=[COLORS["shifted"], COLORS["muted"], COLORS["repaired"]])
    ax.bar_label(bars, padding=4)
    ax.set(ylim=(0, max(1, counts.sum())*1.1), xlim=(-.6, 2.6), ylabel="Number of queries")
    titles(ax, "Dense: scores lower, equal, or higher")
    return fig


def query_example(ctx, result, query_id=EXAMPLE_QUERY):
    """Reuse M1's selected case, with text and scores from the current condition."""
    severity = result["config"]["severity"]
    scores = result["systems"]["Dense"][severity]["scores"]
    if query_id not in scores.index:
        raise ValidationError("The example must be a relevance-evaluable query in the current result.")
    variant = ctx.variants[str(severity)][query_id] if severity else None
    row = scores.loc[query_id]
    return {"query_id": query_id, "severity": severity, "original": ctx.queries[query_id],
            "changed": variant["text"] if variant else ctx.queries[query_id],
            "actual_edits": len(variant["edits"]) if variant else 0,
            "before": float(row.clean), "after": float(row.shifted), "change": float(row.delta)}


def outcome_reading(result):
    """Explain how offsetting gains and unchanged queries shrink Dense's average."""
    dense = result["systems"]["Dense"][result["config"]["severity"]]
    label = metric_label(result["config"]["metric"], result["config"]["k"])
    counts = outcome_counts(result)
    if not counts["Lower score"]:
        rise = f" and {counts['Higher score']} rise" if counts["Higher score"] else ""
        return (f"No Dense score drops: {counts['Same score']} stay the same{rise}. "
                f"Mean Δ{label} is {dense['delta']:+.4f} (after − before).")
    delta = dense["scores"].delta
    parts = [f"Dense: {counts['Lower score']} lower, {counts['Same score']} same, and "
             f"{counts['Higher score']} higher scores. Queries with lower scores lose "
             f"{-delta[delta < -1e-12].mean():.4f} on average."]
    if counts["Higher score"]:
        parts.append("Gains on other queries cancel out part of this loss in the average.")
    if counts["Same score"]:
        parts.append("Queries with unchanged scores also count in the average.")
    parts.append(f"Across all {dense['n']} queries, mean {label} goes from "
                 f"{dense['clean']:.4f} to {dense['shifted']:.4f} (Δ{dense['delta']:+.4f}).")
    return " ".join(parts)


def example_reading(result, ex):
    """Set the average beside the revisited M1 query."""
    dense = result["systems"]["Dense"][result["config"]["severity"]]
    if ex["change"] < 0:
        return (f"Dense's mean score goes from {dense['clean']:.4f} to {dense['shifted']:.4f}, "
                f"but this example alone loses {abs(ex['change']):.4f} points: its score "
                f"falls from {ex['before']:.4f} to {ex['after']:.4f}.")
    return f"This example does not lose score at this setting ({ex['before']:.4f} → {ex['after']:.4f})."


def show_query_outcomes(ctx, result):
    from IPython.display import Markdown, display
    show(outcome_figure(result))
    c = result["config"]
    display(Markdown(outcome_reading(result)))
    show_interpretation(outcome_interpretation(result))
    label = metric_label(c["metric"], c["k"])
    scores = result["systems"]["Dense"][1]["scores"]
    names = {"1303": "Tirasemtiv", "1012": "Radioiodine", "628": "HTLV-1"}
    rows, changes = [], []
    for query_id in ctx.demo_ids:
        score = scores.loc[query_id]
        rows.append({"Query ID": query_id, "Example": names[query_id],
                     f"Original {label}": score.clean, f"One-edit {label}": score.shifted,
                     f"Δ{label}": score.delta})
        outcome = (f"loses {-score.delta:.4f} points" if score.delta < -1e-12 else
                   f"gains {score.delta:.4f} points" if score.delta > 1e-12 else "keeps the same score")
        changes.append(f"{names[query_id]} {outcome}")
    show_table(pd.DataFrame(rows),
               f"Selected severity-1 cases (Dense {label}; fixed, regardless of current severity)",
               "At one edit, " + "; ".join(changes) + ".")
    kinds = []
    for query_id in ctx.demo_ids:
        score = scores.loc[query_id]
        kind = (("a total loss" if score.shifted <= TIE else "a loss") if score.delta < -TIE else
                "a gain" if score.delta > TIE else "no change")
        kinds.append(f"{names[query_id]} {kind}")
    show_interpretation(f"One character edit gives {_join(kinds)}, so a single example cannot stand for "
                        f"all {result['n_requested']} queries; the bars above show how often each outcome occurs.")


def outcome_interpretation(result):
    """Why a small mean change can hide large individual failures."""
    c = result["config"]
    dense = result["systems"]["Dense"][c["severity"]]
    scores = dense["scores"]
    counts = outcome_counts(result)
    label = metric_label(c["metric"], c["k"])
    if not c["severity"]:
        return "With no edits, every query keeps its score, so the average hides nothing here."
    if not counts["Lower score"]:
        return f"No Dense query loses {label} at this setting, so the average hides no failures."
    lower = scores.delta[scores.delta < -TIE]
    zeroed = int(((scores.clean > TIE) & (scores.shifted <= TIE)).sum())
    dilute = [f"{counts['Same score']} unchanged queries" if counts["Same score"] else "",
              f"{counts['Higher score']} gains" if counts["Higher score"] else ""]
    dilute = _join([part for part in dilute if part])
    losers = ("the only query that loses drops" if counts["Lower score"] == 1 else
              f"the {counts['Lower score']} queries that do lose drop")
    return (f"The mean loss of {abs(dense['delta']):.4f} is diluted by {dilute}, while {losers} "
            f"{-lower.mean():.4f} on average"
            + (f", and {zeroed} of them fall to 0 with no known relevant paper left in the top ten." if zeroed else "."))


def show_severity_experiment(result):
    show(question_figure(result, 6))
    label = metric_label(result["config"]["metric"], result["config"]["k"])
    leaders = {level: ", ".join(observed_leaders(result, severity=level)) for level in (1, 2, 3)}
    show_table(question_table(result, 6),
               f"Mean Δ{label} at all edit settings (after − before; higher is better)",
               f"{leaders[1]} loses the fewest points at one edit, {leaders[2]} at up to two edits, "
               f"and {leaders[3]} at up to three edits.")
    deltas = {name: [levels[s]["delta"] for s in range(4)] for name, levels in result["systems"].items()}
    deeper = all(values[s+1] <= values[s]+TIE for values in deltas.values() for s in range(3))
    worst = min(deltas, key=lambda name: deltas[name][3])
    one, three = observed_leaders(result, severity=1), observed_leaders(result, severity=3)
    trend = ("Each extra edit deepens the average loss for every system" if deeper else
             "More edits generally deepen the average loss")
    ending = (f"and the smallest loss shifts from {_join(one)} at one edit to {_join(three)} at up to "
              "three edits, so test more than one typo level" if set(one) != set(three) else
              f"and {_join(three)} has the smallest loss at both one and three edits")
    show_interpretation(f"{trend}, {worst} falls furthest ({deltas[worst][3]:+.4f} at up to three edits), {ending}.")


def severity_reading(result):
    """Compare score losses and name the observed leaders at one and up to three edits."""
    c = result["config"]
    label = metric_label(c["metric"], c["k"])
    leaders = {level: observed_leaders(result, severity=level) for level in (1, 3)}
    drops = {level: result["systems"][names[0]][level]["delta"] for level, names in leaders.items()}
    reading = (f"Negative values mean lost {label} points; lower points mean larger losses. "
               f"With one edit, {', '.join(leaders[1])} has the smallest average loss "
               f"(Δ{label}: {drops[1]:+.4f}). With up to three edits, "
               f"{', '.join(leaders[3])} has the smallest average loss (Δ{label}: {drops[3]:+.4f}).")
    if leaders[1] != leaders[3]:
        reading += (f" {', '.join(leaders[3])} loses fewer points than {', '.join(leaders[1])} with up to "
                    "three edits, reversing their order by average loss.")
    else:
        reading += " The same system loses the fewest points at both edit settings."
    return reading + (f" A robustness comparison combines original and typo {label} scores and their "
                      "difference with the edit operations (character insertion, deletion, or replacement), "
                      "the maximum edits per query, and the actual edit counts.")


def lesson_summary(result):
    c = result["config"]
    counts = outcome_counts(result)
    dense = result["systems"]["Dense"][c["severity"]]
    leader = ", ".join(observed_leaders(result))
    quality = ", ".join(observed_leaders(result, "shifted"))
    label = metric_label(c["metric"], c["k"])
    quality_score = result["systems"][observed_leaders(result, "shifted")[0]][c["severity"]]["shifted"]
    leader_delta = result["systems"][observed_leaders(result)[0]][c["severity"]]["delta"]
    if not c["severity"]:
        observation = (f"No edits: every query keeps its original score. {quality} has the highest "
                       f"mean {label} ({quality_score:.4f}).")
    else:
        observation = (f"Highest score after typos: {quality} ({quality_score:.4f}). Smallest average "
                       f"loss: {leader} (Δ{label}: {leader_delta:+.4f}, after − before).")
    example = dense["scores"].loc[EXAMPLE_QUERY]
    failure = (f" M1's default test query alone falls from {example.clean:.4f} to {example.shifted:.4f}."
               if example.delta < 0 else "")
    one, three = (", ".join(observed_leaders(result, severity=level)) for level in (1, 3))
    one_delta = result["systems"][observed_leaders(result, severity=1)[0]][1]["delta"]
    three_delta = result["systems"][observed_leaders(result, severity=3)[0]][3]["delta"]
    return (f"Results for {result['n_requested']} SciFact queries, maximum edits per query: {c['severity']}.\n\n"
            f"- {observation}\n"
            f"- Individual Dense outcomes: {counts['Lower score']} lower, {counts['Same score']} equal, "
            f"and {counts['Higher score']} higher scores. Mean {label}: {dense['clean']:.4f} → "
            f"{dense['shifted']:.4f}.{failure}\n"
            f"- More typos: {one} loses the least at one edit (Δ{label}: {one_delta:+.4f}); "
            f"{three} loses the least at up to three edits (Δ{label}: {three_delta:+.4f}).")


def show_lesson_summary(result):
    from IPython.display import Markdown, display
    display(Markdown(lesson_summary(result)))
