"""Attack-and-response displays for M5, using the shared transfer calculations."""
from html import escape
import textwrap

import pandas as pd

from . import transfer
from .presentation import interpretation_html, show_interpretation


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else \
        ", ".join(items[:-1]) + ", and " + items[-1]


def _count(result, condition, column):
    return int(result["metrics"].loc[condition, column])


def _note(text):
    from IPython.display import HTML, display
    display(HTML(f"<p>{escape(text)}</p>"))


def _table(frame, caption, interpretation):
    from IPython.display import display
    # Scope readable table typography to M5's tables.
    html = frame.to_html(index=False, border=0, classes="m5-table", escape=True,
                         float_format=lambda value: f"{value:.3f}")
    opening = html.index(">")+1
    html = html[:opening] + f"<caption>{escape(caption)}</caption>" + html[opening:]
    style = """<style>
    table.m5-table {font-size:16px; line-height:1.45; border-collapse:collapse;
                    margin:12px 0; max-width:100%;}
    table.m5-table th, table.m5-table td {text-align:left !important; vertical-align:top;
                    padding:8px 10px; white-space:normal; overflow-wrap:normal;}
    table.m5-table th:first-child, table.m5-table td:first-child {
                    min-width:3em; white-space:nowrap;}
    table.m5-table caption {font-size:16px; text-align:left; padding:0 0 8px;}
    </style>"""
    display({"text/html": style+html+interpretation_html(interpretation)}, raw=True)


def _kept(result, condition):
    count = int(result["metrics"].loc[condition, "judged_relevant_count"])
    return f"{count} / {result['positive_qrels']}"


def _counts(result, condition):
    """Split the K selected references into attack documents, evidence, and other papers."""
    metrics = result["metrics"].loc[condition]
    attack, relevant = int(metrics["InjectedCount@K"]), int(metrics["judged_relevant_count"])
    return {"Attack documents": attack, "Evidence kept": _kept(result, condition),
            "Other papers": int(metrics["packed_passages"]) - attack - relevant,
            "Attack share": f"{metrics['InjectedExposure@K']:.0%}"}


def _summary(result, conditions):
    return pd.DataFrame([{"Context": label, **_counts(result, condition)}
                         for condition, label in conditions])


def show_insertion(ctx, query_id="1303", k=1):
    result = transfer.evidence_experiment(ctx, query_id, "synthetic_spam", "none", k)
    _note(f"Selected test query: {result['query']}")

    def label(record):
        if record["source"] == "synthetic":
            return "Attack document: repeats the target query twice"
        title = textwrap.shorten(record["text"].split("\n")[0], width=65, placeholder="…")
        return f"Relevant evidence: {title}" if record["judgment"] == "relevant" else title

    contexts = result["retrieved_contexts"]
    rows = []
    for index in range(max(len(contexts[c]) for c in ("clean", "attacked"))):
        row = {"Rank": index+1}
        for condition, heading in (("clean", "Before attack"), ("attacked", "After attack")):
            row[heading] = label(contexts[condition][index]) if index < len(contexts[condition]) else "—"
        rows.append(row)
    metrics = result["metrics"]
    _table(pd.DataFrame(rows), f"Reference documents selected for the LLM (K={k})",
           f"At K={k}, the attack changes evidence kept from {_kept(result, 'clean')} "
           f"to {_kept(result, 'attacked')} and attack documents from "
           f"{int(metrics.loc['clean', 'InjectedCount@K'])} to {int(metrics.loc['attacked', 'InjectedCount@K'])}.")
    ranks = result["injection_ranks"]
    _note(f"The attack document reaches search rank {ranks['dense_top20']} "
          f"and reranked position {ranks['reranked_top20']}.")
    attacks = _count(result, "attacked", "InjectedCount@K")
    lost = _count(result, "clean", "judged_relevant_count") - _count(result, "attacked", "judged_relevant_count")
    if attacks and lost > 0:
        show_interpretation(
            "With only one reference slot, the attack takes it, so the LLM would read the attack passage "
            "instead of the relevant paper." if k == 1 else
            f"The attack takes {attacks} of the {k} reference slots and pushes out {lost} relevant "
            f"paper{'s' if lost != 1 else ''}, so the LLM would read the attack instead of that evidence.")
    elif attacks:
        show_interpretation(f"The attack takes {attacks} of the {k} reference slot{'s' if k != 1 else ''}, "
                            "but the known evidence stays in the references.")
    else:
        show_interpretation(f"The attack does not reach the first {k} reference{'s' if k != 1 else ''}, "
                            "so the LLM's input is unchanged.")
    return result


def show_budget_sweep(ctx, query_id="1303"):
    experiments = {k: transfer.evidence_experiment(ctx, query_id, "synthetic_spam", "none", k)
                   for k in (1, 5, 10)}
    frame = pd.DataFrame([{"References for the LLM (K)": k, **_counts(result, "attacked")}
                          for k, result in experiments.items()])
    kept = ", ".join(_kept(result, "attacked") for result in experiments.values())
    attacks = ", ".join(str(int(result["metrics"].loc["attacked", "InjectedCount@K"]))
                        for result in experiments.values())
    _table(frame, f"Reference contents at K=1/5/10: {experiments[1]['query']}",
           f"At K=1/5/10, evidence kept is {kept} and attack counts are {attacks}; "
           "increasing K does not filter out the attack.")
    _note("Evidence kept = relevant papers included / all known relevant papers in the collection for this claim.")
    counts = [_count(result, "attacked", "InjectedCount@K") for result in experiments.values()]
    evidence = [_count(result, "attacked", "judged_relevant_count") for result in experiments.values()]
    shares = [f"{result['metrics'].loc['attacked', 'InjectedExposure@K']:.0%}" for result in experiments.values()]
    if all(counts):
        back = (f"brings the evidence back ({_kept(experiments[1], 'attacked')} at K=1, "
                f"{_kept(experiments[10], 'attacked')} at K=10) but " if evidence[-1] > evidence[0] else "")
        show_interpretation(f"A larger K {back}never removes the attack; it only shrinks the attack's share "
                            f"of the references from {shares[0]} to {shares[-1]}.")
    else:
        show_interpretation("The attack is missing from at least one reference set here, so check its rank "
                            "before relying on K alone.")
    return {"table": frame, "experiments": experiments}


def show_filter(ctx, query_id="1303", k=5):
    result = transfer.evidence_experiment(ctx, query_id, "synthetic_spam", "filter", k)
    _table(_summary(result, [("clean", "Before attack"), ("attacked", "After attack"),
                             ("defended", "After filtering")]),
           f"References after attack filtering (K={k}): {result['query']}",
           f"Filtering changes attack documents from "
           f"{int(result['metrics'].loc['attacked', 'InjectedCount@K'])} to "
           f"{int(result['metrics'].loc['defended', 'InjectedCount@K'])} and evidence kept from "
           f"{_kept(result, 'attacked')} to {_kept(result, 'defended')}.")
    ids = {condition: [record["document_id"] for record in records]
           for condition, records in result["retrieved_contexts"].items()}
    if ids["defended"] == ids["clean"]:
        _note("The filter restores the same reference papers selected before the attack.")
    references = "single reference" if k == 1 else f"{k} references"
    if ids["defended"] == ids["clean"]:
        show_interpretation(f"Removing the attack before choosing the {references} gives the LLM exactly "
                            f"the {'paper' if k == 1 else 'papers'} it would have read without the attack.")
    elif not _count(result, "defended", "InjectedCount@K"):
        same = (_count(result, "defended", "judged_relevant_count") >= _count(result, "clean", "judged_relevant_count"))
        show_interpretation(f"The filter keeps the attack out of the {references}"
                            + (" and keeps the evidence, but some other references differ from the pre-attack set."
                               if same else ", but some evidence from the pre-attack references is missing."))
    else:
        show_interpretation("The filter does not remove every attack document, so the LLM would still read one.")
    return result


def show_reference(ctx, query_id="1303", k=5):
    """Keep complete controls and policy costs after the lesson summary."""
    filtered = transfer.evidence_experiment(ctx, query_id, "synthetic_spam", "filter", k)
    metrics = filtered["metrics"]
    columns = {"EvidenceRecall@K": "Evidence recall", "InjectedCount@K": "Attack count",
               "InjectedHit@K": "Attack hit", "InjectedExposure@K": "Attack share",
               "trusted_source_ratio": "Original-source ratio",
               "trusted_displacement": "Original references lost (fraction)"}
    _table(metrics[list(columns)].rename(columns=columns).rename_axis("Condition").reset_index(),
           f"Before attack, after attack, and after filtering, K={k}: {filtered['query']}",
           f"After filtering: evidence recall {metrics.loc['defended', 'EvidenceRecall@K']:.3f}, "
           f"attack share {metrics.loc['defended', 'InjectedExposure@K']:.3f}. "
           "Without an attack, this filter leaves the references unchanged.")
    attacks = _count(filtered, "attacked", "InjectedCount@K")
    replaced = round(metrics.loc["attacked", "trusted_displacement"]*k)
    restored = all(abs(metrics.loc[row, column]-metrics.loc["clean", column]) <= 1e-12
                   for row in ("defended", "clean+policy") for column in columns)
    slots = "the only slot" if k == 1 and attacks == 1 else f"{attacks} of the {k} slot{'s' if k != 1 else ''}"
    attack_text = (f"The attack fills {slots} and replaces {replaced} "
                   f"original reference{'s' if replaced != 1 else ''}")
    show_interpretation(attack_text + ("; filtering restores every metric to its clean value, and without an "
                                       "attack the filter changes nothing." if restored else
                                       "; filtering does not restore every metric to its clean value."))
    rows = []
    for strategy in transfer.STRATEGIES:
        result = transfer.evidence_experiment(ctx, query_id, "synthetic_spam", strategy, k)
        work = result["costs"].loc["defended"]
        rows.append({"Policy": strategy, "Evidence kept": _kept(result, "defended"),
                     "Attack documents": int(result["metrics"].loc["defended", "InjectedCount@K"]),
                     "Search paths": int(work.retrieval_paths), "Reranker pairs": int(work.reranker_pairs)})
    policies = pd.DataFrame(rows)
    safe = ", ".join(policies.loc[policies["Attack documents"] == 0, "Policy"].str.replace("_", " ", regex=False))
    fallback = policies.set_index("Policy").loc["retrieval_fallback"]
    _table(policies.assign(Policy=policies["Policy"].str.replace("_", " ", regex=False)),
           f"Policies and search work after a document attack, K={k}: {filtered['query']}",
           f"{safe} keep the attack out of the selected references; retrieval fallback uses "
           f"{int(fallback['Search paths'])} search paths and {int(fallback['Reranker pairs'])} reranker pairs.")
    show_interpretation(_policy_interpretation(policies, _kept(filtered, "clean")))
    _note("Inserted document (cached experiment input, verbatim): " + filtered["injected_text"])
    return {"filter": filtered, "policies": policies}


def _policy_interpretation(policies, clean_kept):
    """Which policies block the attack while keeping the evidence, and at what search cost."""
    rows = policies.assign(Policy=policies["Policy"].str.replace("_", " ", regex=False)).set_index("Policy")
    defenses = [p for p in rows.index if p != "none"]
    safe = [p for p in defenses if rows.loc[p, "Attack documents"] == 0 and rows.loc[p, "Evidence kept"] == clean_kept]
    lead = ("Without a policy the attack stays in; " if rows.loc["none", "Attack documents"] else
            "The attack does not reach the references even without a policy; ")
    if not safe:
        return lead + "no policy both removes the attack and keeps all of the evidence here."
    cheapest = min(rows.loc[p, "Reranker pairs"] for p in safe)
    cheap = [p for p in safe if rows.loc[p, "Reranker pairs"] == cheapest]
    costly = [p for p in safe if rows.loc[p, "Reranker pairs"] > cheapest]
    every = "both" if len(safe) == 2 else "all" if len(safe) > 2 else ""
    text = lead + f"{_join(safe)} {every + ' ' if every else ''}remove{'s' if len(safe) == 1 else ''} it and keep{'s' if len(safe) == 1 else ''} the evidence"
    if not costly:
        return text + " with the same search work."
    ratio = max(rows.loc[p, "Reranker pairs"] for p in costly)/cheapest
    work = "twice" if ratio == 2 else f"{ratio:g} times"
    return (text + f", but {_join(costly)} needs {work} the search work of {_join(cheap)} "
            f"({int(rows.loc[costly[0], 'Search paths'])} search paths and "
            f"{int(rows.loc[costly[0], 'Reranker pairs'])} reranker pairs).")
