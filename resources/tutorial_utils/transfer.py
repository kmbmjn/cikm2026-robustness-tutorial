"""M5 evidence-stage proxies over the cached corpus-insertion scenario.

The scenario has genuine MiniLM scores. Policies and context packing are
recomputed locally; no generator or detector is evaluated.
"""
import numpy as np
import pandas as pd
from .config import ValidationError
from .metrics import checked_ranking, recall

ATTACKS = ("none", "synthetic_spam", "drop_relevant_passage")
STRATEGIES = ("none", "filter", "source_priority", "retrieval_fallback")


def context_metrics(docs, clean_docs, qrels, injected_id, original_ids, k):
    """Keep slot, packed-document, and relevant-document denominators distinct."""
    docs, clean_docs = checked_ranking(docs), checked_ranking(clean_docs)
    if type(k) is not int or not 1 <= k <= 10 or len(docs) > k or len(clean_docs) > k:
        raise ValidationError("Pack at most K unique passages; K must be an integer from 1 to 10.")
    if any(not np.isfinite(v) or v < 0 for v in qrels.values()):
        raise ValidationError("Relevance grades must be finite and nonnegative.")
    count = docs.count(injected_id)
    relevant = sum(qrels.get(d, 0) > 0 for d in docs)
    return {"EvidenceRecall@K": recall(docs, qrels, k),
            "InjectedCount@K": count, "InjectedHit@K": int(count > 0),
            "InjectedExposure@K": count/k,
            "trusted_source_ratio": sum(d in original_ids for d in docs)/len(docs) if docs else np.nan,
            "trusted_displacement": len(set(clean_docs)-set(docs))/len(clean_docs) if clean_docs else np.nan,
            "packed_passages": len(docs), "judged_relevant_count": relevant,
            "judged_nonrelevant_count": sum(d in qrels and qrels[d] == 0 for d in docs),
            "unjudged_count": sum(d not in qrels for d in docs)}


def _policy(ranking, clean, original_ids, injected_id, strategy):
    if strategy == "filter":
        return [(d, s) for d, s in ranking if d != injected_id]
    if strategy == "source_priority":
        # Stable partition: preserve the scored order within each source group.
        return sorted(ranking, key=lambda ds: ds[0] not in original_ids)
    if strategy == "retrieval_fallback":
        return list(clean)
    return list(ranking)


def evidence_experiment(ctx, query_id, attack="synthetic_spam", strategy="none", k=5):
    if ctx.mode != "cached":
        raise ValidationError("M5 compares fixed cached snapshots. Keep MODE='cached'; live insertion inference is not implemented.")
    if query_id not in ctx.demo_ids or type(k) is not int or not 1 <= k <= 10:
        raise ValidationError(f"Evidence cache supports query ID strings {ctx.demo_ids} and integer K=1–10.")
    if attack not in ATTACKS or strategy not in STRATEGIES:
        raise ValidationError("Select a declared attack and strategy.")
    clean = list(ctx.runs["rerank_s0"][query_id])
    scenario=ctx.scenarios[query_id]["synthetic_spam"]
    sid=scenario["injected_id"]
    clean_candidates = checked_ranking(ctx.runs["dense_s0"][query_id])
    inserted_candidates = checked_ranking(scenario["shifted"])
    for ranking, candidates in [(clean, clean_candidates), (scenario["reranked"], inserted_candidates)]:
        if len(candidates) != 20 or set(checked_ranking(ranking)) != set(candidates):
            raise ValidationError("M5 requires the same 20 candidates within each condition's retrieval and reranking.")
    if attack == "synthetic_spam":
        attacked = list(scenario["reranked"])
    elif attack == "drop_relevant_passage":
        # Oracle diagnostic: remove ALL positive-qrel documents from the scored
        # pool AFTER reranking. Do not rerun retrieval or shrink the qrel target.
        attacked = [(d, s) for d, s in clean if ctx.qrels[query_id].get(d, 0) <= 0]
    else:
        attacked = list(clean)
    defended = _policy(attacked, clean, ctx.corpus, sid, strategy)
    clean_policy = _policy(clean, clean, ctx.corpus, sid, strategy)
    rankings = {"clean": clean, "attacked": attacked, "defended": defended, "clean+policy": clean_policy}
    rows, contexts, costs = [], {}, []
    original = checked_ranking(clean)[:k]
    for label, rank in rankings.items():
        docs = checked_ranking(rank)[:k]
        rows.append({"condition": label, **context_metrics(docs, original, ctx.qrels[query_id], sid, ctx.corpus, k)})
        records = []
        for i, d in enumerate(docs):
            text = scenario["injected_text"] if d == sid else ctx.corpus[d]
            records.append({"rank": i+1, "document_id": d,
                            "source": "synthetic" if d == sid else "original corpus",
                            "judgment": ("relevant" if ctx.qrels[query_id].get(d, 0) > 0 else
                                         "judged nonrelevant" if d in ctx.qrels[query_id] else "unjudged"),
                            "text": text, "excerpt": text[:200] + ("…" if len(text) > 200 else "")})
        contexts[label] = pd.DataFrame(records, columns=["rank", "document_id", "source", "judgment", "text", "excerpt"])
        paths = 1 + int(strategy == "retrieval_fallback" and label in {"defended", "clean+policy"})
        costs.append({"condition": label, "retrieval_paths": paths, "reranker_pairs": 20*paths,
                      "available_before_packing": len(rank), "packed_passages": len(docs)})
    removed = [d for d, _ in clean if d not in {d for d, _ in attacked}] if attack == "drop_relevant_passage" else []
    primary_candidates = inserted_candidates if attack == "synthetic_spam" else clean_candidates
    primary_reranked = checked_ranking(scenario["reranked"]) if attack == "synthetic_spam" else checked_ranking(clean)
    injection_ranks = {"dense_top20": primary_candidates.index(sid)+1 if sid in primary_candidates else None,
                       "reranked_top20": primary_reranked.index(sid)+1 if sid in primary_reranked else None}
    return {"metrics":pd.DataFrame(rows).set_index("condition"),"contexts":contexts,"k":k,"query_id":query_id,
            "config": {"query_id": query_id, "attack": attack, "strategy": strategy, "k": k},
            "rankings": rankings, "costs": pd.DataFrame(costs).set_index("condition"),
            "retrieved_contexts": {label: frame.to_dict("records") for label, frame in contexts.items()},
            "injected_id": sid, "injected_text": scenario["injected_text"] if attack == "synthetic_spam" else None,
            "removed_ids": removed, "positive_qrels": sum(v > 0 for v in ctx.qrels[query_id].values()),
            "injection_ranks": injection_ranks,
            "query": ctx.queries[query_id], "mode": ctx.mode,
            "budget":"20 retrieved / 20 reranked per primary path; post-ranking deletion/filtering does not refill candidates. Fallback models one additional 20/20 path to the retained clean snapshot.",
            "scope":"Evidence-stage proxies only. Source membership is not factual correctness."}
