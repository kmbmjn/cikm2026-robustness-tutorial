"""BEIR/trec_eval linear-gain nDCG; explicit zero-IDCG accounting."""
import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from .config import ValidationError

def checked_ranking(ranking):
    ids = [str(x[0]) if isinstance(x, (list, tuple)) else str(x) for x in ranking]
    if len(ids) != len(set(ids)):
        raise ValidationError("Duplicate document IDs in a ranking; fix the run before evaluation.")
    return ids

def cutoff(k):
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValidationError("k must be a positive integer.")

def ndcg(ranking, qrels, k=10):
    cutoff(k)
    ids = checked_ranking(ranking)
    if any(not np.isfinite(v) or v < 0 for v in qrels.values()):
        raise ValidationError("Relevance grades must be finite and nonnegative.")
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if not idcg:
        return float("nan")
    return sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ids[:k])) / idcg

def recall(ranking, qrels, k=10):
    cutoff(k)
    ids = checked_ranking(ranking)
    relevant = {d for d, g in qrels.items() if g > 0}
    return len(set(ids[:k]) & relevant) / len(relevant) if relevant else float("nan")

def rbo(a, b, k=10, p=0.9):
    """Normalized finite-depth weighted overlap (not extrapolated RBO_EXT).

    sum_{d=1..k} (1-p)p^(d-1) overlap@d / (1-p^k).
    Short lists stop at min(k, max length); overlap at depth d divides by d.
    """
    cutoff(k)
    if not 0 < p < 1:
        raise ValidationError("RBO p must lie strictly between 0 and 1.")
    a, b = checked_ranking(a)[:k], checked_ranking(b)[:k]
    depth = min(k, max(len(a), len(b)))
    if depth == 0:
        return float("nan")
    # Cancel the common (1-p) analytically. Subtracting p**depth from 1
    # loses precision near p=1 and can give identity rankings a nonzero loss.
    weights = [p**(d-1) for d in range(1, depth+1)]
    overlaps = [len(set(a[:d]) & set(b[:d])) / d for d in range(1, depth+1)]
    return math.fsum(w*a_d for w, a_d in zip(weights, overlaps)) / math.fsum(weights)

def common_kendall(a, b, k=10):
    cutoff(k)
    a, b = checked_ranking(a)[:k], checked_ranking(b)[:k]
    common = sorted(set(a) & set(b))
    tau = float(kendalltau([a.index(x) for x in common], [b.index(x) for x in common]).statistic) if len(common) >= 2 else float("nan")
    return {"tau": tau, "common_count": len(common), "coverage": len(common) / k}

def paired_bootstrap(clean, shifted, seed=42, n_boot=500):
    a, b = np.asarray(clean, float), np.asarray(shifted, float)
    if a.shape != b.shape or a.ndim != 1 or not 1 <= n_boot <= 10000:
        raise ValidationError("Paired bootstrap requires equal 1-D arrays and 1–10000 resamples.")
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid] - a[valid]
    if not len(delta):
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    indices = np.random.default_rng(seed).integers(0, len(delta), (n_boot, len(delta)))
    low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    return dict(estimate=float(delta.mean()), low=float(low), high=float(high), n=len(delta))

def evaluate_suite(ctx, clean_runs, shifted_runs, query_ids=None, k=10, p=0.9,
                   tail_fraction=0.1, label_valid=True, n_boot=500):
    cutoff(k)
    if not label_valid:
        raise ValidationError("Changed corpus labels are unverified. Use rank drift/exposure, not nDCG.")
    if not 0 < tail_fraction <= 1:
        raise ValidationError("tail_fraction must be in (0, 1].")
    ids = list(query_ids if query_ids is not None else ctx.test_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValidationError("Query IDs must be nonempty and unique.")
    if set(clean_runs) != set(shifted_runs) or not set(ids) <= set(clean_runs):
        raise ValidationError("Clean and shifted runs must contain the same query IDs.")
    rows = []
    for q in ids:
        if q not in ctx.qrels:
            raise ValidationError(f"Missing qrels for query {q}; select an official judged query.")
        a, b, rel = clean_runs[q], shifted_runs[q], ctx.qrels[q]
        ka = common_kendall(a, b, k)
        row = {"query_id": q, "clean_ndcg": ndcg(a, rel, k), "shifted_ndcg": ndcg(b, rel, k),
               "clean_recall": recall(a, rel, k), "shifted_recall": recall(b, rel, k),
               "rbo": rbo(a, b, k, p), **ka,
               "judged_coverage": sum(d in rel for d in checked_ranking(b)[:k]) / k}
        row["delta_ndcg"] = row["shifted_ndcg"] - row["clean_ndcg"]
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("query_id")
    valid = frame.dropna(subset=["clean_ndcg", "shifted_ndcg"])
    n_tail = max(1, math.ceil(len(valid) * tail_fraction)) if len(valid) else 0
    summary = {c: float(frame[c].mean()) for c in frame.columns}
    summary.update({"n_requested": len(ids), "n_evaluated": len(valid), "zero_idcg_excluded": len(ids)-len(valid),
                    "tail_count": n_tail, "tail_fraction": tail_fraction,
                    "bottom_shifted_ndcg": float(valid.shifted_ndcg.nsmallest(n_tail).mean()),
                    "worst_paired_delta": float(valid.delta_ndcg.nsmallest(n_tail).mean()),
                    "k": k, "p": p})
    return {"per_query": frame, "summary": summary,
            "ci": paired_bootstrap(frame.clean_ndcg, frame.shifted_ndcg, ctx.seed, n_boot),
            "query_ids": ids, "label_valid": True}
