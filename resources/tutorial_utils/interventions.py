"""Unsupervised input cleanup and a train-only residual ridge adapter."""
from collections import Counter
import re
import unicodedata
import numpy as np
from .config import ValidationError

def normalize_rows(x, epsilon=1e-12):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValidationError("Embeddings must be a finite 2-D float32 array.")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), epsilon)

def input_normalization(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def build_vocabulary(corpus):
    return Counter(w for doc in corpus.values() for w in re.findall(r"[a-z]{4,}", doc.lower()))

def vocab_spell_correction(text, vocabulary):
    def correct(match):
        word = match.group()
        if not word.islower() or len(word) < 5 or word in vocabulary:
            return word
        splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
        edits = {a+b[1:] for a,b in splits if b}
        edits |= {a+c+b[1:] for a,b in splits if b for c in "abcdefghijklmnopqrstuvwxyz"}
        edits |= {a+c+b for a,b in splits for c in "abcdefghijklmnopqrstuvwxyz"}
        known = [x for x in edits if vocabulary.get(x, 0) >= 3]
        return min(known, key=lambda x: (-vocabulary[x], x)) if known else word
    return re.sub(r"\b[A-Za-z]+\b", correct, text)

def fit_repair_adapter(noisy, clean, lambda_reg=1.0, *, query_ids, allowed_train_ids, test_ids=()):
    x, y = np.asarray(noisy, np.float64), np.asarray(clean, np.float64)
    ids = list(query_ids)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 384 or len(ids) != len(x):
        raise ValidationError("Adapter needs aligned N×384 clean/noisy embeddings and training IDs.")
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(allowed_train_ids) or set(ids) & set(test_ids):
        raise ValidationError("Adapter fit accepts only unique derived training query IDs; no val/test IDs.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(lambda_reg) or lambda_reg <= 0:
        raise ValidationError("Adapter inputs must be finite and lambda_reg strictly positive.")
    # Row-vector convention: X @ A.T approximates Y-X.
    a = np.linalg.solve(x.T @ x + lambda_reg*np.eye(384), x.T @ (y-x)).T
    return a.astype(np.float32)

def apply_adapter(x, a):
    a = np.asarray(a, np.float32)
    if a.shape != (384, 384) or not np.isfinite(a).all():
        raise ValidationError("Adapter must be a finite 384×384 matrix.")
    return normalize_rows(np.asarray(x, np.float32) @ (np.eye(384, dtype=np.float32)+a).T)

def select_adapter(ctx, lambdas=(0.1, 1., 10., 100.), train_noise_severity=1, seed=42,
                   *, validation_severity=None, train_ids=None, validation_ids=None):
    """Fit on train; select on validation, with an independently fixed stress level.

    None matches the validation noise level to the training noise level.
    Experiments comparing training noise levels must pass validation_severity.
    Validation corruptions always use the fixed context seed, not the training seed.
    Optional ID subsets support audited exclusions without changing source splits
    or the defaults used by cache preparation.
    """
    from .retrieval import rank_embeddings
    from .metrics import ndcg
    import pandas as pd
    if type(train_noise_severity) is not int or train_noise_severity not in (1, 2, 3) or type(seed) is not int or seed not in (42, 43, 44):
        raise ValidationError("Training cache supports severity 1–3 and seeds 42, 43, 44.")
    validation_severity = train_noise_severity if validation_severity is None else validation_severity
    if type(validation_severity) is not int or validation_severity not in range(4):
        raise ValidationError("validation_severity must be 0, 1, 2, or 3; keep it fixed when comparing training configurations.")
    try:
        lambdas = tuple(float(value) for value in lambdas)
    except (TypeError, ValueError) as error:
        raise ValidationError("Use a nonempty grid of distinct, finite, positive lambdas.") from error
    if not lambdas or len(set(lambdas)) != len(lambdas) or not all(np.isfinite(v) and v > 0 for v in lambdas):
        raise ValidationError("Use a nonempty grid of distinct, finite, positive lambdas.")
    if (set(ctx.train_ids) & set(ctx.val_ids) or set(ctx.train_ids) & set(ctx.test_ids)
            or set(ctx.val_ids) & set(ctx.test_ids) or not ctx.train_ids or not ctx.val_ids):
        raise ValidationError("Adapter selection requires nonempty, disjoint train/validation/test splits.")
    fit_ids = list(ctx.train_ids if train_ids is None else train_ids)
    val_ids = list(ctx.val_ids if validation_ids is None else validation_ids)
    for ids, allowed in [(fit_ids, ctx.train_ids), (val_ids, ctx.val_ids)]:
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(allowed):
            raise ValidationError("Adapter ID subsets must be nonempty, unique, and stay within their original split.")
    tr, va = ctx.indices(fit_ids), ctx.indices(val_ids)
    key = f"train_s{train_noise_severity}_seed{seed}"
    if key not in ctx.embeddings:
        raise ValidationError("Training cache supports severity 1–3 and seeds 42, 43, 44.")
    x, y = ctx.embeddings[key][tr], ctx.embeddings["q_s0"][tr]
    rows, models = [], {}
    for lam in lambdas:
        a = fit_repair_adapter(x, y, lam, query_ids=fit_ids,
                               allowed_train_ids=fit_ids, test_ids=ctx.test_ids)
        models[float(lam)] = a
        zc, zs = ctx.embeddings["q_s0"][va], ctx.embeddings[f"q_s{validation_severity}"][va]
        cr = rank_embeddings(ctx, apply_adapter(zc, a), val_ids, 10)
        sr = rank_embeddings(ctx, apply_adapter(zs, a), val_ids, 10)
        rows.append({"lambda": lam, "validation_clean_ndcg": np.mean([ndcg(cr[q], ctx.qrels[q]) for q in val_ids]),
                     "validation_shifted_ndcg": np.mean([ndcg(sr[q], ctx.qrels[q]) for q in val_ids]),
                     "train_mse": float(np.mean((x @ (np.eye(384)+a).T-y)**2)),
                     "validation_mse": float(np.mean((zs @ (np.eye(384)+a).T-zc)**2)),
                     "regularized_objective": float(np.sum((x @ (np.eye(384)+a).T-y)**2)+lam*np.sum(a*a))})
    frame = pd.DataFrame(rows).sort_values(["validation_shifted_ndcg", "validation_clean_ndcg", "lambda"], ascending=[False, False, True])
    chosen = float(frame.iloc[0]["lambda"])
    return {"adapter": models[chosen], "lambda": chosen, "validation": frame,
            "selection": "validation shifted nDCG, then clean nDCG; no test selection",
            "train_noise_severity": train_noise_severity, "seed": seed,
            "validation_severity": validation_severity, "lambda_grid": lambdas,
            "fit_query_ids": fit_ids, "validation_query_ids": val_ids}
