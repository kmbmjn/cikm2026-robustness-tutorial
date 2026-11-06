"""Exact retrieval, deterministic ties, BM25, and candidate-checked reranking."""
from collections import Counter
import re
import numpy as np
from .config import ValidationError
from .interventions import normalize_rows, apply_adapter, input_normalization, vocab_spell_correction, build_vocabulary

def top_scores(scores, ids, k):
    scores = np.asarray(scores)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all() or len(ids) != len(set(ids)):
        raise ValidationError("Scores must be finite and aligned to unique document IDs.")
    if not isinstance(k,int) or k < 1:
        raise ValidationError("top_k must be a positive integer.")
    order = np.lexsort((np.asarray(ids, str), -scores))[:k]
    return [(ids[i], float(scores[i])) for i in order]

def rank_embeddings(ctx, query_vectors, query_ids, top_k=20, doc_vectors=None, doc_ids=None):
    docs = ctx.embeddings["doc"] if doc_vectors is None else doc_vectors
    ids = ctx.doc_ids if doc_ids is None else doc_ids
    query_vectors = normalize_rows(query_vectors)
    if len(query_vectors) != len(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValidationError("Query vectors and unique IDs must align.")
    if not np.isfinite(docs).all():
        raise ValidationError("Nonfinite document embeddings.")
    scores = query_vectors @ docs.T
    return {q:top_scores(s,ids,top_k) for q,s in zip(query_ids,scores)}

class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.ids = sorted(corpus); self.k1, self.b = k1,b
        counts = [Counter(re.findall(r"\b\w+\b", corpus[d].casefold())) for d in self.ids]
        self.lengths = np.array([sum(c.values()) for c in counts])
        self.avg = max(float(self.lengths.mean()),1)
        self.postings = {}
        for i,c in enumerate(counts):
            for term,tf in c.items(): self.postings.setdefault(term,[]).append((i,tf))
    def retrieve(self, text, k=20):
        scores = np.zeros(len(self.ids))
        for term in sorted(set(re.findall(r"\b\w+\b", text.casefold()))):
            pairs = self.postings.get(term, [])
            if not pairs: continue
            idx, tf = np.array(pairs).T
            idf = np.log(1+(len(self.ids)-len(pairs)+0.5)/(len(pairs)+0.5))
            scores[idx] += idf*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths[idx]/self.avg))
        return top_scores(scores,self.ids,k)

def query_text(ctx, q, severity=0):
    if q not in ctx.query_ids:
        raise ValidationError(f"Unknown query {q}; choose a cached query ID.")
    if severity not in (0,1,2,3): raise ValidationError("Cached severity must be 0–3.")
    return ctx.queries[q] if severity == 0 else ctx.variants[str(severity)][q]["text"]

def retrieve(ctx, queries=None, method="dense", top_k=20, severity=0, intervention="none", adapter=None):
    ids = list(ctx.test_ids if queries is None else queries)
    if len(set(ids)) != len(ids): raise ValidationError("Duplicate query IDs.")
    ctx.indices(ids)
    if severity not in (0,1,2,3): raise ValidationError("Cached severity must be 0–3.")
    if method not in {"dense","bm25"}: raise ValidationError("Method must be dense or bm25.")
    if intervention not in {"none","input_normalization","vocab_spell_correction","repair_adapter"}:
        raise ValidationError("Unknown intervention; select a declared method.")
    if ctx.mode == "toy" and (severity or intervention != "none"):
        raise ValidationError("Toy smoke test supports clean retrieval only; it is not a SciFact replacement.")
    if method == "bm25":
        if intervention != "none": raise ValidationError("BM25 exercise uses no intervention.")
        key=f"bm25_s{severity}"
        if ctx.mode == "cached" and key in ctx.runs and all(q in ctx.runs[key] for q in ids) and top_k<=20:
            return {q:ctx.runs[key][q][:top_k] for q in ids}
        bm=BM25(ctx.corpus)
        return {q:bm.retrieve(query_text(ctx,q,severity),top_k) for q in ids}
    key = f"q_s{severity}" if intervention in {"none","repair_adapter"} else f"{intervention}_s{severity}"
    if ctx.mode == "live":
        from .models import live_models
        texts = [query_text(ctx,q,severity) for q in ids]
        if intervention == "input_normalization": texts = [input_normalization(t) for t in texts]
        if intervention == "vocab_spell_correction":
            vocab=build_vocabulary(ctx.corpus); texts=[vocab_spell_correction(t,vocab) for t in texts]
        vectors = live_models(ctx).encode(texts)
    else:
        if key not in ctx.embeddings: raise ValidationError(f"Missing cache {key}; regenerate, do not substitute clean inputs.")
        vectors = ctx.embeddings[key][ctx.indices(ids)]
    if intervention == "repair_adapter":
        if adapter is None: raise ValidationError("repair_adapter requires a train-only fitted adapter.")
        vectors=apply_adapter(vectors,adapter)
    return rank_embeddings(ctx,vectors,ids,top_k)

def rerank(ctx, rankings, top_k=10, severity=0):
    if not 1 <= top_k <= 20: raise ValidationError("Rerank output K must be 1–20.")
    result={}
    for q, ranking in rankings.items():
        docs=[d for d,s in ranking]
        if len(docs)!=20 or len(set(docs))!=20:
            raise ValidationError("Reranker contract requires exactly 20 distinct candidates.")
        if ctx.mode == "live":
            from .models import live_models
            scores=live_models(ctx).score([(query_text(ctx,q,severity),ctx.corpus[d]) for d in docs])
            result[q]=top_scores(scores,docs,top_k)
        else:
            key=f"rerank_s{severity}"
            if key not in ctx.runs or q not in ctx.runs[key]: raise ValidationError("No cached reranking for this query/config; use live.")
            cached=ctx.runs[key][q]
            if set(docs)!={d for d,s in cached}: raise ValidationError("Cached reranker candidates differ. Recompute in live mode.")
            result[q]=cached[:top_k]
    return result
