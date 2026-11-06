#!/usr/bin/env python3
"""Build genuine CPU inference caches. Explicit --rebuild; default verifies existing assets."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np
from tutorial_utils.config import CACHE, DATA, DATA_URL, ENCODER, RERANKER, sha256, fingerprint, code_fingerprint
from tutorial_utils.data import read_dataset, TutorialContext, verify_cache
from tutorial_utils.models import Models
from tutorial_utils.perturbations import make_query_typos, perturbation_accounting
from tutorial_utils.retrieval import rank_embeddings, BM25, top_scores
from tutorial_utils.metrics import ndcg
from tutorial_utils.interventions import build_vocabulary, input_normalization, vocab_spell_correction, select_adapter, apply_adapter

def write(name, data):
    (CACHE/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",default="scifact",choices=["scifact"])
    ap.add_argument("--model",default=ENCODER,choices=[ENCODER])
    ap.add_argument("--seed",type=int,default=42,choices=[42])
    ap.add_argument("--rebuild",action="store_true")
    ap.add_argument("--refresh-manifest",action="store_true",help="Update code fingerprint after reviewed code-only changes; rehash existing data.")
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    if args.refresh_manifest:
        m=verify_cache(); m["code_fingerprint"]=code_fingerprint()
        m["code_review_timestamp"]=datetime.now(timezone.utc).isoformat()
        write("manifest.json",m); print("Updated reviewed code fingerprint; inference timestamp unchanged."); return
    if (CACHE/"manifest.json").exists() and not args.rebuild:
        m=verify_cache(); print(f"Verified {len(m['files'])} checksums; cached SciFact assets ready."); return
    archive=ROOT/"data/scifact.zip"
    if not (DATA/"corpus.jsonl").exists():
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists(): urllib.request.urlretrieve(DATA_URL,archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest=(archive.parent/member.filename).resolve()
                if not dest.is_relative_to(archive.parent.resolve()): raise ValueError("Unsafe archive path")
            z.extractall(archive.parent)
    corpus, queries, qrels, splits=read_dataset()
    qids=sorted(set(splits["official_train"]+splits["test"]))
    dids=sorted(corpus)
    revpath=CACHE/"model_revisions.json"
    if revpath.exists(): revisions=json.loads(revpath.read_text())
    else:
        from huggingface_hub import HfApi
        revisions={m:HfApi().model_info(m).sha for m in [ENCODER,RERANKER]}; write("model_revisions.json",revisions)
    print("Loading pinned CPU models",revisions,flush=True)
    models=Models(revisions)
    arrays={}
    # Separate reusable checkpoints make a cancelled first preparation resumable.
    staging=CACHE/".preparation"; staging.mkdir(exist_ok=True)
    def encode(key,texts):
        path=staging/f"{key}.npy"
        signature=fingerprint([revisions[ENCODER],texts,256])
        meta=path.with_suffix(".json")
        if path.exists() and meta.exists() and json.loads(meta.read_text())==signature:
            arrays[key]=np.load(path,allow_pickle=False); print("Resume",key,flush=True)
        else:
            start=time.perf_counter(); arrays[key]=models.encode(texts,progress=True)
            np.save(path,arrays[key]); meta.write_text(json.dumps(signature))
            print(key,arrays[key].shape,round(time.perf_counter()-start,2),"seconds",flush=True)
    encode("doc",[corpus[d] for d in dids])
    encode("q_s0",[queries[q] for q in qids])
    variants={}
    for s in [1,2,3]:
        variants[str(s)]=make_query_typos({q:queries[q] for q in qids},s,42)
        encode(f"q_s{s}",[variants[str(s)][q]["text"] for q in qids])
    vocabulary=build_vocabulary(corpus)
    for s in range(4):
        texts=[queries[q] if not s else variants[str(s)][q]["text"] for q in qids]
        for method,fn in [("input_normalization",input_normalization),("vocab_spell_correction",lambda t:vocab_spell_correction(t,vocabulary))]:
            encode(f"{method}_s{s}",[fn(t) for t in texts])
    for seed in [42,43,44]:
        for s in [1,2,3]:
            key=f"train_s{s}_seed{seed}"
            if seed==42: arrays[key]=arrays[f"q_s{s}"]
            else:
                records=make_query_typos({q:queries[q] for q in splits["official_train"]},s,seed)
                # Test rows are zero padding, inaccessible to the train-only API.
                trids=splits["official_train"]
                encode(key,[records[q]["text"] for q in trids])
                full=np.zeros_like(arrays["q_s0"])
                for i,q in enumerate(trids): full[qids.index(q)]=arrays[key][i]
                arrays[key]=full
    manifest={"schema_version":1,"seed":42,"dtype":"float32","splits":splits,
              "query_ids":qids,"document_ids":dids,"model_revisions":revisions,
              "dataset":{"name":"BEIR SciFact","version":"official archive; SHA256 in files","source":DATA_URL,
                         "corpus_count":len(corpus),"test_query_count":len(splits["test"]),
                         "license":"SciFact claims/annotations CC BY 4.0; corpus abstracts ODC-By 1.0 (upstream LICENSE.md)"},
              "preprocessing":{"corpus":"title + newline + abstract","dense_max_tokens":256,"reranker_max_tokens":512,
                               "pooling":"masked mean including special tokens","normalization":"L2 epsilon=1e-12",
                               "ties":"score descending, document ID string ascending","bm25":"Unicode word regex, casefold; k1=1.2 b=0.75"},
              "perturbations":{"severity":[0,1,2,3],"seed":42,"algorithm":"per-ID SHA256 RNG; adjacent keyboard substitution/insertion/deletion",
                               "validity":"conditional, not automatically meaning preserving","training_seeds":[42,43,44]},
              "generated_at":datetime.now(timezone.utc).isoformat(),"device":"cpu","threads":4,
              "environment":{"python":platform.python_version(),"platform":platform.platform(),
                             **{p:importlib.metadata.version(p) for p in ["numpy","torch","transformers","sentence-transformers"]}}}
    ctx=TutorialContext("cached",42,corpus,queries,qrels,manifest,arrays,variants=variants)
    runs={}; bm=BM25(corpus)
    rerank_signature=fingerprint([revisions,sha256(DATA/"corpus.jsonl"),variants])
    for s in range(4):
        ids=splits["test"]
        runs[f"dense_s{s}"]=rank_embeddings(ctx,arrays[f"q_s{s}"][ctx.indices(ids)],ids,20)
        texts={q:queries[q] if not s else variants[str(s)][q]["text"] for q in ids}
        runs[f"bm25_s{s}"]={q:bm.retrieve(texts[q],20) for q in ids}
        rrfile=staging/f"rerank_s{s}.json"
        if rrfile.exists() and json.loads(rrfile.read_text()).get("signature")==rerank_signature:
            runs[f"rerank_s{s}"]=json.loads(rrfile.read_text())["runs"]
        else:
            rr={}; start=time.perf_counter()
            for i,q in enumerate(ids):
                docs=[d for d,v in runs[f"dense_s{s}"][q]]
                scores=models.score([(texts[q],corpus[d]) for d in docs])
                rr[q]=top_scores(scores,docs,20)
                if i%50==0: print("Rerank",s,i,"/",len(ids),flush=True)
            runs[f"rerank_s{s}"]=rr
            rrfile.write_text(json.dumps({"signature":rerank_signature,"runs":rr}))
            print("Rerank severity",s,"seconds",round(time.perf_counter()-start,2),flush=True)
    ctx.runs=runs
    delta={q:ndcg(runs["dense_s1"][q],qrels[q])-ndcg(runs["dense_s0"][q],qrels[q]) for q in ctx.test_ids}
    failure=min(delta,key=lambda q:(delta[q],q))
    if delta[failure]>=0: raise RuntimeError("No measured one-edit failure; inspect perturbations before release.")
    stable=min(delta,key=lambda q:(abs(delta[q]),-ndcg(runs["dense_s0"][q],qrels[q]),q))
    improved=max(delta,key=lambda q:(delta[q],q))
    manifest["demo_ids"]=list(dict.fromkeys([failure,stable,improved]))
    manifest["demo_selection"]={"status":"exploratory test-case selection, not a generalization estimate","failure":failure,"stable":stable,"improved":improved}
    print("Exploratory demo IDs",manifest["demo_selection"],flush=True)
    scenarios={}
    for q in manifest["demo_ids"]:
        old=runs["dense_s0"][q]
        spam_id="synthetic-spam-"+q
        spam_text="Synthetic teaching passage. "+queries[q]+" "+queries[q]+" This unverified claim is intentionally repeated; this is not scientific evidence."
        spam_vector=models.encode([spam_text])
        attacked=rank_embeddings(ctx,arrays["q_s0"][[qids.index(q)]],[q],20,
                                 doc_vectors=np.concatenate([arrays["doc"],spam_vector]),doc_ids=dids+[spam_id])[q]
        docs=[d for d,v in attacked]
        rr=top_scores(models.score([(queries[q],spam_text if d==spam_id else corpus[d]) for d in docs]),docs,20)
        scenarios[q]={"synthetic_spam":{"clean":old,"shifted":attacked,"reranked":rr,"label_valid":False,
             "injected_id":spam_id,"injected_text":spam_text,"synthetic":True,"qrel_status":"unjudged; synthetic origin alone does not establish irrelevance",
             "content_snapshot":"v1+one synthetic document","index_snapshot":"fresh","control":"remove known synthetic source; retain fixed candidate budget 20",
             "threat_model":{"surface":"corpus insertion","budget":1,"goal":"document exposure; no external service or generation"}}}
    # Matched historical CPU inference timing on a declared validation subset.
    selection=select_adapter(ctx)
    latency_ids=ctx.val_ids[:32]
    timings={"provenance":"historical CPU end-to-end query preprocessing + encoding + optional adapter + exact search; reranker off",
             "query_ids":latency_ids,"query_count":len(latency_ids),"repetitions":3,"batch_size":32,"warmup_batches":1,
             "platform":platform.platform(),"torch_threads":4,"blas_threads":2,"methods":{}}
    texts=[variants["1"][q]["text"] for q in latency_ids]
    for method in ["none","input_normalization","vocab_spell_correction","repair_adapter"]:
        elapsed=[]
        for rep in range(4):
            start=time.perf_counter()
            ts=[input_normalization(t) for t in texts] if method=="input_normalization" else [vocab_spell_correction(t,vocabulary) for t in texts] if method=="vocab_spell_correction" else texts
            z=models.encode(ts)
            if method=="repair_adapter": z=apply_adapter(z,selection["adapter"])
            rank_embeddings(ctx,z,latency_ids,20)
            if rep: elapsed.append((time.perf_counter()-start)*1000/len(texts))
        timings["methods"][method]={"ms_per_query":float(np.median(elapsed)),"replicate_ms_per_query":elapsed}
    timings["adapter_config"]={k:selection[k] for k in ["lambda","train_noise_severity","seed"]}
    manifest["reference_adapter"]={**timings["adapter_config"],"selection":selection["selection"]}
    manifest["perturbation_accounting"]={s:perturbation_accounting({q:r[q] for q in ctx.test_ids}) for s,r in variants.items()}
    np.savez_compressed(CACHE/"embeddings.npz",**arrays)
    for name,obj in [("runs.json",runs),("variants.json",variants),("scenarios.json",scenarios),("timings.json",timings)]: write(name,obj)
    manifest["archive_sha256"]=sha256(archive) if archive.exists() else None
    manifest["code_fingerprint"]=code_fingerprint()
    manifest["config_hash"]=fingerprint({k:manifest[k] for k in ["model_revisions","splits","preprocessing","perturbations"]})
    files=list(DATA.rglob("*.jsonl"))+list(DATA.rglob("*.tsv"))+[p for p in CACHE.iterdir() if p.is_file() and p.name!="manifest.json"]
    manifest["files"]={str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)}
    write("manifest.json",manifest)
    print("Verified genuine SciFact cache ready.",flush=True)

if __name__=="__main__": main()
