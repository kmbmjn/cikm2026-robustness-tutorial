"""Read-only SciFact data and checksum-verified cached experiment context."""
import csv
import json
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np
from .config import CACHE, DATA, ROOT, SEED, ValidationError, mode_value, sha256, fingerprint

def read_dataset(path=DATA):
    def jsonl(name):
        with open(path / name, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    corpus = {str(d["_id"]): d.get("title", "") + "\n" + d.get("text", "") for d in jsonl("corpus.jsonl")}
    queries = {str(q["_id"]): q["text"] for q in jsonl("queries.jsonl")}
    splits, qrels = {}, {}
    for split in ("train", "test"):
        rel = {}
        with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
        splits[split] = sorted(rel)
        if set(qrels) & set(rel):
            raise ValidationError("Official train/test IDs overlap; stop before fitting anything.")
        qrels.update(rel)
    ordered = sorted(splits["train"], key=lambda q: (fingerprint([SEED, q]), q))
    n_val = max(1, round(len(ordered) * 0.2))
    splits["validation"] = sorted(ordered[:n_val])
    splits["official_train"] = splits["train"]
    splits["train"] = sorted(ordered[n_val:])
    return corpus, queries, qrels, splits

@dataclass
class TutorialContext:
    mode: str
    seed: int
    corpus: dict
    queries: dict
    qrels: dict
    manifest: dict
    embeddings: dict
    runs: dict = field(default_factory=dict)
    variants: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def train_ids(self): return self.manifest["splits"]["train"]
    @property
    def val_ids(self): return self.manifest["splits"]["validation"]
    @property
    def test_ids(self): return self.manifest["splits"]["test"]
    @property
    def query_ids(self): return self.manifest["query_ids"]
    @property
    def doc_ids(self): return self.manifest["document_ids"]
    @property
    def demo_ids(self): return self.manifest.get("demo_ids", self.test_ids[:1])
    def indices(self, ids):
        lookup = {q:i for i,q in enumerate(self.query_ids)}
        try: return [lookup[q] for q in ids]
        except KeyError as e: raise ValidationError(f"Unknown query ID {e}. Choose an official ID.") from e
    def banner(self):
        label = "TOY — NOT SciFact benchmark" if self.mode == "toy" else f"SciFact | {len(self.corpus)} documents | {len(self.test_ids)} test queries"
        if self.mode == "toy":
            return f"MODE=toy | {label} | Illustrative toy example; invented 384-D features, not MiniLM inference."
        return (f"MODE={self.mode} | {label}\n"
                "BM25 / MiniLM dense exact retrieval → MiniLM cross-encoder (20 candidates)")

def verify_cache(cache=CACHE):
    path = cache / "manifest.json"
    if not path.exists():
        raise ValidationError("Missing SciFact cache. Restore resources/assets/caches + resources/data/scifact, run resources/scripts/prepare_assets.py, or explicitly choose toy mode.")
    m = json.loads(path.read_text())
    if m.get("schema_version") != 1 or m.get("dtype") != "float32" or m.get("seed") != 42:
        raise ValidationError("Unsupported cache schema/dtype/seed; regenerate the declared cache.")
    for relative, digest in m["files"].items():
        p = ROOT / relative
        if not p.exists() or sha256(p) != digest:
            raise ValidationError(f"Cache checksum mismatch: {relative}. Restore or regenerate assets.")
    revisions=json.loads((cache/"model_revisions.json").read_text())
    if m.get("model_revisions") != revisions:
        raise ValidationError("Model provenance mismatch between manifest and pinned revision artifact.")
    expected=fingerprint({k:m[k] for k in ["model_revisions","splits","preprocessing","perturbations"]})
    if expected != m.get("config_hash"):
        raise ValidationError("Cache config fingerprint mismatch; restore the declared manifest.")
    return m

def load_tutorial_context(mode="cached", dataset="scifact", seed=42, verify=True):
    mode = mode_value(mode)
    if dataset != "scifact" or seed != 42:
        raise ValidationError("Declared context supports dataset=scifact, seed=42. Adapter training seeds are separate.")
    if mode == "toy":
        return toy_context(seed)
    manifest = verify_cache() if verify else json.loads((CACHE / "manifest.json").read_text())
    corpus, queries, qrels, splits = read_dataset()
    if splits != manifest["splits"]:
        raise ValidationError("Split manifest mismatch; do not train or report until assets are restored.")
    if manifest["document_ids"] != sorted(corpus) or manifest["query_ids"] != sorted(set(splits["official_train"]+splits["test"])):
        raise ValidationError("Cache document/query row ordering does not match the source dataset.")
    with np.load(CACHE / "embeddings.npz", allow_pickle=False) as f:
        embeddings = {k:f[k] for k in f.files}
    for a in embeddings.values():
        if a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValidationError("Nonfinite or non-float32 cache array.")
        a.flags.writeable = False
    read = lambda name: json.loads((CACHE / name).read_text())
    return TutorialContext(mode, seed, MappingProxyType(corpus), MappingProxyType(queries),
        MappingProxyType({q: MappingProxyType(r) for q,r in qrels.items()}), manifest, embeddings,
        read("runs.json"), read("variants.json"), read("scenarios.json"), read("timings.json"))

def toy_context(seed=42):
    corpus = {"a":"Illustrative toy example: blue birds eat seeds", "b":"Illustrative toy example: red fish swim",
              "c":"Illustrative toy example: green trees grow"}
    queries = {"toy-q":"blue birds"}
    # Deliberately invented features; never represented as MiniLM inference.
    d = np.zeros((3,384), np.float32); d[:,:3] = np.eye(3)
    q = d[:1].copy()
    manifest = {"splits":{"train":[], "validation":[], "test":["toy-q"]}, "query_ids":["toy-q"],
                "document_ids":list(corpus), "source":"Illustrative toy example"}
    return TutorialContext("toy", seed, corpus, queries, {"toy-q":{"a":1}}, manifest, {"doc":d, "q_s0":q})
