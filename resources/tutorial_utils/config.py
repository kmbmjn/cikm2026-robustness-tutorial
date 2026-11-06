"""One explicit experiment contract, shared by all six notebooks."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "assets" / "caches"
DATA = ROOT / "data" / "scifact"
SEED = 42
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

class ValidationError(ValueError):
    """Invalid experiment or missing asset; never silently substitute a baseline."""

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def code_fingerprint():
    files = sorted((ROOT / "tutorial_utils").glob("*.py")) + [ROOT / "scripts/prepare_assets.py"]
    return fingerprint({str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()})

def mode_value(mode):
    mode = os.environ.get("TUTORIAL_MODE", mode)
    if mode not in {"cached", "live", "toy"}:
        raise ValidationError("MODE must be cached, live, or toy. Use cached for offline lessons.")
    return mode
