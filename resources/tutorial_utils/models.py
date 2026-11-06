"""Lazy optional CPU inference; no model import or network use in cached mode."""
import numpy as np
from .config import ENCODER, RERANKER, ValidationError
from .interventions import normalize_rows

class Models:
    def __init__(self, revisions, threads=4):
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        self.torch = torch
        self.encoder = SentenceTransformer(ENCODER, revision=revisions[ENCODER], device="cpu")
        self.encoder.max_seq_length = 256
        self.encoder.eval()
        self.reranker = CrossEncoder(RERANKER, revision=revisions[RERANKER], device="cpu", max_length=512)
        self.reranker.model.eval()

    def encode(self, texts, progress=False):
        if any(not isinstance(t, str) for t in texts):
            raise ValidationError("Texts must be strings; empty strings are encoded explicitly.")
        return normalize_rows(self.encoder.encode(list(texts), batch_size=32, show_progress_bar=progress,
                                                  convert_to_numpy=True, normalize_embeddings=False))

    def score(self, pairs):
        # Raw cross-encoder logits, not probabilities.
        return np.asarray(self.reranker.predict(pairs, batch_size=32, show_progress_bar=False,
                           activation_fct=self.torch.nn.Identity()), dtype=np.float32).reshape(-1)

def live_models(ctx):
    if ctx.mode != "live":
        raise ValidationError("Live inference requires MODE='live' and requirements-live.txt.")
    if not hasattr(ctx, "_models"):
        ctx._models = Models(ctx.manifest["model_revisions"])
    return ctx._models
