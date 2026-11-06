"""Reusable support for the six teaching notebooks."""
from threadpoolctl import threadpool_limits
from .config import ValidationError
from .data import load_tutorial_context
from .retrieval import retrieve, rerank
from .perturbations import make_query_typos
from .metrics import evaluate_suite
from .interventions import fit_repair_adapter, select_adapter
from .reporting import build_robustness_report

# Apply the limit AFTER NumPy/SciPy load their BLAS libraries. Applying it before
# those imports silently leaves fresh processes at the machine-wide default.
_cpu_limit = threadpool_limits(limits=2)
