<img src="resources/assets/cikm2026-logo.png" alt="CIKM 2026 logo" width="320">

# Beyond Benchmarks: Robustness for Real-World Information Systems

Bum Jun Kim, The University of Tokyo, CIKM 2026

7 November 2026 · San Pietro in Vincoli Campus, Sapienza University of Rome · Rome, Italy

Six teaching notebooks for Bum Jun Kim’s CIKM 2026 tutorial. The notebooks contain the lecture, equations, graphs, demonstrations, comparisons, and summaries.

| Notebook | Main focus |
|---|---|
| [M1: Observe](M1_Observe_Failures.ipynb) | Retrieval failures and the candidate boundary |
| [M2: Diagnose](M2_Diagnose_Shifts.ipynb) | Query Embedding Drift and Document Insertion |
| [M3: Measure](M3_Evaluate_Robustness.ipynb) | Typo Quality Comparison, Individual Query Outcomes, and Typo Severity Sweep |
| [M4: Intervene](M4_Improve_Robustness.ipynb) | Normalization, spelling correction, adapter example, search quality, and one failure case per method |
| [M5: Protect RAG references from document attacks](M5_Transfer_to_RAG.ipynb) | The top K reranked papers as the LLM's references; counting an attack document at K=1/5/10; an ideal filter before selecting K |
| [M6: Report and check updates](M6_Report_and_Check_Updates.ipynb) | Evaluate a repair, decide Go or Hold, repeat a fixed quality check, and verify recovery after rollback |

## Folder layout

The six notebooks and this README sit at the top of the tutorial folder. Everything the notebooks use lives in `resources/`; keep it next to the notebooks.

| Path | Contents |
|---|---|
| `resources/tutorial_utils/` | Helper package imported by the notebooks |
| `resources/data/` | SciFact source files, provenance, and license |
| `resources/assets/caches/` | Verified model outputs read by the notebooks |
| `resources/scripts/prepare_assets.py` | How the caches were generated; verifies or rebuilds them |
| `resources/requirements.txt` | Python dependencies (`requirements-live.txt` adds optional live inference) |
| `resources/outputs/` | Reports written by M6 (created when needed) |

## Set up a conda environment

Use Python 3.11, the tested version. Open a terminal in the tutorial folder (the one that contains the six notebooks and `resources/`) and run:

```bash
conda create -n cikm-tutorial python=3.11 pip -y
conda activate cikm-tutorial
python -m pip install -r resources/requirements.txt
```

`cikm-tutorial` is only an example name. The last command installs the pinned packages (NumPy, pandas, Matplotlib, JupyterLab, and a few helpers) from PyPI. After that, the lessons need no GPU, API key, model download, or internet connection.

If `conda create` stops and asks you to accept the Terms of Service of Anaconda's default channels, either accept them with `conda tos accept` or create the environment from the community conda-forge channel instead:

```bash
conda create -n cikm-tutorial -c conda-forge --override-channels python=3.11 pip -y
```

Without conda, a virtual environment works the same way: `python3.11 -m venv .venv`, then activate it (`source .venv/bin/activate` on macOS and Linux, `.venv\Scripts\activate` on Windows) and run the same `pip install` command. An environment folder is tied to the machine that created it; on another machine, create a new one instead of copying it.

## Run the notebooks in JupyterLab

From the tutorial folder:

```bash
conda activate cikm-tutorial
jupyter lab
```

JupyterLab opens in the browser. Because it was started inside the environment, the notebooks' standard Python 3 kernel runs that environment's Python; no kernel registration is needed.

1. Open `M1_Observe_Failures.ipynb` from the file browser on the left.
2. Choose Kernel → Restart Kernel and Run All Cells…. The first code cell checks the supplied data and cached results. Each notebook runs on its own; no earlier notebook has to run first.
3. To try another setting, edit one or two values in the Experiment Settings cell, keep that cell selected, and choose Run → Run Selected Cell and All Below.

If you open the notebooks from a Jupyter installation outside this environment (another environment, or an editor such as VS Code), register the environment once and pick it with Kernel → Change Kernel…:

```bash
conda activate cikm-tutorial
python -m ipykernel install --user --name cikm-tutorial --display-name "Python 3 (CIKM tutorial)"
```

## Run the notebooks in Google Colab

Colab is optional. It opens one notebook at a time, while every notebook needs the `resources/` folder next to it, so each Colab session first copies the tutorial repository. After the repository is public:

1. Open a notebook from GitHub with `https://colab.research.google.com/github/<OWNER>/<REPO>/blob/<BRANCH>/M1_Observe_Failures.ipynb`. Replace `<OWNER>`, `<REPO>`, and `<BRANCH>` with the public repository; if the notebooks are in a subfolder, put its path before the file name.
2. Insert a code cell at the top (Insert → Code cell) and paste the following, with the same replacements. If the notebooks are in a subfolder, add it to the `%cd` path.

   ```python
   import os
   if not os.path.isdir("/content/<REPO>"):
       !git clone --depth 1 https://github.com/<OWNER>/<REPO>.git /content/<REPO>
   %cd /content/<REPO>
   ```

3. Choose Runtime → Run all.
4. To try another setting, edit the Experiment Settings cell, keep it selected, and choose Runtime → Run cell and below.

Do not install `resources/requirements.txt` in Colab. Its pins are for Python 3.11, while Colab runs a newer Python that already provides NumPy, SciPy, pandas, Matplotlib, Beautiful Soup, and threadpoolctl; the pinned Jupyter packages would also replace Colab's own. After Runtime → Restart session, run the cell from step 2 again before the other cells. M6's reports stay in the Colab session; use the download links in M6 to keep them.

## Data, models, and experiment contract

The supplied SciFact files contain 5,183 documents and 300 official test queries, counted from the files. The official train set has 809 judged query IDs, split deterministically into 647 train / 162 validation by a seeded hash. Original string IDs and source files are retained. All relevance evaluation uses the full corpus; selected demonstration tables explicitly say n=1 and exploratory.

- Corpus: title + newline + abstract. Original source: [BEIR SciFact download instructions](https://github.com/beir-cellar/beir/wiki/Datasets-available).
- Dense model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Reranker: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Normalized query and document vectors; exact search over the full corpus. Reranking uses the top 20 dense candidates. Detailed preprocessing and numerical settings are recorded in the manifest below.
- Test typo severities 0–3 use seed 42 and per-query deterministic edit logs. Adapter corruption seeds 42/43/44 vary training noise; M4 validation/test corruption instances remain fixed at seed 42. No encoder fine-tuning. Lambda is chosen with validation qrels; test qrels never select an intervention.
- M3 asks three questions: how much nDCG@10 drops after typos, whether a small average loss hides queries that lose all relevant papers from the top 10, and how much further nDCG@10 falls with more character edits per query. The lesson fixes nDCG@10, varies only `severity`, and ends with Lesson Summary and sources.
- M4’s intervention comparison is dense-only for every method. Its table does not conflate the reranker’s candidate ceiling with repair effectiveness. M5 uses real cached cross-encoder scores on each condition’s 20 candidates.

The [manifest](resources/assets/caches/manifest.json) records source checksums, exact splits, model revisions, preprocessing, configuration/code fingerprints, generation time, device, environment, and perturbation coverage. All result tables derive from actual model outputs or labeled toy computations. No cached-read time is represented as inference latency. Model weights are not bundled; they are unnecessary for cached Run All.

## Licenses

- SciFact / BEIR data and derived caches: retain attribution to Wadden et al. and the BEIR authors. Claims and evidence annotations are CC BY 4.0; the abstract corpus is ODC-By 1.0 (from S2ORC). See [data provenance](resources/data/README.md) and the upstream [license notice](resources/data/LICENSE-SciFact.md). Source files are unchanged; embeddings, typos, rankings, and synthetic passages in the caches are derived tutorial artifacts.
- Models: the model cards identify Apache-2.0 for both pinned models. The tutorial distributes computed embeddings and scores, not model weights; optional live mode downloads the weights from the original source.

## Optional live inference and cache regeneration

`MODE="cached"` reuses saved query vectors and reranker results; `MODE="live"` recomputes them with the models, while both reuse saved document vectors.

```bash
python -m pip install -r resources/requirements-live.txt
# Verify the supplied cache:
python resources/scripts/prepare_assets.py --dataset scifact --model sentence-transformers/all-MiniLM-L6-v2 --seed 42
# Explicitly regenerate genuine CPU model outputs:
python resources/scripts/prepare_assets.py --dataset scifact --model sentence-transformers/all-MiniLM-L6-v2 --seed 42 --rebuild
```

Cache verification needs only `requirements.txt`. Live mode downloads the models from Hugging Face on first use. For cached runs, use the query IDs and severity levels listed in the notebooks.

Live initialization and cache rebuilding can take time on CPU. Cached mode does not require PyTorch or a model runtime. If cached assets are missing or corrupted, restore the supplied files or regenerate them with `--rebuild`.

## Outputs and interpretation

M6 writes `resources/outputs/robustness_report.json` and `.md`, with on-screen download links. If writing is denied, it still displays downloadable text and completes. Reports identify the cached reference or configured run; no previous module output is required. `resources/outputs/` holds generated reports, while original data and precomputed caches are treated as read-only by the lessons.

FAQ: if a notebook cannot find the package, start Jupyter from the tutorial folder (the one containing the six notebooks and `resources/`), or in Colab rerun the copy cell; if the wrong Python starts, select or register the environment's kernel; if a config is unsupported, restore declared settings or use optional live APIs; if internet fails, stay in cached mode.
