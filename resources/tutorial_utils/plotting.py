"""Readable, colour-accessible graphs; all measurements come from supplied runs."""
import base64
import io
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Pick up installed Arial files even when Matplotlib's font cache predates them.
# Skip the system scan when the cache already lists Arial: on macOS the scan runs
# system_profiler, which can take more than ten seconds.
if not any(font.name == "Arial" for font in font_manager.fontManager.ttflist):
    _known_fonts = {font.fname for font in font_manager.fontManager.ttflist}
    for _font_path in font_manager.findSystemFonts():
        if "arial" in Path(_font_path).stem.lower() and _font_path not in _known_fonts:
            font_manager.fontManager.addfont(_font_path)

COLORS={"clean":"#0072B2","shifted":"#D55E00","repaired":"#009E73","ink":"#25364A","muted":"#6E7781",
        "axis":"#9AA3AD","grid":"#E4E7EB"}
PANEL=5.6  # inches per square plot panel, including its labels
SCALE=3    # PNG pixels per screen pixel: sharp on high-density screens and when zoomed in
plt.rcParams.update({"figure.figsize":(PANEL,PANEL),"figure.dpi":110,"savefig.bbox":"tight",
                     "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
                     "font.family":"Arial","mathtext.fontset":"custom",
                     "mathtext.rm":"Arial","mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold",
                     "font.size":12,"axes.titlesize":14,"axes.titleweight":"bold","axes.labelsize":12,
                     "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11,"legend.frameon":False,
                     "text.color":COLORS["ink"],"axes.labelcolor":COLORS["ink"],"axes.edgecolor":COLORS["axis"],
                     "axes.linewidth":.8,"xtick.color":COLORS["axis"],"ytick.color":COLORS["axis"],
                     "xtick.labelcolor":COLORS["ink"],"ytick.labelcolor":COLORS["ink"],
                     "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":2,"lines.markersize":7})

def show(fig, description=None):
    """Display a high-resolution PNG at the figure's usual on-screen size."""
    from IPython.display import display
    captions=[t.get_text() for t in fig.texts if t.get_text()]
    for ax in fig.axes:
        captions += [ax.get_title(loc) for loc in ("left","center","right") if ax.get_title(loc)]
    description=description or "; ".join(captions) or "Computed tutorial graph; read the surrounding explanation and result table."
    buffer=io.BytesIO()
    fig.savefig(buffer,format="png",dpi=fig.dpi*SCALE)
    png=buffer.getvalue()
    width,height=struct.unpack(">II",png[16:24])  # pixel size from the PNG header
    display({"image/png":base64.b64encode(png).decode("ascii"),"text/plain":repr(fig)},raw=True,
            metadata={"image/png":{"width":round(width/SCALE),"height":round(height/SCALE),"alt":description},
                      "tutorial_alt":description})
    plt.close(fig)

def square_axes(ncols=1):
    """A row of square plot areas (equal x- and y-axis lengths) with a light value grid."""
    fig,axes=plt.subplots(1,ncols,figsize=(PANEL*ncols,PANEL),layout="constrained",squeeze=False)
    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.grid(axis="y",color=COLORS["grid"],linewidth=.8); ax.set_axisbelow(True)
        ax.tick_params(length=3)
    return (fig,axes[0,0]) if ncols==1 else (fig,list(axes[0]))

def titles(ax, title):
    """Left-aligned title above the plot area; details belong in the notebook text, not a subtitle."""
    ax.set_title(title,loc="left",pad=12)

def legend_below(fig, ax, ncols=None):
    """One legend under the plot, so it never covers the data."""
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=ncols or len(labels),
               handlelength=1.6,columnspacing=1.6)

def report_figure(report):
    from .reporting import LABELS
    s = report['evaluation']['mean_and_tail']
    base = report['evaluation']['baseline_comparison']['baseline_mean_and_tail']
    series = [('Original search', [base['clean_ndcg'], base['shifted_ndcg']], COLORS['clean'])]
    if report['intervention']['name'] != 'none':
        series.append((LABELS[report['intervention']['name']], [s['clean_ndcg'], s['shifted_ndcg']],
                       COLORS['repaired']))
    fig, ax = square_axes()
    x = np.array([0, 1]); width = .3 if len(series) > 1 else .42
    # A small surface gap separates the paired bars.
    offsets = (np.arange(len(series)) - (len(series)-1)/2) * (width+.04)
    for offset, (label, values, color) in zip(offsets, series):
        bars = ax.bar(x+offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11)
    ax.set(xticks=x, xticklabels=['Original inputs', 'After typos'], xlim=(-.6, 1.6),
           ylabel='nDCG@10 (higher is better)', ylim=(0, 1.05), yticks=np.linspace(0, 1, 6))
    titles(ax, 'Search quality before and after typos')
    if len(series) > 1:
        legend_below(fig, ax)
    return fig

# A compact overview: one label per box, with generation shown as a described step.
def pipeline_figure():
    """Retrieval and reference selection executed in M1–M6; generation is described."""
    fig, ax = plt.subplots(figsize=(11, 2.1))
    ax.set(xlim=(-.1, 10.9), ylim=(0, 2.1))
    ax.axis("off")
    width, height, y = 1.8, .6, .45
    xs = [1 + 2.2*i for i in range(5)]

    def box(x, y, label, dashed=False, data=False):
        ax.add_patch(FancyBboxPatch(
            (x-width/2, y-height/2), width, height,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor="#F1F3F5" if data else ("white" if dashed else "#EAF2F9"),
            edgecolor=COLORS["muted"] if data else COLORS["clean"],
            linewidth=1.5, linestyle=(0, (4, 2.5)) if dashed else "-"))
        ax.text(x, y, label, ha="center", va="center", fontsize=12)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=13, color=COLORS["ink"], linewidth=1.5))

    for x, label in zip(xs, ["Query", "Retrieval", "Reranking", "Top-K references", "LLM"]):
        box(x, y, label, dashed=label == "LLM")
    for left, right in zip(xs, xs[1:]):
        arrow((left+width/2, y), (right-width/2, y))
    box(xs[1], 1.6, "Corpus", data=True)
    arrow((xs[1], 1.6-height/2), (xs[1], y+height/2))
    fig.tight_layout(pad=.3)
    return fig
