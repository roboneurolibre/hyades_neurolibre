"""Output helpers for the analysis notebooks: static figures and results tables.

One source serves both outputs: the website uses the interactive figures and HTML tables, while the
Word/PDF/LaTeX exports use the static figure files (MyST `:placeholder:` with a `.*` extension, so each
export picks its format) and the Markdown tables.

Static figures are only written when the environment variable STATIC_FIGURE_FORMATS lists formats, e.g.
`png` (Word), `pdf` (LaTeX/journal) or `svg` (editable text, for Inkscape). Use the pixi task
`pixi run static-figures [png|pdf|svg]` (default png; a comma list such as png,svg writes several).
"""
import os

import matplotlib
import matplotlib.text
from matplotlib import font_manager

# Data folder (paths relative to content/, where the notebooks run). NeuroLibre and repo2data put the Zenodo record in
# data/<projectName>/ (binder/data_requirement.json); a plain download of the record into data/ works as well.
DATA_PROJECT_NAME = "hyades-tabular"
_DATA_ROOT = os.path.join("..", "data")
DATA_DIR = (os.path.join(_DATA_ROOT, DATA_PROJECT_NAME)
            if os.path.isdir(os.path.join(_DATA_ROOT, DATA_PROJECT_NAME)) else _DATA_ROOT)


def data_path(filename):
    """Path of a data file: data/hyades-tabular/<filename> if that folder exists, else data/<filename>."""
    return os.path.join(DATA_DIR, filename)


STATIC_FIGURE_DIR = os.path.join("..", "static", "figures")
STATIC_FIGURE_FORMATS = [f.strip().lower() for f in os.environ.get("STATIC_FIGURE_FORMATS", "").split(",") if f.strip()]
# Imaging Neuroscience asks for Arial or Helvetica in figures; Liberation Sans is metrically identical to Arial (Linux).
# Keep only installed fonts: matplotlib warns about every missing family in the list.
_PREFERRED_FONTS = ["Arial", "Helvetica", "Liberation Sans"]
_INSTALLED_FONTS = {f.name for f in font_manager.fontManager.ttflist}
FIGURE_FONTS = [f for f in _PREFERRED_FONTS if f in _INSTALLED_FONTS][:1] or ["DejaVu Sans"]

# Static figures are drawn at their printed size (journal text width 17.8 cm), so these font sizes are the printed
# sizes and are the same in every figure (target >= 7 pt)
TEXT_WIDTH_IN = 17.8 / 2.54
STATIC_RC = {
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "figure.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.title_fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "patch.linewidth": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3, "boxplot.flierprops.markersize": 2,
}


def static_theme(style="ticks"):
    """Seaborn theme for the static figures: journal font (FIGURE_FONTS) at the printed sizes of STATIC_RC."""
    import seaborn as sns
    sns.set_theme(style=style, font=FIGURE_FONTS[0], rc=STATIC_RC)


def save_static_figure(fig, label, folder=STATIC_FIGURE_DIR, dpi=200):
    """Save a matplotlib figure as <folder>/<label>.<format> for each format in STATIC_FIGURE_FORMATS.

    PNG is raster (for Word); PDF is vector (for LaTeX and the journal); SVG keeps text as text (for Inkscape).
    Files are written without timestamps, so re-running the notebooks does not change unchanged figures.
    """
    if not STATIC_FIGURE_FORMATS:
        return
    os.makedirs(folder, exist_ok=True)
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(FIGURE_FONTS)
    for ext in STATIC_FIGURE_FORMATS:
        path = os.path.join(folder, f"{label}.{ext}")
        if ext == "png":
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        elif ext == "pdf":
            # TrueType (Type 42) fonts: Type 3 fonts are often rejected by journal production checks
            with matplotlib.rc_context({"pdf.fonttype": 42}):
                fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
        elif ext == "svg":
            with matplotlib.rc_context({"svg.fonttype": "none"}):
                fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
        else:
            raise ValueError(f"Unsupported static figure format: {ext}")


def display_table(df):
    """Display a results table as HTML (website) and Markdown (Word/PDF/LaTeX exports)."""
    from IPython.display import display
    display({"text/html": df.to_html(index=False),
             "text/markdown": df.to_markdown(index=False),  # requires the tabulate package
             "text/plain": df.to_string(index=False)}, raw=True)
