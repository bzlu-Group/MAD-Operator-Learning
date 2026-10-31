"""Typography matched to the manuscript's existing Arial figures."""

from contextlib import contextmanager


@contextmanager
def figure_style(kind="prediction"):
    import matplotlib.pyplot as plt

    size = 8 if kind == "loss" else 10
    settings = {
        "font.family": "Arial", "font.size": size, "font.weight": "normal",
        "axes.titlesize": 9 if kind == "loss" else 10,
        "axes.titleweight": "bold", "axes.labelsize": 9 if kind == "loss" else 10,
        "axes.labelweight": "normal", "xtick.labelsize": size, "ytick.labelsize": size,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.bfit": "Arial:bold:italic", "mathtext.sf": "Arial",
        "mathtext.tt": "Arial", "mathtext.cal": "Arial",
        "mathtext.default": "regular",
    }
    with plt.rc_context(settings):
        yield
