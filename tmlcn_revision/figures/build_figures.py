"""Redraw the main-paper figures at their printed size so all text is readable.

Every figure is drawn at the exact width it has in the PDF (column = 246 pt,
text width = 504 pt), so a 9 pt font in this script is 9 pt on paper.

Data sources
------------
* Fig. 3 (rare mechanism): evidence/controlled_confirmation/all_seed_metrics.csv
* Fig. 5 (CICV5G development, coverage and Winkler): evidence/overall_metrics.csv
* Fig. 6 (CICV5G reserved runs): evidence/reserved_frequency_confidence_intervals.csv
* Fig. 2, Fig. 4 and the worst-RSRP panel of Fig. 5 have no raw CSV in the
  bundle. Their values were read from the vector PDFs of the original figures
  (calibrated on the axis gridlines) and match Tables 2 and 4 and the text of
  Section VII-D. They are listed in the dictionaries below.

Run:  python build_figures.py   (writes PDFs into figures/)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"

PT = 1 / 72.27
COL_W = 246 * PT          # \columnwidth
TEXT_W = 504 * PT         # \textwidth
WIDE_W = 0.82 * TEXT_W    # figures included at .82\textwidth

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

# One color per method, the same in every figure. Method names are always
# written on the axis, so color is never the only cue.
C = {
    "Standard CP": "#5f5f5f",
    "EC-NCS": "#2a78d6",
    "OOF normalized CP": "#eb6834",
    "CQR": "#1baf7a",
    "Null-signal": "#eda100",
    "Other": "#a3a3a3",
    "Oracle": "#1baf7a",
}
TARGET_COLOR = "#c0392b"
TARGET_LW = 1.0


def t_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def dot_panel(ax, labels, mean, lo, hi, colors, xlabel, target=None, bold=()):
    y = np.arange(len(labels))[::-1]
    for yi, m, l, h, c in zip(y, mean, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=1.2, solid_capstyle="round", zorder=2)
        ax.plot(m, yi, "o", ms=4.5, color=c, mec="white", mew=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text() in bold:
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel(xlabel)
    if target is not None:
        ax.axvline(target, color=TARGET_COLOR, lw=TARGET_LW, ls=(0, (4, 2)), zorder=2.5)
    return ax


# ---------------------------------------------------------------- Fig. 2
def fig_controlled():
    # mean, CI low, CI high of empirical coverage per ground-truth tertile
    v = {
        "Standard CP": [(0.9791, 0.9745, 0.9844), (0.9077, 0.9024, 0.9131), (0.8107, 0.8005, 0.8194)],
        "Informative stratification": [(0.9109, 0.9051, 0.9167), (0.8994, 0.8952, 0.9035), (0.8904, 0.8850, 0.8954)],
        "Null-signal stratification": [(0.9802, 0.9757, 0.9853), (0.9097, 0.9040, 0.9144), (0.8130, 0.8024, 0.8215)],
    }
    col = {"Standard CP": C["Standard CP"], "Informative stratification": C["EC-NCS"],
           "Null-signal stratification": C["Null-signal"]}
    mk = {"Standard CP": "s", "Informative stratification": "o", "Null-signal stratification": "^"}
    off = {"Standard CP": -0.12, "Informative stratification": 0.0, "Null-signal stratification": 0.12}
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    x = np.arange(3)
    for k, rows in v.items():
        m = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows]); hi = np.array([r[2] for r in rows])
        ax.errorbar(x + off[k], m, yerr=[m - lo, hi - m], color=col[k], marker=mk[k], ms=4.5,
                    lw=1.3, elinewidth=1.0, capsize=0, mec="white", mew=0.5, label=k, zorder=3)
    ax.axhline(0.90, color=TARGET_COLOR, lw=TARGET_LW, ls=(0, (4, 2)), zorder=2.5)
    ax.text(2.42, 0.903, "target 0.90", color=TARGET_COLOR, fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x, ["Low", "Medium", "High"])
    ax.set_xlim(-0.35, 2.45)
    ax.set_xlabel("True prediction-difficulty tertile")
    ax.set_ylabel("Empirical coverage")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    fig.savefig(OUT / "controlled_mechanism.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3
def fig_rare():
    df = pd.read_csv(ROOT / "evidence/controlled_confirmation/all_seed_metrics.csv")
    names = {
        "Standard CP": "Standard CP",
        "EC-NCS candidate (ungated)": "EC-NCS (ungated)",
        "Validation-gated EC-NCS": "Validation-gated EC-NCS",
        "Oracle scale CP": "Oracle-scale CP",
        "CQR-GBM": "CQR-GBM",
        "CQR-GBM interactions": "CQR-GBM (interactions)",
        "CQR-GBM explanation features": "CQR-GBM (expl. features)",
        "CQR-RF": "CQR-RF",
        "CQR-ExtraTrees": "CQR-ExtraTrees",
        "CQR-Neural": "CQR-Neural",
    }

    def color(m):
        if m == "Standard CP":
            return C["Standard CP"]
        if "EC-NCS" in m:
            return C["EC-NCS"]
        if m.startswith("Oracle"):
            return C["Oracle"]
        return C["Other"]

    rows = [("sample_size", "(a) Main setting"),
            ("explanation_aligned_reuse", "(b) Explanation-aligned setting")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 4.1), sharey="row",
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for r, (exp, title) in enumerate(rows):
        d = df[df.Experiment == exp]
        order = [m for m in names if m in d.Method.unique()]
        labs = [names[m] for m in order]
        cols = [color(names[m]) for m in order]
        for c, (metric, xl, tgt) in enumerate([
                ("RareCoverage", "Coverage in the rare group (higher is better)", 0.90),
                ("RareWinkler", "Winkler loss in the rare group (lower is better)", None)]):
            st = [t_ci(d[d.Method == m][metric]) for m in order]
            dot_panel(axes[r, c], labs, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st],
                      cols, xl, target=tgt, bold=("Standard CP", "Validation-gated EC-NCS", "CQR-GBM (expl. features)"))
        axes[r, 0].set_xlim(0.2, 1.0)
        axes[r, 0].text(0.905, len(order) - 0.45, "target 0.90", color=TARGET_COLOR,
                        fontsize=7, ha="left", va="bottom")
        axes[r, 0].set_title(title, loc="left", fontweight="bold", x=-0.62)
    fig.savefig(OUT / "rare_mechanism_confirmation.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig_owd():
    # Read from the original figure; coverage matches Table 4. Width is the
    # mean interval width divided by the IQR of the target.
    cov = {"NTNU": [(0.8987, 0.8967, 0.9006), (0.8973, 0.8924, 0.9023), (0.8996, 0.8975, 0.9017)],
           "WUE": [(0.8987, 0.8941, 0.9032), (0.8985, 0.8938, 0.9028), (0.8997, 0.8956, 0.9052)]}
    wid = {"NTNU": [(1.7084, 1.7009, 1.7159), (1.6156, 1.5995, 1.6325), (1.7257, 1.7142, 1.7368)],
           "WUE": [(1.6896, 1.6820, 1.7001), (1.5892, 1.5766, 1.6024), (1.7118, 1.7004, 1.7262)]}
    methods = ["Standard CP", "OOF normalized CP", "Informative stratification"]
    site_style = {"NTNU": ("o", "#3a3a3a", -0.13), "WUE": ("s", "#9a9a9a", 0.13)}
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_W, 1.75), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(3)[::-1]
    for ax, data, xl, tgt in [(axes[0], cov, "Empirical coverage", 0.90),
                              (axes[1], wid, "Mean width / target IQR (lower is better)", None)]:
        for site, rows in data.items():
            mk, col, dy = site_style[site]
            m = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows]); hi = np.array([r[2] for r in rows])
            ax.errorbar(m, y + dy, xerr=[m - lo, hi - m], fmt=mk, color=col, ms=4.2, elinewidth=1.0,
                        capsize=0, mec="white", mew=0.5, label=site, zorder=3)
        ax.set_yticks(y, methods)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel(xl)
        if tgt:
            ax.axvline(tgt, color=TARGET_COLOR, lw=TARGET_LW, ls=(0, (4, 2)), zorder=2.5)
            ax.set_xlim(0.888, 0.910)
            ax.text(0.9003, 2.52, "target 0.90", color=TARGET_COLOR, fontsize=7, ha="left", va="bottom")
    axes[1].legend(title=None, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                   ncol=2, handletextpad=0.3, borderaxespad=0.1)
    fig.savefig(OUT / "owd_overall.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5
def fig_cicv5g_dev():
    d = pd.read_csv(ROOT / "evidence/overall_metrics.csv")
    d = d[(d.Model == "Random Forest") & (d.Protocol == "ordinary unseen runs")]
    order = ["Standard CP", "OOF Normalized CP", "CQR", "Validation-gated EC-NCS"]
    labs = ["Standard CP", "OOF normalized CP", "CQR", "Validation-gated EC-NCS"]
    cols = [C["Standard CP"], C["OOF normalized CP"], C["CQR"], C["EC-NCS"]]
    # Worst RSRP-tertile coverage error, read from the original figure
    # (means match Section VII-D: 0.237, 0.050, 0.057, 0.232).
    worst = {"Standard CP": (0.2367, 0.1394, 0.3448), "OOF Normalized CP": (0.0497, 0.0346, 0.0647),
             "CQR": (0.0571, 0.0353, 0.0842), "Validation-gated EC-NCS": (0.2320, 0.1384, 0.3368)}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    st = [t_ci(d[d.Method == m]["Coverage"]) for m in order]
    dot_panel(axes[0], labs, *zip(*st), cols, "Coverage on unseen runs", target=0.90,
              bold=("Standard CP", "Validation-gated EC-NCS"))
    st = [t_ci(d[d.Method == m]["Winkler"]) for m in order]
    dot_panel(axes[1], labs, *zip(*st), cols, "Winkler loss (lower is better)")
    w = [worst[m] for m in order]
    dot_panel(axes[2], labs, *zip(*w), cols, "Worst RSRP-tertile coverage error\n(lower is better)")
    axes[0].text(0.9015, 3.55, "target 0.90", color=TARGET_COLOR, fontsize=7, ha="left", va="bottom")
    fig.savefig(OUT / "cicv5g_development.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6
def fig_reserved():
    ci = pd.read_csv(ROOT / "evidence/reserved_frequency_confidence_intervals.csv")
    ci = ci[ci.Model == "Random Forest"]
    order = ["Standard CP", "OOF Normalized CP", "CQR", "Validation-gated EC-NCS"]
    labs = ["Standard CP", "OOF normalized CP", "CQR", "Validation-gated EC-NCS"]
    cols = [C["Standard CP"], C["OOF normalized CP"], C["CQR"], C["EC-NCS"]]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_W, 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, metric, xl, tgt in [(axes[0], "Coverage", "Empirical coverage", 0.90),
                                (axes[1], "Winkler", "Winkler loss (lower is better)", None)]:
        s = ci[ci.Metric == metric].set_index("Method").loc[order]
        dot_panel(ax, labs, s.Mean, s.CI95Lower, s.CI95Upper, cols, xl, target=tgt,
                  bold=("Standard CP", "Validation-gated EC-NCS"))
    axes[0].text(0.9015, 3.55, "target 0.90", color=TARGET_COLOR, fontsize=7, ha="left", va="bottom")
    fig.savefig(OUT / "cicv5g_reserved_confirmation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_controlled()
    fig_rare()
    fig_owd()
    fig_cicv5g_dev()
    fig_reserved()
    print("figures written to", OUT)
