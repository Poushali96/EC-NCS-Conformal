"""Figures and numbers for the new CICV5G analyses (reserved runs, RF).

Inputs (written by the rerun of the original pipeline, rerun_reserved.py):
  rerun/packets.pkl       per-packet test intervals of every method, per fold
  rerun/packets_cal.pkl   calibration residuals, scales and transmission frequency
Outputs:
  figures/cicv5g_delay_budget.pdf      delay-budget admission (Section VII-F)
  figures/cicv5g_run_coverage.pdf      per-run coverage and frequency calibration (VII-E)
  rerun/new_analysis_numbers.json      every number quoted in the paper
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_figures import C, COL_W, TEXT_W, TARGET_COLOR  # shared style  # noqa: E402

RERUN = Path("/home/claude/rerun")
OUT = Path(__file__).resolve().parent / "figures"
P = pickle.load(open(RERUN / "packets.pkl", "rb"))
PC = pickle.load(open(RERUN / "packets_cal.pkl", "rb"))
ALPHA = 0.1
METHODS = ["Standard CP", "OOF Normalized CP", "CQR", "Validation-gated EC-NCS"]
LABEL = {"Point prediction": "Point prediction", "Standard CP": "Standard CP",
         "OOF Normalized CP": "OOF normalized CP", "CQR": "CQR",
         "Validation-gated EC-NCS": "Validation-gated EC-NCS"}
COLOR = {"Point prediction": "#111111", "Standard CP": C["Standard CP"],
         "OOF Normalized CP": C["OOF normalized CP"], "CQR": C["CQR"],
         "Validation-gated EC-NCS": C["EC-NCS"]}
STYLE = {"Point prediction": (0, (2, 1.5)), "Standard CP": "-", "OOF Normalized CP": "-",
         "CQR": "-", "Validation-gated EC-NCS": "-"}
numbers = {}


def conformal_q(scores):
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------- pooled arrays
y = np.concatenate([P[f]["y"] for f in P])
run = np.concatenate([P[f]["run"] for f in P])
upper = {"Point prediction": np.concatenate([P[f]["pte"] for f in P])}
for m in METHODS:
    upper[m] = np.concatenate([P[f]["intervals"][m][1] for f in P])
budget90 = float(np.quantile(P[0]["ytrain"], 0.9))
numbers["budget_train_p90_ms"] = budget90
numbers["n_test_packets"] = int(len(y))
runs = np.unique(run)
run_index = {r: np.flatnonzero(run == r) for r in runs}


def admission(B, idx=None):
    yy = y if idx is None else y[idx]
    out = {}
    for m, u in upper.items():
        uu = u if idx is None else u[idx]
        acc = uu <= B
        viol = yy > B
        out[m] = (acc.mean(), (acc & viol).sum() / max(1, acc.sum()))
    return out


budgets = np.arange(20, 41, 1.0)
curves = {m: np.array([admission(B)[m] for B in budgets]) for m in upper}
# run bootstrap bands
rng = np.random.default_rng(2027)
boot = {m: [] for m in upper}
for _ in range(400):
    pick = rng.choice(runs, len(runs), replace=True)
    idx = np.concatenate([run_index[r] for r in pick])
    res = [admission(B, idx) for B in budgets]
    for m in upper:
        boot[m].append([r[m] for r in res])
for m in upper:
    boot[m] = np.array(boot[m])  # (B, budgets, 2)

at = admission(budget90)
numbers["budget_at_p90"] = {m: {"admitted": float(a), "violating_among_admitted": float(v)} for m, (a, v) in at.items()}
numbers["violation_rate_all_packets_at_p90"] = float((y > budget90).mean())
for m in upper:
    i90 = int(np.argmin(np.abs(budgets - budget90)))
    lo = np.quantile(boot[m][:, i90, :], 0.025, axis=0)
    hi = np.quantile(boot[m][:, i90, :], 0.975, axis=0)
    numbers["budget_at_p90"][m]["admitted_ci"] = [float(lo[0]), float(hi[0])]
    numbers["budget_at_p90"][m]["violating_ci"] = [float(lo[1]), float(hi[1])]

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), gridspec_kw={"wspace": 0.30})
ax = axes[0]
for m in ["Point prediction"] + METHODS:
    c = curves[m][:, 0]
    if m in ("Validation-gated EC-NCS", "Standard CP"):
        lo = np.quantile(boot[m][:, :, 0], 0.025, axis=0); hi = np.quantile(boot[m][:, :, 0], 0.975, axis=0)
        ax.fill_between(budgets, lo, hi, color=COLOR[m], alpha=0.10, lw=0)
    ax.plot(budgets, c, color=COLOR[m], ls=STYLE[m], lw=1.8 if "EC-NCS" in m else 1.2, label=LABEL[m], zorder=3)
ax.axvline(budget90, color="#999999", lw=0.8, ls=":", zorder=1)
ax.text(budget90 + 0.3, 0.03, f"{budget90:.0f} ms", color="#777777", fontsize=7)
ax.set_xlim(20, 40); ax.set_ylim(0, 1.0)
ax.set_xlabel("Delay budget (ms)"); ax.set_ylabel("Admitted packets")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.set_title("(a) Traffic admitted at each budget", loc="left", fontweight="bold")

ax = axes[1]
sel = (budgets >= 25) & (budgets <= 40)
for m in ["Point prediction"] + METHODS:
    a_, v_ = curves[m][sel, 0], curves[m][sel, 1]
    ax.plot(a_, v_, color=COLOR[m], ls=STYLE[m], lw=1.8 if "EC-NCS" in m else 1.2, zorder=3)
    a31, v31 = at[m]
    ax.plot(a31, v31, "o", ms=5.5, color=COLOR[m], mec="white", mew=0.7, zorder=4)
ax.annotate("better", xy=(0.97, 0.004), xytext=(0.70, 0.045), fontsize=7, color="#666666",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 0.09)
ax.set_xlabel("Admitted packets"); ax.set_ylabel("Admitted packets that exceed the budget")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.set_title("(b) Admitted traffic vs. violations (25–40 ms)", loc="left", fontweight="bold")
ax.text(0.31, 0.084, f"dots: {budget90:.0f} ms budget", fontsize=7, color="#666666", va="top")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=2.0)
fig.savefig(OUT / "cicv5g_delay_budget.pdf")
plt.close(fig)

# ---------------------------------------------------------------- frequency-conditional calibration
def mondrian(method, f):
    d, c = P[f], PC[f]
    pte = d["pte"]
    ft, fcal = c["freq_test"], c["freq_cal"]
    if method == "Standard CP":
        sc_cal, lo_fun = np.abs(c["ycal"] - c["pca"]), lambda q, k: (pte[k] - q, pte[k] + q)
    elif method == "OOF Normalized CP":
        sc_cal = np.abs(c["ycal"] - c["pca"]) / c["sca"]
        lo_fun = lambda q, k: (pte[k] - q * c["ste"][k], pte[k] + q * c["ste"][k])
    elif method == "Validation-gated EC-NCS":
        sc_cal = np.abs(c["ycal"] - c["pca"]) / c["sec"]
        lo_fun = lambda q, k: (pte[k] - q * c["setest"][k], pte[k] + q * c["setest"][k])
    else:  # CQR
        sc_cal = np.maximum(c["lca"] - c["ycal"], c["ycal"] - c["uca"])
        lo_fun = lambda q, k: (c["lte"][k] - q, c["ute"][k] + q)
    lo = np.empty(len(pte)); hi = np.empty(len(pte)); nruns = {}
    for fr in np.unique(ft):
        k = ft == fr
        cal_k = fcal == fr
        nruns[int(fr)] = int(len(np.unique(c["run_cal"][cal_k])))
        q = conformal_q(sc_cal[cal_k]) if cal_k.sum() >= 30 else conformal_q(sc_cal)
        lo[k], hi[k] = lo_fun(q, k)
    return lo, hi, nruns


rows = []
cal_runs_per_freq = {}
for f in P:
    d = P[f]
    freq = PC[f]["freq_test"]
    for m in METHODS:
        lo0, hi0 = d["intervals"][m]
        lo1, hi1, nr = mondrian(m, f)
        cal_runs_per_freq[f] = nr
        for r in np.unique(d["run"]):
            k = d["run"] == r
            for name, lo, hi in [("Pooled calibration", lo0, hi0), ("Calibration per frequency", lo1, hi1)]:
                yy = d["y"][k]
                cov = float(((yy >= lo[k]) & (yy <= hi[k])).mean())
                w = hi[k] - lo[k]
                wink = float(np.mean(w + (2 / ALPHA) * (lo[k] - yy) * (yy < lo[k]) + (2 / ALPHA) * (yy - hi[k]) * (yy > hi[k])))
                rows.append({"Fold": f, "Run": r, "FrequencyHz": int(freq[k][0]), "Method": m, "Calibration": name,
                             "Coverage": cov, "Width": float(w.mean()), "Winkler": wink})
RC = pd.DataFrame(rows)
RC.to_csv(RERUN / "run_coverage_by_calibration.csv", index=False)
numbers["cal_runs_per_frequency_per_fold"] = {str(k): v for k, v in cal_runs_per_freq.items()}
summ = RC.groupby(["Calibration", "Method"])[["Coverage", "Width", "Winkler"]].mean()
numbers["run_mean_by_calibration"] = {f"{a}|{b}": v for (a, b), v in summ.round(4).to_dict("index").items()}
byf = RC.groupby(["Calibration", "Method", "FrequencyHz"]).Coverage.mean()
numbers["coverage_by_frequency"] = {f"{a}|{b}|{c}": float(v) for (a, b, c), v in byf.round(4).items()}
numbers["runs_below_0.8"] = {f"{a}|{b}": int(v) for (a, b), v in RC.groupby(["Calibration", "Method"]).Coverage.apply(lambda s: (s < 0.8).sum()).items()}

# paired test, frequency calibration vs pooled, per method (run level)
from scipy.stats import wilcoxon
tests = {}
for m in METHODS:
    a = RC[(RC.Method == m) & (RC.Calibration == "Pooled calibration")].set_index("Run")
    b = RC[(RC.Method == m) & (RC.Calibration == "Calibration per frequency")].set_index("Run").loc[a.index]
    ce_a = (a.Coverage - 0.9).abs(); ce_b = (b.Coverage - 0.9).abs()
    tests[m] = {"coverage_error_pooled": float(ce_a.mean()), "coverage_error_freq": float(ce_b.mean()),
                "wilcoxon_p_ce": float(wilcoxon(ce_b - ce_a).pvalue),
                "winkler_pooled": float(a.Winkler.mean()), "winkler_freq": float(b.Winkler.mean()),
                "wilcoxon_p_winkler": float(wilcoxon(b.Winkler - a.Winkler).pvalue)}
numbers["frequency_calibration_tests"] = tests

# does the calibration set contain a run at the test run's transmission frequency?
inc = []
for f in P:
    calf = set(np.unique(PC[f]["freq_cal"]).tolist())
    for r in np.unique(P[f]["run"]):
        k = P[f]["run"] == r
        inc.append({"Run": r, "FrequencyHz": int(PC[f]["freq_test"][k][0]), "CalHasFreq": int(PC[f]["freq_test"][k][0]) in calf})
INC = pd.DataFrame(inc)
RP = RC[RC.Calibration == "Pooled calibration"].merge(INC[["Run", "CalHasFreq"]], on="Run")
numbers["coverage_by_cal_has_freq"] = {f"{m}|{h}": {"mean": float(g.Coverage.mean()), "runs": int(len(g))}
                                       for (m, h), g in RP.groupby(["Method", "CalHasFreq"])}
numbers["coverage_100hz_by_cal_has_freq"] = {f"{m}|{h}": {"mean": float(g.Coverage.mean()), "runs": int(len(g))}
                                             for (m, h), g in RP[RP.FrequencyHz == 100].groupby(["Method", "CalHasFreq"])}
numbers["coverage_by_frequency_pooled"] = {f"{m}|{fr}": float(g.Coverage.mean()) for (m, fr), g in RP.groupby(["Method", "FrequencyHz"])}

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), gridspec_kw={"wspace": 0.62, "width_ratios": [1.15, 1]})
freq_order = [100, 33, 10]
freq_label = {100: "100 Hz", 33: "33 Hz", 10: "10 Hz"}
offs = {"Standard CP": -0.27, "OOF Normalized CP": -0.09, "CQR": 0.09, "Validation-gated EC-NCS": 0.27}
ax = axes[0]
for i, fr in enumerate(freq_order):
    ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5" if i % 2 == 0 else "white", zorder=0, lw=0)
    for m in METHODS:
        v = RP[(RP.FrequencyHz == fr) & (RP.Method == m)].Coverage.to_numpy()
        yy = np.full(len(v), i + offs[m]) + np.linspace(-0.035, 0.035, len(v))
        ax.scatter(v, yy, s=12, color=COLOR[m], alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([v.mean()] * 2, [i + offs[m] - 0.08, i + offs[m] + 0.08], color=COLOR[m], lw=2.4, zorder=4)
ax.axvline(0.9, color=TARGET_COLOR, lw=1.0, ls=(0, (4, 2)), zorder=2)
ax.set_xlim(0.3, 1.02)
ax.set_yticks(range(3), [freq_label[f] for f in freq_order])
ax.set_ylim(2.55, -0.55)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Coverage of each test run (dots) and mean (bars)")
ax.set_ylabel("Transmission frequency")
ax.set_title("(a) Coverage by transmission frequency", loc="left", fontweight="bold")
ax.text(0.905, -0.5, "target 0.90", color=TARGET_COLOR, fontsize=7, va="top")

ax = axes[1]
groups = [(True, "Same frequency\nin calibration"), (False, "Frequency missing\nfrom calibration")]
for i, (flag, lab) in enumerate(groups):
    for m in METHODS:
        v = RP[(RP.CalHasFreq == flag) & (RP.Method == m)].Coverage.to_numpy()
        yy = np.full(len(v), i + offs[m]) + np.linspace(-0.035, 0.035, len(v))
        ax.scatter(v, yy, s=12, color=COLOR[m], alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([v.mean()] * 2, [i + offs[m] - 0.08, i + offs[m] + 0.08], color=COLOR[m], lw=2.4, zorder=4)
ax.axvline(0.9, color=TARGET_COLOR, lw=1.0, ls=(0, (4, 2)), zorder=2)
ax.set_xlim(0.3, 1.02)
ax.set_yticks(range(2), [g[1] for g in groups])
ax.set_ylim(1.55, -0.55)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Coverage of each test run (dots) and mean (bars)")
ax.set_title("(b) Coverage by calibration content", loc="left", fontweight="bold")
handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[m], label=LABEL[m], ms=4) for m in METHODS]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.09))
fig.savefig(OUT / "cicv5g_run_coverage.pdf")
plt.close(fig)

(RERUN / "new_analysis_numbers.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1)[:6000])
