# Code and results for the IEEE TMLCN revision

This folder holds the code and results for every analysis added in the revised manuscript
*Explanation-Conditioned Conformal Prediction for Communication-Network Delay: Validated Gains and Failure Boundaries*.
The original experiments (notebook `EC_NCS_EXTRA_pynb.ipynb`, `conformal.py`) are in the repository root.

## Data

| Dataset | Source | Used for |
|---|---|---|
| CICV5G v1.0.0 | https://github.com/zxr805/CICV5G (commit `ae05da3`) | Main reserved evaluation, admission analyses |
| NordicDat | https://doi.org/10.5281/zenodo.10964584 (`nordicdat.csv`) | External replication |

The data are not redistributed here. Download them from the sources above.

## Folder layout

| Folder | Contents |
|---|---|
| `pipeline/` | `cicv5g_pipeline.py` (the original CICV5G pipeline, unchanged), `rerun_reserved.py` (re-runs the 20 reserved runs and stores intervals), `rerun_extended.py` (also stores attribution vectors and signed out-of-fold residuals), `rerun_predgate.py` (same gate with predicted-delay rank), `time_components.py` (cost table), `cicv5g_pipeline_nogate.py` and `nogate_cic.py` (gate disabled, for the gate analysis) |
| `analyses/` | Admission and decision analyses (`paired_decisions.py`, `paper_frontier.py`, `calsize.py`, `calsize_perfreq.py`), few-drive and missing-regime tests (`fewdrives.py`, `case2.py`, `case3.py`, `reviewcheck.py`), signal ablations (`addvalue.py`, `addvalue_grouped.py`), and exploratory scripts (`onesided.py`, `evaluate.py`, `robust.py`, `combo.py`, `deploy.py`, `deploy_worst.py`, `final_frontier.py`) |
| `nordicdat/` | `PROTOCOL.md` (prespecified protocol and log), `run_nordic.py`, `run_nordic_v2.py` (replication), `fewsess.py` (few-session test), `gate_cond.py`, `nogate_nordic.py` (gate and condition analyses), per-session results |
| `treeshap/` | `treeshap_wrap.cc` and `treeshap.py`: a ctypes wrapper around the C++ TreeSHAP code of the `shap` package, used where `shap` could not be installed. Attributions match the RF setting of the paper (`approximate=True`, Saabas). |
| `stubs/` | Minimal stand-ins for `shap`, `torch`, and `captum` so the pipeline runs for the random forest without a GPU stack. Not needed if the real packages are installed. |
| `figures/` | `build_figures.py`, `build_new_figures.py`: redraw all figures of the revised paper from the result files |
| `results/` | All result files behind the numbers in the paper (CSV/JSON/logs) |

## Paths

The scripts were run in one working directory and use absolute paths. Before running, replace these prefixes with your own locations:

| Prefix in the scripts | Meaning |
|---|---|
| `/content/cicv5g_data` | CICV5G repository clone (the path the original Colab notebook uses) |
| `/home/claude/nb` | this `pipeline/` folder |
| `/home/claude/rerun` | output folder for re-run results (`packets.pkl`, `extended.pkl`, CSV files) |
| `/home/claude/tools` | `treeshap/` and `stubs/` |
| `/home/claude/nordic` | `nordicdat/` folder and its outputs |
| `/home/claude/work` | the LaTeX project (figures are written to `figures/` there) |
| `/mnt/user-data/uploads/...nordicdat.csv` | the downloaded NordicDat CSV |

## Order of execution

1. `pipeline/rerun_reserved.py`, then `pipeline/rerun_extended.py` (needs CICV5G; writes `packets.pkl`, `packets_cal.pkl`, `extended.pkl`).
2. `pipeline/time_components.py`, `pipeline/rerun_predgate.py`, `pipeline/nogate_cic.py`.
3. Scripts in `analyses/` (they read the files from step 1).
4. `nordicdat/run_nordic.py` (or `run_nordic_v2.py`, which also stores packet-level intervals), `nordicdat/fewsess.py`, `nordicdat/nogate_nordic.py`, `nordicdat/gate_cond.py`.
5. `figures/build_figures.py`, `figures/build_new_figures.py`.

## Map to the paper

| Paper item | Script(s) | Result file(s) |
|---|---|---|
| Reserved CICV5G evaluation (13.2%) | `pipeline/rerun_reserved.py` | `results/rerun_run_metrics.csv` |
| Coverage gap by frequency and calibration | `figures/build_new_figures.py` | `results/run_coverage_by_calibration.csv` |
| Delay-budget admission, paired decisions | `analyses/paired_decisions.py` | `results/paired_decisions.json`, `results/new_analysis_numbers.json` |
| Admission trade-off and calibration-chosen levels | `analyses/paper_frontier.py` | `results/paper_frontier.json` |
| Transfer with 1 to 6 calibration runs | `analyses/calsize.py`, `analyses/calsize_perfreq.py` | printed tables |
| Which signal drives the gain (gated predicted-delay scale) | `pipeline/rerun_predgate.py` | `results/predgate.csv` |
| Attributions beyond predicted delay and frequency | `analyses/addvalue.py`, `analyses/addvalue_grouped.py` | `results/addvalue*.csv` |
| Few drives and no 100 Hz drives (Table of reduced data) | `analyses/fewdrives.py`, `analyses/case3.py`, `analyses/reviewcheck.py` | `results/fewdrives.csv`, `results/case3.csv`, `results/reviewcheck.csv` |
| Cost of each method | `pipeline/time_components.py` | `results/timing_components.json` |
| NordicDat replication (12.6%) | `nordicdat/run_nordic.py` | `nordicdat/results.csv` |
| NordicDat few-session test | `nordicdat/fewsess.py` | `nordicdat/fewsess.csv` |
| Gate analysis and conditions | `nordicdat/nogate_nordic.py`, `pipeline/nogate_cic.py`, `nordicdat/gate_cond.py` | printed tables |

The exploratory scripts in `analyses/` (`onesided.py`, `evaluate.py`, `robust.py`, `combo.py`, `deploy.py`, `deploy_worst.py`, `final_frontier.py`, `case2.py`) are included for completeness. They contain analyses that were tried and not reported as findings, as described in the revision notes.

## Requirements

Python 3.11+, numpy, pandas, scipy, scikit-learn, matplotlib. For explanations either the `shap` package or the `treeshap/` wrapper (build with
`g++ -O3 -fPIC -shared -std=c++17 -I<shap>/shap/cext -I<python-include> treeshap_wrap.cc -o libtreeshap.so`). The MLP experiments of the original notebook need PyTorch and Captum; the revision analyses use the random forest only.
