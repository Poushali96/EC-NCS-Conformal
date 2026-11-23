"""Time each extra component of the four interval methods on reserved fold 1 (RF).

Repeats each measurement 3 times and reports the median. Uses the original
pipeline functions (executed from the original script, unchanged).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/claude/tools/stubs")
SRC = Path("/home/claude/nb/cicv5g_pipeline.py").read_text()
head = SRC.split("# %% Main ten-seed experiment")[0].replace(
    "CFG = Config()", 'CFG = Config(outdir="/home/claude/rerun/out_t", models=("Random Forest",))', 1)
g = {"__name__": "ecncs", "display": lambda *a, **k: None}
exec(compile(head, "head", "exec"), g)
np, pd, CFG = g["np"], g["pd"], g["CFG"]
fc = g["frequency_confirm"]
gt, _ = g["make_balanced_confirmation_folds"](fc, CFG.confirmation_folds, 2027)
fold, seed = 0, 2027
test_groups = set(gt.loc[gt.Fold == fold, "RunGroup"])
rem = fc[~fc.RunGroup.isin(test_groups)].copy()
tri, cai, _, _ = g["train_calibration_group_split"](rem, seed)
train, cal = rem.iloc[tri].copy(), rem.iloc[cai].copy()
test = fc[fc.RunGroup.isin(test_groups)].copy()
F = g["FEATURES"]
prep = g["build_preprocessor"](train[F]); Xtr = prep.fit_transform(train[F]).astype(np.float32)
Xca = prep.transform(cal[F]).astype(np.float32); Xte = prep.transform(test[F]).astype(np.float32)
ytr, yca = train.delay.to_numpy(float), cal.delay.to_numpy(float)
groups = train.RunGroup.to_numpy()
T = time.perf_counter
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor


def med(fn, reps=3):
    out = []
    for _ in range(reps):
        t = T(); fn(); out.append(T() - t)
    return float(np.median(out))


res = {}
model = g["build_model"]("Random Forest", seed).fit(Xtr, ytr)
res["point_fit_s"] = med(lambda: g["build_model"]("Random Forest", seed).fit(Xtr, ytr), 1)
res["point_predict_ms"] = 1000 * med(lambda: model.predict(Xte)) / len(Xte)

# OOF residuals: three refits of the point model (needed by OOF normalized CP AND by EC-NCS)
def oof_resid():
    oof = np.full(len(ytr), np.nan)
    for k, (a, b) in enumerate(GroupKFold(3).split(Xtr, groups=groups)):
        m = g["build_model"]("Random Forest", seed + 100 + k).fit(Xtr[a], ytr[a]); oof[b] = m.predict(Xtr[b])
    return np.abs(ytr - oof) + 1e-6
res["oof_residual_refits_s"] = med(oof_resid, 1)
target = oof_resid()
et = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10, random_state=seed + 200, n_jobs=-1)
res["oof_scale_model_fit_s"] = med(lambda: et.fit(Xtr, target), 1)
res["oof_scale_predict_ms"] = 1000 * med(lambda: et.predict(Xte)) / len(Xte)

# CQR: two gradient-boosted quantile models
def cqr_fit():
    ms = []
    for a, off in [(0.05, 0), (0.95, 1)]:
        ms.append(GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=250, max_depth=3,
                                            learning_rate=.04, random_state=seed + off).fit(Xtr, ytr))
    return ms
res["cqr_fit_s"] = med(cqr_fit, 1)
ms = cqr_fit()
res["cqr_predict_ms"] = 1000 * med(lambda: (ms[0].predict(Xte), ms[1].predict(Xte))) / len(Xte)

# EC-NCS: explanations, scale fitting, gate
bg = Xtr[:64]
ref_idx = g["run_balanced_reference_indices"](groups, CFG.max_explanation_reference_rows, seed + 100)
res["ec_explain_predict_ms"] = 1000 * med(lambda: g["explain_strength"]("Random Forest", model, Xte, bg)) / len(Xte)
res["ec_explain_train_cal_s"] = med(lambda: (g["explain_strength"]("Random Forest", model, Xtr[ref_idx], bg),
                                             g["explain_strength"]("Random Forest", model, Xca, bg)))
phi = g["explain_strength"]("Random Forest", model, Xtr[ref_idx], bg)
z = g["EmpiricalCDF"]().fit(phi).transform(phi)
res["ec_scale_fit_s"] = med(lambda: g["fit_explanation_scale"](z, target[ref_idx], groups[ref_idx], 4, .75, 2.0))
res["ec_gate_s"] = med(lambda: g["select_ec_hyperparameters"](train, seed, "Random Forest"), 1)
import json
print(json.dumps(res, indent=1))
Path("/home/claude/rerun/timing_components.json").write_text(json.dumps(res, indent=1))
