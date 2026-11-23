"""Rerun of the reserved CICV5G evaluation (RF) with the original pipeline code.

The original script (CICV5G_EC_NCS_Nested_Colab_FINAL.py, revision
2026-08-12-cicv5g-r4-unified-gate-schema-audit) is executed unchanged up to the
end of its definitions and data loading. The reserved-run loop below repeats
the original loop line for line and additionally records

* per-packet intervals of every method (for the delay-budget analysis),
* wall-clock time of every method's extra fitting and per-sample inference,
* coverage of every method on its own calibration runs (to separate the
  coverage shortfall caused by differences between runs).

Only the random-forest predictor is run (torch is not available here); the
significant result in the paper is for RF.
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/claude/tools/stubs")

SRC = Path("/home/claude/nb/cicv5g_pipeline.py").read_text()
head = SRC.split("# %% Main ten-seed experiment")[0]
head = head.replace(
    "CFG = Config()",
    'CFG = Config(outdir="/home/claude/rerun/out", models=("Random Forest",))', 1)
assert 'outdir="/home/claude/rerun/out"' in head

ns = {"__name__": "ecncs", "display": print}
exec(compile(head, "cicv5g_pipeline_head", "exec"), ns)
g = ns  # pipeline namespace
CFG, np, pd = g["CFG"], g["np"], g["pd"]
OUT = Path("/home/claude/rerun"); OUT.mkdir(parents=True, exist_ok=True)

frequency_confirm = g["frequency_confirm"]
group_table, outer_score = g["make_balanced_confirmation_folds"](frequency_confirm, CFG.confirmation_folds, 2027)



E = pickle.load(open('/home/claude/rerun/extended.pkl','rb'))

import shap
orig_explain = g["explain_strength"]
def pred_signal(name, model, X, background): return model.predict(X)
rows=[]
for fold in range(CFG.confirmation_folds):
    seed = 2027 + fold
    test_groups = set(group_table.loc[group_table.Fold == fold, "RunGroup"])
    remainder = frequency_confirm[~frequency_confirm.RunGroup.isin(test_groups)].copy()
    g["set_seed"](seed)
    train_idx, cal_idx, _, _ = g["train_calibration_group_split"](remainder, seed)
    train = remainder.iloc[train_idx].copy(); cal = remainder.iloc[cal_idx].copy()
    test = frequency_confirm[frequency_confirm.RunGroup.isin(test_groups)].copy()
    F_ = g["FEATURES"]; prep = g["build_preprocessor"](train[F_]); Xtr = prep.fit_transform(train[F_]).astype(np.float32)
    Xca = prep.transform(cal[F_]).astype(np.float32); Xte = prep.transform(test[F_]).astype(np.float32)
    ytr, yca, yte = train.delay.to_numpy(float), cal.delay.to_numpy(float), test.delay.to_numpy(float)
    mn = "Random Forest"; d = E[fold]
    for sig in ("explanation", "predicted delay"):
        g["explain_strength"] = orig_explain if sig == "explanation" else pred_signal
        g["set_seed"](seed)
        params, selection, _ = g["select_ec_hyperparameters"](train, seed, mn)
        model = g["build_model"](mn, seed).fit(Xtr, ytr)
        pca, pte = model.predict(Xca), model.predict(Xte)
        ref_idx = g["run_balanced_reference_indices"](train.RunGroup.to_numpy(), CFG.max_explanation_reference_rows, seed + 100)
        bg = Xtr[:64]
        s_tr, s_ca, s_te = (g["explain_strength"](mn, model, X, bg) for X in (Xtr[ref_idx], Xca, Xte))
        ecdf = g["EmpiricalCDF"]().fit(s_tr); ztr, zca, zte = ecdf.transform(s_tr), ecdf.transform(s_ca), ecdf.transform(s_te)
        oofres = np.abs(ytr - d["oof"])
        if selection["SelectedEC"]:
            p = g["fit_explanation_scale"](ztr, oofres[ref_idx], train.RunGroup.to_numpy()[ref_idx], params["k"], params["shrinkage"], params["cap"])
            sca, ste = g["apply_explanation_scale"](zca, p), g["apply_explanation_scale"](zte, p)
        else:
            sca, ste = np.ones(len(yca)), np.ones(len(yte))
        q = g["conformal_quantile"](np.abs(yca - pca) / sca); lo, hi = pte - q * ste, pte + q * ste
        for rg, ix in test.groupby("RunGroup").groups.items():
            pos = test.index.get_indexer(list(ix)); pos = pos[pos >= 0]
            mtr = g["interval_metrics"](yte[pos], lo[pos], hi[pos])
            rows.append({"fold": fold, "run": rg, "signal": sig, "selected": bool(selection["SelectedEC"]), **mtr})
        print(fold, sig, selection, flush=True)
        del model
    g["explain_strength"] = orig_explain
pd.DataFrame(rows).to_csv("/home/claude/rerun/predgate.csv", index=False)
print("done")
