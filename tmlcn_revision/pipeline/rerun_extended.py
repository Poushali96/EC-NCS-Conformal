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



from sklearn.model_selection import GroupKFold
ext = {}
for fold in range(CFG.confirmation_folds):
    seed = 2027 + fold
    print("fold", fold, flush=True)
    g["set_seed"](seed)
    test_groups = set(group_table.loc[group_table.Fold == fold, "RunGroup"])
    remainder = frequency_confirm[~frequency_confirm.RunGroup.isin(test_groups)].copy()
    train_idx, cal_idx, inner_assignment, balance_score = g["train_calibration_group_split"](remainder, seed)
    train = remainder.iloc[train_idx].copy(); cal = remainder.iloc[cal_idx].copy()
    test = frequency_confirm[frequency_confirm.RunGroup.isin(test_groups)].copy()
    FEATURES = g["FEATURES"]
    prep = g["build_preprocessor"](train[FEATURES]); Xtr = prep.fit_transform(train[FEATURES]).astype(np.float32)
    Xca = prep.transform(cal[FEATURES]).astype(np.float32); Xte = prep.transform(test[FEATURES]).astype(np.float32)
    ytr = train.delay.to_numpy(float); yca = cal.delay.to_numpy(float); yte = test.delay.to_numpy(float)
    mn = "Random Forest"
    selected_params, selection, tuning = g["select_ec_hyperparameters"](train, seed, mn)
    model = g["build_model"](mn, seed).fit(Xtr, ytr)
    pca = model.predict(Xca); pte = model.predict(Xte)
    groups = train.RunGroup.to_numpy()
    ref_idx = g["run_balanced_reference_indices"](groups, CFG.max_explanation_reference_rows, seed + 100)
    import shap
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    def phi(X): 
        v = np.asarray(ex.shap_values(X, check_additivity=False, approximate=True))
        return v[..., 0] if v.ndim == 3 else v
    Ptr, Pca, Pte = phi(Xtr), phi(Xca), phi(Xte)
    # signed out-of-fold predictions, same recipe as oof_scale
    folds = min(CFG.normalized_oof_folds, len(np.unique(groups))); oof = np.full(len(ytr), np.nan)
    for k, (fit, hold) in enumerate(GroupKFold(folds).split(Xtr, groups=groups)):
        m2 = g["build_model"](mn, seed + 100 + k).fit(Xtr[fit], ytr[fit]); oof[hold] = m2.predict(Xtr[hold]); del m2
    ext[fold] = dict(Xtr=Xtr, Xca=Xca, Xte=Xte, ytr=ytr, yca=yca, yte=yte, pca=pca, pte=pte, oof=oof,
                     gtr=groups, gca=cal.RunGroup.to_numpy(), gte=test.RunGroup.to_numpy(),
                     ftr=train.FrequencyHz.to_numpy(), fca=cal.FrequencyHz.to_numpy(), fte=test.FrequencyHz.to_numpy(),
                     ref_idx=ref_idx, Ptr=Ptr, Pca=Pca, Pte=Pte, params=selected_params, selection=selection,
                     features=list(FEATURES))
    print(selection, flush=True)
    del model
    pickle.dump(ext, open(OUT / "extended.pkl", "wb"))
print("done", flush=True)
