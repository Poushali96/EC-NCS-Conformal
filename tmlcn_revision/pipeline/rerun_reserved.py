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
group_table.to_csv(OUT / "fold_assignment.csv", index=False)

packet_store_cal = {}
run_rows, packet_store, timing_rows, calcov_rows, select_rows, split_rows = [], {}, [], [], [], []
T = time.perf_counter
for fold in range(CFG.confirmation_folds):
    seed = 2027 + fold
    print(f"===== reserved fold {fold+1}/5 =====", flush=True)
    g["set_seed"](seed)
    test_groups = set(group_table.loc[group_table.Fold == fold, "RunGroup"])
    remainder = frequency_confirm[~frequency_confirm.RunGroup.isin(test_groups)].copy()
    train_idx, cal_idx, inner_assignment, balance_score = g["train_calibration_group_split"](remainder, seed)
    train = remainder.iloc[train_idx].copy(); cal = remainder.iloc[cal_idx].copy()
    test = frequency_confirm[frequency_confirm.RunGroup.isin(test_groups)].copy()
    g["assert_nonempty_group_split"](train, cal, test, f"fold {fold+1}")
    split_rows.append({"Fold": fold, "TrainRuns": train.RunGroup.nunique(), "CalibrationRuns": cal.RunGroup.nunique(),
                       "TestRuns": test.RunGroup.nunique(), "TrainN": len(train), "CalibrationN": len(cal), "TestN": len(test)})
    FEATURES = g["FEATURES"]
    prep = g["build_preprocessor"](train[FEATURES]); Xtr = prep.fit_transform(train[FEATURES]).astype(np.float32)
    Xca = prep.transform(cal[FEATURES]).astype(np.float32); Xte = prep.transform(test[FEATURES]).astype(np.float32)
    ytr = train.delay.to_numpy(float); yca = cal.delay.to_numpy(float); yte = test.delay.to_numpy(float)
    model_name = "Random Forest"
    tm = {"Fold": fold, "TrainN": len(Xtr), "CalN": len(Xca), "TestN": len(Xte)}

    t0 = T(); selected_params, selection, tuning = g["select_ec_hyperparameters"](train, seed, model_name); tm["EC_gate_s"] = T() - t0
    select_rows.append({"Fold": fold, **selection})
    t0 = T(); model = g["build_model"](model_name, seed).fit(Xtr, ytr); tm["PointModel_fit_s"] = T() - t0
    pca = model.predict(Xca); t0 = T(); pte = model.predict(Xte); tm["PointModel_predict_ms"] = 1000 * (T() - t0) / len(Xte)
    rng = np.random.default_rng(seed)
    ref_idx = g["run_balanced_reference_indices"](train.RunGroup.to_numpy(), CFG.max_explanation_reference_rows, seed + 100)
    background = Xtr[rng.choice(len(Xtr), min(64, len(Xtr)), replace=False)]
    t0 = T(); phi_tr = g["explain_strength"](model_name, model, Xtr[ref_idx], background); tm["EC_explain_ref_s"] = T() - t0
    t0 = T(); phi_ca = g["explain_strength"](model_name, model, Xca, background); tm["EC_explain_cal_s"] = T() - t0
    t0 = T(); phi_te = g["explain_strength"](model_name, model, Xte, background); tm["EC_explain_test_ms"] = 1000 * (T() - t0) / len(Xte)
    ecdf = g["EmpiricalCDF"]().fit(phi_tr); ztr, zca, zte = ecdf.transform(phi_tr), ecdf.transform(phi_ca), ecdf.transform(phi_te)
    residual = np.abs(yca - pca); q = g["conformal_quantile"](residual)
    intervals = {"Standard CP": (pte - q, pte + q)}
    cal_intervals = {"Standard CP": (pca - q, pca + q)}
    t0 = T(); scale, oof_residual = g["oof_scale"](Xtr, ytr, train.RunGroup.to_numpy(), seed, model_name); tm["OOF_scale_fit_s"] = T() - t0
    sca = np.maximum(scale.predict(Xca), 1e-6)
    t0 = T(); ste = np.maximum(scale.predict(Xte), 1e-6); tm["OOF_scale_predict_ms"] = 1000 * (T() - t0) / len(Xte)
    qn = g["conformal_quantile"](residual / sca)
    intervals["OOF Normalized CP"] = (pte - qn * ste, pte + qn * ste); cal_intervals["OOF Normalized CP"] = (pca - qn * sca, pca + qn * sca)
    # CQR, timed: fit two quantile models, then predict
    t0 = T(); lo_c, hi_c = g["cqr"](Xtr, ytr, Xca, yca, Xte, seed); tm["CQR_fit_and_predict_s"] = T() - t0
    intervals["CQR"] = (lo_c, hi_c)
    params = g["fit_explanation_scale"](ztr, oof_residual[ref_idx], train.RunGroup.to_numpy()[ref_idx],
                                         selected_params["k"], selected_params["shrinkage"], selected_params["cap"])
    sec, setest = g["apply_explanation_scale"](zca, params), g["apply_explanation_scale"](zte, params)
    qe = g["conformal_quantile"](residual / sec)
    intervals["Validation-gated EC-NCS"] = (pte - qe * setest, pte + qe * setest)
    cal_intervals["Validation-gated EC-NCS"] = (pca - qe * sec, pca + qe * sec)
    # Per-run metrics exactly as in the original loop
    for run_group, run_indices in test.groupby("RunGroup").groups.items():
        pos = test.index.get_indexer(list(run_indices)); pos = pos[pos >= 0]
        for method, (lo, hi) in intervals.items():
            run_rows.append({"Seed": run_group, "Fold": fold, "Model": model_name,
                             "Protocol": "reserved frequency confirmation", "Method": method,
                             **g["interval_metrics"](yte[pos], lo[pos], hi[pos])})
    # Coverage on the calibration runs themselves, per run
    for run_group, run_indices in cal.groupby("RunGroup").groups.items():
        pos = cal.index.get_indexer(list(run_indices)); pos = pos[pos >= 0]
        for method, (lo, hi) in cal_intervals.items():
            calcov_rows.append({"Fold": fold, "Run": run_group, "Method": method,
                                "Coverage": float(((yca[pos] >= lo[pos]) & (yca[pos] <= hi[pos])).mean())})
    # extra arrays for frequency-conditional (Mondrian) calibration
    cqr_models = []
    for alpha_q, off in [(g["ALPHA"] / 2, 0), (1 - g["ALPHA"] / 2, 1)]:
        cqr_models.append(g["GradientBoostingRegressor"](loss="quantile", alpha=alpha_q, n_estimators=250, max_depth=3,
                                                          learning_rate=.04, random_state=seed + off).fit(Xtr, ytr))
    lca, uca = cqr_models[0].predict(Xca), cqr_models[1].predict(Xca)
    lte, ute = cqr_models[0].predict(Xte), cqr_models[1].predict(Xte)
    packet_store_cal[fold] = {"ycal": yca, "pca": pca, "freq_cal": cal.FrequencyHz.to_numpy(), "run_cal": cal.RunGroup.to_numpy(),
                              "freq_test": test.FrequencyHz.to_numpy(), "sca": sca, "ste": ste, "sec": sec, "setest": setest,
                              "lca": lca, "uca": uca, "lte": lte, "ute": ute}
    packet_store[fold] = {"y": yte, "run": test.RunGroup.to_numpy(), "ytrain": ytr, "pte": pte,
                          "intervals": {m: (np.asarray(a), np.asarray(b)) for m, (a, b) in intervals.items()}}
    timing_rows.append(tm)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm.items()}), flush=True)
    print(selection, flush=True)
    del model
    pd.DataFrame(run_rows).to_csv(OUT / "rerun_run_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT / "rerun_timing.csv", index=False)
    pd.DataFrame(calcov_rows).to_csv(OUT / "rerun_calibration_run_coverage.csv", index=False)
    pd.DataFrame(select_rows).to_csv(OUT / "rerun_selection.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "rerun_split_audit.csv", index=False)
    with open(OUT / "packets.pkl", "wb") as fh:
        pickle.dump(packet_store, fh)
    with open(OUT / "packets_cal.pkl", "wb") as fh:
        pickle.dump(packet_store_cal, fh)
print("done", flush=True)
