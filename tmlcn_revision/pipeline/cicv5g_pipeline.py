"""Nested CICV5G real-network development and locked confirmation for EC-NCS.

Run with:
    %run /content/CICV5G_EC_NCS_Nested_Colab_FINAL.py

The script downloads the official CICV5G repository, validates its schema,
uses complete measurement files as disjoint experimental groups, evaluates
ordinary unseen runs over ten seeds, and treats W2S traces only as an external
distribution-shift stress test. No test labels select models, strata, or
hyperparameters.
"""

# %% Dependencies
import importlib.util, subprocess, sys
_required = {
    "shap": "shap>=0.46", "captum": "captum>=0.7",
    "openpyxl": "openpyxl>=3.1", "psutil": "psutil>=5.9",
}
_missing = [pkg for mod, pkg in _required.items() if importlib.util.find_spec(mod) is None]
if _missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_missing])

import gc, hashlib, json, math, os, random, re, shutil, time, warnings, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil, requests, seaborn as sns, shap
import torch
import torch.nn as nn
from captum.attr import GradientShap
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))


@dataclass
class Config:
    github_zip: str = "https://github.com/zxr805/CICV5G/archive/refs/heads/main.zip"
    zenodo_record: str = "17475688"
    data_dir: str = "/content/cicv5g_data"
    outdir: str = "/content/cicv5g_ec_ncs_nested_results"
    clean_run: bool = True
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)
    nominal_coverage: float = .90
    train_group_fraction: float = .60
    calibration_group_fraction: float = .20
    test_group_fraction: float = .20
    split_candidates: int = 2000
    confirmation_folds: int = 5
    confirmation_split_candidates: int = 5000
    confirmation_min_calibration_runs: int = 2
    models: Tuple[str, ...] = ("Random Forest", "MLP")
    rf_trees: int = 300
    rf_min_leaf: int = 3
    mlp_epochs: int = 80
    mlp_batch_size: int = 512
    mlp_patience: int = 10
    explanation_batch_size: int = 512
    gradient_shap_samples: int = 6
    max_explanation_reference_rows: int = 5000
    explanation_groups: int = 4
    minimum_group_calibration: int = 30
    normalized_oof_folds: int = 3
    bootstrap_resamples: int = 2000
    primary_frequency_hz: int = 20
    restrict_primary_to_20hz: bool = True
    include_coordinates: bool = False
    run_w2s_stress: bool = True
    save_sample_predictions: bool = False
    # Prespecified training-only EC scale grid. Never edit after seeing the
    # confirmatory frequency-test results.
    ec_k_grid: Tuple[int, ...] = (2, 3, 4, 5)
    ec_shrinkage_grid: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    ec_cap_grid: Tuple[float, ...] = (2.0, 4.0)
    ec_scale_floor: float = 0.50
    ec_min_bin_rows: int = 100
    ec_min_validation_gain: float = 0.01


CFG = Config()
ALPHA = 1 - CFG.nominal_coverage
SCRIPT_REVISION = "2026-08-12-cicv5g-r4-unified-gate-schema-audit"
DATA_DIR, OUTDIR = Path(CFG.data_dir), Path(CFG.outdir)
if CFG.clean_run and OUTDIR.exists():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = OUTDIR.with_name(OUTDIR.name + "_backup_" + stamp)
    counter = 1
    while backup.exists():
        backup = OUTDIR.with_name(OUTDIR.name + f"_backup_{stamp}_{counter}"); counter += 1
    shutil.move(str(OUTDIR), str(backup)); print("Previous results preserved:", backup)
for directory in (DATA_DIR, OUTDIR, OUTDIR/"figures", OUTDIR/"supplementary"):
    directory.mkdir(parents=True, exist_ok=True)
print(CFG); print("Script revision:", SCRIPT_REVISION)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def download_stream(url: str, path: Path, timeout=120):
    if path.exists() and path.stat().st_size > 1000: return path
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "EC-NCS-research/1.0"}) as response:
        response.raise_for_status()
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(1024*1024):
                if chunk: handle.write(chunk)
    temporary.replace(path); return path


def obtain_dataset() -> Path:
    archive = DATA_DIR / "CICV5G-main.zip"
    try:
        print("Downloading official CICV5G GitHub archive...")
        download_stream(CFG.github_zip, archive)
    except Exception as github_error:
        print("GitHub download failed; trying the official Zenodo record:", github_error)
        metadata = requests.get(f"https://zenodo.org/api/records/{CFG.zenodo_record}", timeout=60)
        metadata.raise_for_status(); files = metadata.json().get("files", [])
        candidates = [f for f in files if f.get("key", "").lower().endswith((".zip", ".tar.gz"))]
        if not candidates: raise RuntimeError("No downloadable archive found in the CICV5G Zenodo record.")
        item = candidates[0]
        url = item.get("links", {}).get("content") or item.get("links", {}).get("self")
        download_stream(url, archive)
    extract = DATA_DIR / "repository"
    marker = extract / ".complete"
    if not marker.exists():
        if extract.exists(): shutil.rmtree(extract)
        extract.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            root = extract.resolve()
            for member in zf.infolist():
                resolved = (extract/member.filename).resolve()
                if root not in resolved.parents and resolved != root:
                    raise RuntimeError("Unsafe archive member: " + member.filename)
            zf.extractall(extract)
        marker.write_text("complete")
    return extract


def canonical(text): return re.sub(r"[^a-z0-9]", "", str(text).lower())


ALIASES = {
    "delay": {"delay", "delayms", "rtt", "rttms", "latency", "latencyms"},
    "pub_time": {"pubtime", "publishtime", "sendtime"},
    "sub_time": {"subtime", "subscribetime", "receivetime"},
    "RSRP": {"rsrp", "ssrsrp", "nr5grsrp"},
    "SINR": {"sinr", "sssinr", "snr", "nr5gsinr"},
    "Velocity": {"velocity", "speed", "vehiclespeed"},
    "Cell_ID": {"cellid", "cell", "pci", "servingcell"},
    "Heading": {"heading", "yaw"},
    "utmX": {"utmx", "x"}, "utmY": {"utmy", "y"},
}


def read_measurement_file(path: Path) -> pd.DataFrame:
    attempts = [dict(sep=r"\s+", engine="python"), dict(sep="\t"), dict(sep=","), dict(sep=None, engine="python")]
    frame = None
    for options in attempts:
        try:
            candidate = pd.read_csv(path, encoding="utf-8-sig", comment="#", **options)
            if candidate.shape[1] >= 3: frame = candidate; break
        except Exception: pass
    if frame is None: raise ValueError("Could not parse")
    rename = {}
    for column in frame.columns:
        key = canonical(column)
        prefix_map = {"delay":"delay", "latency":"delay", "rtt":"delay",
                      "rsrp":"RSRP", "sinr":"SINR", "snr":"SINR",
                      "velocity":"Velocity", "speed":"Velocity",
                      "cellid":"Cell_ID", "heading":"Heading",
                      "utmx":"utmX", "utmy":"utmY"}
        matched_prefix = next((desired for prefix, desired in prefix_map.items()
                               if key.startswith(prefix)), None)
        if matched_prefix is not None:
            rename[column] = matched_prefix
            continue
        for desired, aliases in ALIASES.items():
            if key in aliases: rename[column] = desired; break
    frame = frame.rename(columns=rename)
    # Some source releases contain synonymous duplicate headers (for example,
    # both delay and delay(ms)). Keep the first canonical field deterministically.
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if "delay" not in frame and {"pub_time", "sub_time"}.issubset(frame):
        frame["delay"] = pd.to_numeric(frame["sub_time"], errors="coerce") - pd.to_numeric(frame["pub_time"], errors="coerce")
    if "delay" not in frame: raise ValueError("No delay field")
    return frame


def parse_metadata(path: Path) -> Dict[str, object]:
    text = "/".join(path.parts).lower()
    if "urban" in text: scenario = "urban"
    elif "arterial" in text: scenario = "arterial"
    elif "rural" in text or "off road" in text or "offroad" in text: scenario = "rural_offroad"
    else: scenario = "unknown"
    if re.search(r"(^|[^a-z0-9])n78([^a-z0-9]|$)", text): mode = "n78_public"
    elif re.search(r"(^|[^a-z0-9])n8([^a-z0-9]|$)", text): mode = "n8_private"
    else: mode = "unknown"
    match = re.search(r"(?:^|[^0-9])(?:hz|freq(?:uency)?)[_ -]?(10|20|33|100)(?:[^0-9]|$)", text)
    if not match: match = re.search(r"(?:^|[_ -])(10|20|33|100)hz(?:[_ .-]|$)", text)
    frequency_stress = "data transmission frequency" in text
    period_match = re.search(r"(?:^|[/_ -])(10|30|50|100)ms(?:[/_ .-]|$)", text)
    if match: frequency = int(match.group(1))
    elif period_match: frequency = int(round(1000/int(period_match.group(1))))
    else: frequency = CFG.primary_frequency_hz
    stress = bool(re.search(r"w2s|strong.?to.?weak|weak.?to.?strong", text))
    return {"Scenario": scenario, "NetworkMode": mode, "FrequencyHz": frequency,
            "IsW2S": stress, "IsFrequencyStress": frequency_stress, "RunGroup": str(path)}


def load_all_data(root: Path):
    files = sorted([p for p in root.rglob("*.txt") if "data" in str(p.parent).lower()])
    if not files: files = sorted(root.rglob("*.txt"))
    # The repository contains convenience files such as westn8all.txt in
    # addition to the constituent run files. Loading both would duplicate
    # observations and invalidate run-disjoint evaluation.
    files = [p for p in files if not re.search(r"(?:all|merged|combined)$", p.stem.lower())]
    rows, audit, seen_digests = [], [], set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_digests:
            audit.append({"File": str(path), "Status": "skipped", "Reason": "exact duplicate file", "Rows": 0})
            continue
        seen_digests.add(digest)
        try:
            frame = read_measurement_file(path)
        except Exception as error:
            audit.append({"File": str(path), "Status": "skipped", "Reason": str(error), "Rows": 0}); continue
        meta = parse_metadata(path)
        for key, value in meta.items(): frame[key] = value
        for col in ("delay", "RSRP", "SINR", "Velocity", "Heading", "utmX", "utmY"):
            if col in frame: frame[col] = pd.to_numeric(frame[col], errors="coerce")
        before = len(frame)
        frame = frame[np.isfinite(frame.delay) & (frame.delay >= 0)].copy()
        if len(frame): rows.append(frame)
        audit.append({"File": str(path), "Status": "loaded", "Reason": "", "RowsRaw": before,
                      "Rows": len(frame), **meta})
    if not rows: raise RuntimeError("No valid CICV5G measurement files were loaded.")
    data = pd.concat(rows, ignore_index=True, sort=False)
    audit = pd.DataFrame(audit)
    data["RowInRun"] = data.groupby("RunGroup").cumcount()
    return data, audit


ROOT = obtain_dataset()
all_data, file_audit = load_all_data(ROOT)
file_audit.to_csv(OUTDIR/"file_audit.csv", index=False)
print("Loaded rows:", len(all_data), "runs:", all_data.RunGroup.nunique())
print(all_data.groupby(["Scenario", "NetworkMode", "IsW2S"], dropna=False).size())


def audit_and_clean_physical_fields(data):
    """Reject impossible radio values without silently relabelling columns.

    CICV5G documents RSRP and SINR separately from UTM coordinates. Values at
    coordinate scale (for example, millions) cannot be radio measurements.
    Invalid subset/field combinations are replaced by missing values so they
    cannot enter models or physical-stratum claims; the decision is recorded.
    """
    checks={"RSRP":(-200.0,0.0),"SINR":(-50.0,100.0),
            "Velocity":(-1.0,120.0)}
    cleaned=data.copy(); rows=[]
    subset_masks={"ordinary":~cleaned.IsW2S,
                  "W2S":cleaned.IsW2S}
    for subset,mask in subset_masks.items():
        for field,(lower,upper) in checks.items():
            if field not in cleaned:
                rows.append({"Subset":subset,"Field":field,"N":0,
                             "FiniteN":0,"ValidFraction":0.0,"Status":"missing"})
                continue
            values=pd.to_numeric(cleaned.loc[mask,field],errors="coerce")
            finite=values[np.isfinite(values)]
            valid=((finite>=lower)&(finite<=upper))
            fraction=float(valid.mean()) if len(finite) else 0.0
            record={"Subset":subset,"Field":field,"N":int(mask.sum()),
                    "FiniteN":int(len(finite)),"Min":float(finite.min()) if len(finite) else np.nan,
                    "Median":float(finite.median()) if len(finite) else np.nan,
                    "Max":float(finite.max()) if len(finite) else np.nan,
                    "ExpectedLower":lower,"ExpectedUpper":upper,"ValidFraction":fraction}
            # Require a strongly dominant physically plausible range. Sparse
            # individual sentinels are set missing; wholesale schema failures
            # invalidate the field in that subset.
            if fraction>=.95:
                bad=mask & (~pd.to_numeric(cleaned[field],errors="coerce").between(lower,upper))
                cleaned.loc[bad,field]=np.nan
                record["Status"]="valid; out-of-range rows set missing"
            else:
                cleaned.loc[mask,field]=np.nan
                record["Status"]="invalidated: implausible physical range"
            rows.append(record)
    audit=pd.DataFrame(rows)
    return cleaned,audit


all_data,physical_field_audit=audit_and_clean_physical_fields(all_data)
physical_field_audit.to_csv(OUTDIR/"physical_field_audit.csv",index=False)
run_physical_audit=[]
for run_group,block in all_data.groupby("RunGroup",sort=False):
    record={"RunGroup":run_group,"IsW2S":bool(block.IsW2S.iloc[0]),
            "Scenario":block.Scenario.iloc[0],"NetworkMode":block.NetworkMode.iloc[0],"N":len(block)}
    for field in ("RSRP","SINR","Velocity"):
        values=pd.to_numeric(block.get(field),errors="coerce").dropna() if field in block else pd.Series(dtype=float)
        record[f"{field}_FiniteN"]=len(values)
        record[f"{field}_Min"]=values.min() if len(values) else np.nan
        record[f"{field}_Median"]=values.median() if len(values) else np.nan
        record[f"{field}_Max"]=values.max() if len(values) else np.nan
    run_physical_audit.append(record)
run_physical_audit=pd.DataFrame(run_physical_audit)
run_physical_audit.to_csv(OUTDIR/"run_physical_field_audit.csv",index=False)
print("\nPhysical-field audit:")
display(physical_field_audit)

ordinary = all_data[(~all_data.IsW2S) & (~all_data.IsFrequencyStress)].copy()
if CFG.restrict_primary_to_20hz:
    candidate = ordinary[ordinary.FrequencyHz == CFG.primary_frequency_hz].copy()
    if candidate.RunGroup.nunique() >= 15: ordinary = candidate
    else: print("Too few detected 20 Hz runs; using all ordinary frequencies with FrequencyHz as a feature.")
w2s = all_data[all_data.IsW2S].copy()
frequency_confirm = all_data[(~all_data.IsW2S) & (all_data.IsFrequencyStress)].copy()
if ordinary.RunGroup.nunique() < 10:
    raise RuntimeError("At least ten ordinary run files are required for group-disjoint validation.")


def available_features(frame):
    features = [c for c in ["RSRP", "SINR", "Velocity", "Heading", "FrequencyHz",
                             "NetworkMode", "Scenario", "Cell_ID"] if c in frame and frame[c].notna().any()]
    if CFG.include_coordinates:
        features += [c for c in ["utmX", "utmY"] if c in frame and frame[c].notna().any()]
    return features


FEATURES = available_features(ordinary)
if not {"RSRP", "SINR"}.intersection(FEATURES):
    raise RuntimeError("Neither RSRP nor SINR was found; inspect file_audit.csv and source schema.")
print("Model features:", FEATURES)
VALID_SIGNAL_FIELDS=[field for field in ("RSRP","SINR")
                     if field in ordinary and ordinary[field].notna().mean()>=.50]
if not VALID_SIGNAL_FIELDS:
    raise RuntimeError("No physically valid radio indicator remains after schema audit.")
PRIMARY_SIGNAL_FIELD=VALID_SIGNAL_FIELDS[0]
W2S_SIGNAL_FIELD=next((field for field in ("RSRP","SINR")
                       if field in w2s and w2s[field].notna().mean()>=.50),None)
print("Validated physical signal fields:",VALID_SIGNAL_FIELDS,
      "| primary diagnostic:",PRIMARY_SIGNAL_FIELD,
      "| W2S trace field:",W2S_SIGNAL_FIELD)
print("Reserved transmission-frequency confirmation:", len(frequency_confirm),
      "rows in", frequency_confirm.RunGroup.nunique(), "complete runs")


def split_score(group_table, assignments):
    score = 0.0
    global_dist = group_table.Stratum.value_counts(normalize=True)
    target = [CFG.train_group_fraction, CFG.calibration_group_fraction, CFG.test_group_fraction]
    n = len(group_table)
    for part, fraction in enumerate(target):
        subset = group_table.iloc[np.flatnonzero(assignments == part)]
        if subset.empty: return 1e9
        score += 4 * abs(len(subset)/n - fraction)
        dist = subset.Stratum.value_counts(normalize=True)
        score += sum(abs(dist.get(k, 0)-v) for k, v in global_dist.items())
    return score


def group_split(frame, seed):
    table = frame.groupby("RunGroup", as_index=False).agg(
        Scenario=("Scenario", "first"), NetworkMode=("NetworkMode", "first"),
        FrequencyHz=("FrequencyHz", "first"), Rows=("delay", "size"))
    table["Stratum"] = table.Scenario.astype(str)+"|"+table.NetworkMode.astype(str)+"|"+table.FrequencyHz.astype(str)
    rng = np.random.default_rng(seed); best, best_score = None, np.inf
    n = len(table); ntr=max(1,round(CFG.train_group_fraction*n)); nca=max(1,round(CFG.calibration_group_fraction*n))
    if ntr+nca >= n: ntr, nca = n-2, 1
    for _ in range(CFG.split_candidates):
        order = rng.permutation(n); assignment = np.full(n, 2); assignment[order[:ntr]]=0; assignment[order[ntr:ntr+nca]]=1
        value = split_score(table, assignment)
        if value < best_score: best, best_score = assignment.copy(), value
    groups = [set(table.loc[best==part, "RunGroup"]) for part in range(3)]
    indices = [np.flatnonzero(frame.RunGroup.isin(group).to_numpy()) for group in groups]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]: raise AssertionError("Group overlap")
    return (*indices, table.assign(Partition=np.array(["train","calibration","test"])[best]), best_score)


def assert_nonempty_group_split(train, calibration, test, context):
    """Fail before preprocessing if any complete-run partition is invalid."""
    partitions = {"train": train, "calibration": calibration, "test": test}
    empty = [name for name, part in partitions.items() if len(part) == 0]
    if empty:
        raise RuntimeError(f"{context}: empty partition(s): {empty}")
    run_sets = {name: set(part.RunGroup.astype(str)) for name, part in partitions.items()}
    if run_sets["train"] & run_sets["calibration"]:
        raise AssertionError(f"{context}: train/calibration run overlap")
    if run_sets["train"] & run_sets["test"]:
        raise AssertionError(f"{context}: train/test run overlap")
    if run_sets["calibration"] & run_sets["test"]:
        raise AssertionError(f"{context}: calibration/test run overlap")
    if len(run_sets["train"]) < 2:
        raise RuntimeError(f"{context}: fewer than two training runs")
    if len(run_sets["calibration"]) < 1:
        raise RuntimeError(f"{context}: no calibration run")
    if len(run_sets["test"]) < 1:
        raise RuntimeError(f"{context}: no test run")


def make_balanced_confirmation_folds(frame, n_splits, seed):
    """Assign every reserved complete run to one globally balanced test fold.

    The earlier stratum-local round-robin restarted at fold zero for every
    stratum. With many singleton strata, this concentrated most runs in fold
    zero and could leave an empty calibration remainder. Here candidate fold
    labels have exactly balanced run counts globally; the objective then
    balances row mass and the NetworkMode/FrequencyHz distribution.
    """
    table = frame.groupby("RunGroup", as_index=False).agg(
        NetworkMode=("NetworkMode", "first"),
        FrequencyHz=("FrequencyHz", "first"),
        Rows=("delay", "size"),
    )
    table["Stratum"] = (table.NetworkMode.astype(str) + "|" +
                        table.FrequencyHz.astype(str))
    n_runs = len(table)
    if n_runs < 2 * n_splits:
        raise RuntimeError(
            f"Reserved confirmation needs at least {2*n_splits} complete runs "
            f"for {n_splits} folds; found {n_runs}."
        )
    base_labels = np.arange(n_runs) % n_splits
    global_dist = table.Stratum.value_counts(normalize=True)
    total_rows = float(table.Rows.sum())
    rng = np.random.default_rng(seed)
    best_assignment, best_score = None, np.inf
    for _ in range(CFG.confirmation_split_candidates):
        assignment = base_labels[rng.permutation(n_runs)]
        score = 0.0
        valid = True
        for fold in range(n_splits):
            subset = table.loc[assignment == fold]
            if subset.empty:
                valid = False
                break
            score += 2.0 * abs(len(subset) / n_runs - 1.0 / n_splits)
            score += 2.0 * abs(float(subset.Rows.sum()) / total_rows - 1.0 / n_splits)
            dist = subset.Stratum.value_counts(normalize=True)
            score += sum(abs(dist.get(key, 0.0) - value)
                         for key, value in global_dist.items())
        if valid and score < best_score:
            best_assignment, best_score = assignment.copy(), float(score)
    if best_assignment is None:
        raise RuntimeError("Could not construct nonempty balanced confirmation folds.")
    table["Fold"] = best_assignment
    counts = table.groupby("Fold").agg(Runs=("RunGroup", "size"), Rows=("Rows", "sum"))
    if len(counts) != n_splits or (counts.Runs < 1).any():
        raise AssertionError("Balanced confirmation fold construction failed.")
    return table, best_score


def train_calibration_group_split(frame, seed):
    """Split the non-test reserved runs into nonempty train/calibration runs."""
    table = frame.groupby("RunGroup", as_index=False).agg(
        NetworkMode=("NetworkMode", "first"),
        FrequencyHz=("FrequencyHz", "first"),
        Rows=("delay", "size"),
    )
    table["Stratum"] = (table.NetworkMode.astype(str) + "|" +
                        table.FrequencyHz.astype(str))
    n_runs = len(table)
    min_cal = CFG.confirmation_min_calibration_runs
    if n_runs < min_cal + 4:
        raise RuntimeError(
            f"Need at least {min_cal+4} non-test runs for confirmation "
            f"training/calibration; found {n_runs}."
        )
    n_cal = max(min_cal, int(round(CFG.calibration_group_fraction * n_runs)))
    n_cal = min(n_cal, n_runs - 4)
    global_dist = table.Stratum.value_counts(normalize=True)
    total_rows = float(table.Rows.sum())
    rng = np.random.default_rng(seed)
    best_calibration, best_score = None, np.inf
    for _ in range(CFG.confirmation_split_candidates):
        calibration_indices = rng.choice(n_runs, n_cal, replace=False)
        mask = np.zeros(n_runs, dtype=bool)
        mask[calibration_indices] = True
        cal_table, train_table = table.loc[mask], table.loc[~mask]
        if cal_table.empty or train_table.empty:
            continue
        cal_dist = cal_table.Stratum.value_counts(normalize=True)
        train_dist = train_table.Stratum.value_counts(normalize=True)
        score = 2.0 * abs(float(cal_table.Rows.sum()) / total_rows - n_cal / n_runs)
        score += sum(abs(cal_dist.get(key, 0.0) - value)
                     for key, value in global_dist.items())
        score += 0.5 * sum(abs(train_dist.get(key, 0.0) - value)
                           for key, value in global_dist.items())
        if score < best_score:
            best_calibration, best_score = mask.copy(), float(score)
    if best_calibration is None:
        raise RuntimeError("Could not construct reserved train/calibration split.")
    train_groups = set(table.loc[~best_calibration, "RunGroup"])
    calibration_groups = set(table.loc[best_calibration, "RunGroup"])
    train_indices = np.flatnonzero(frame.RunGroup.isin(train_groups).to_numpy())
    calibration_indices = np.flatnonzero(frame.RunGroup.isin(calibration_groups).to_numpy())
    assignment = np.where(best_calibration, "calibration", "train")
    return train_indices, calibration_indices, table.assign(Partition=assignment), best_score


def save_partial_csv(name, rows):
    """Write recoverable progress after every seed/fold without changing inference."""
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(OUTDIR / name, index=False)


def build_preprocessor(frame):
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in FEATURES if c not in numeric]
    transformers=[]
    if numeric: transformers.append(("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric))
    if categorical: transformers.append(("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2))]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class TorchMLP:
    def __init__(self, seed): self.seed=seed
    def fit(self, X, y):
        set_seed(self.seed); n=X.shape[1]
        self.net=nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1)).to(DEVICE)
        idx=np.arange(len(X)); np.random.default_rng(self.seed).shuffle(idx); cut=max(1,int(.85*len(idx))); fit,val=idx[:cut],idx[cut:]
        mean,std=float(np.mean(y[fit])),float(np.std(y[fit])+1e-8); self.y_mean,self.y_std=mean,std
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.SmoothL1Loss()
        best,best_state,wait=np.inf,None,0
        for epoch in range(CFG.mlp_epochs):
            self.net.train(); order=np.random.default_rng(self.seed+epoch).permutation(fit)
            for start in range(0,len(order),CFG.mlp_batch_size):
                b=order[start:start+CFG.mlp_batch_size]; xb=torch.tensor(X[b],dtype=torch.float32,device=DEVICE)
                target=torch.tensor((y[b]-mean)/std,dtype=torch.float32,device=DEVICE).view(-1,1)
                optimizer.zero_grad(); loss=loss_fn(self.net(xb),target); loss.backward(); optimizer.step()
            self.net.eval()
            with torch.no_grad():
                pv=self.net(torch.tensor(X[val],dtype=torch.float32,device=DEVICE)).cpu().numpy().ravel()*std+mean
            metric=float(np.mean(np.abs(y[val]-pv)))
            if metric<best-1e-5: best=metric; best_state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}; wait=0
            else: wait+=1
            if wait>=CFG.mlp_patience: break
        self.net.load_state_dict(best_state); return self
    def predict(self,X):
        self.net.eval(); out=[]
        with torch.no_grad():
            for start in range(0,len(X),CFG.explanation_batch_size):
                out.append(self.net(torch.tensor(X[start:start+CFG.explanation_batch_size],dtype=torch.float32,device=DEVICE)).cpu().numpy().ravel())
        return np.concatenate(out)*self.y_std+self.y_mean


def build_model(name,seed):
    return (RandomForestRegressor(n_estimators=CFG.rf_trees,min_samples_leaf=CFG.rf_min_leaf,
                                  random_state=seed,n_jobs=-1,max_features=.8)
            if name=="Random Forest" else TorchMLP(seed))


def explain_strength(name, model, X, background):
    if name=="Random Forest":
        explainer=shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
        try: values=explainer.shap_values(X,check_additivity=False,approximate=True)
        except TypeError: values=explainer.shap_values(X,check_additivity=False)
        values=np.asarray(values)
        if values.ndim==3: values=values[...,0]
        return np.abs(values).mean(axis=1)
    baseline=torch.tensor(background,dtype=torch.float32,device=DEVICE)
    gradient=GradientShap(model.net); output=[]
    for start in range(0,len(X),CFG.explanation_batch_size):
        batch=torch.tensor(X[start:start+CFG.explanation_batch_size],dtype=torch.float32,device=DEVICE,requires_grad=True)
        attrs=gradient.attribute(batch,baselines=baseline,n_samples=CFG.gradient_shap_samples,stdevs=0.0)
        output.append(attrs.detach().abs().mean(dim=1).cpu().numpy())
    return np.concatenate(output)


class EmpiricalCDF:
    def fit(self,x): self.sorted=np.sort(np.asarray(x,float)); return self
    def transform(self,x): return np.searchsorted(self.sorted,np.asarray(x,float),side="right")/max(1,len(self.sorted))


def conformal_quantile(scores):
    scores=np.asarray(scores,float).reshape(-1)
    scores=scores[np.isfinite(scores)]
    if len(scores)==0:
        raise RuntimeError("Conformal calibration received no finite scores.")
    level=min(1.0,math.ceil((len(scores)+1)*(1-ALPHA))/len(scores))
    return float(np.quantile(scores,level,method="higher"))


def interval_metrics(y,lo,hi):
    covered=(y>=lo)&(y<=hi); width=hi-lo; penalty=(2/ALPHA)*(lo-y)*(y<lo)+(2/ALPHA)*(y-hi)*(y>hi)
    return {"N":len(y),"Coverage":float(covered.mean()),"CoverageError":float(abs(covered.mean()-CFG.nominal_coverage)),
            "MeanWidth":float(width.mean()),"MedianWidth":float(np.median(width)),"P90Width":float(np.quantile(width,.9)),
            "Winkler":float(np.mean(width+penalty))}


def mondrian_radii(residual,cal_group,test_group):
    fallback=conformal_quantile(residual); out=np.full(len(test_group),fallback)
    for group in np.unique(test_group):
        values=residual[cal_group==group]
        if len(values)>=CFG.minimum_group_calibration: out[test_group==group]=conformal_quantile(values)
    return out


def oof_scale(X,y,groups,seed,model_name):
    folds=min(CFG.normalized_oof_folds,len(np.unique(groups))); splitter=GroupKFold(folds); oof=np.full(len(y),np.nan)
    for fold,(fit,hold) in enumerate(splitter.split(X,groups=groups)):
        # The residual-scale target must come from OOF predictions of the same
        # point-model family; otherwise the normalized baseline is mismatched.
        model=build_model(model_name,seed+100+fold).fit(X[fit],y[fit])
        oof[hold]=model.predict(X[hold])
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    target=np.abs(y-oof)+1e-6
    scale_model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=10,random_state=seed+200,n_jobs=-1).fit(X,target)
    return scale_model,target


def weighted_quantile(values, quantile, weights):
    """Deterministic weighted empirical quantile (used only on training data)."""
    values=np.asarray(values,float); weights=np.asarray(weights,float)
    mask=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    values,weights=values[mask],weights[mask]
    if not len(values): return np.nan
    order=np.argsort(values); values,weights=values[order],weights[order]
    cumulative=np.cumsum(weights)/weights.sum()
    return float(values[min(np.searchsorted(cumulative,quantile,side="left"),len(values)-1)])


def run_balanced_weights(groups):
    groups=np.asarray(groups).astype(str); counts=pd.Series(groups).value_counts()
    weights=np.array([1.0/counts[g] for g in groups],float)
    return weights/weights.mean()


def run_balanced_reference_indices(groups,max_rows,seed):
    """Bound explanation cost while spreading the reference across all runs."""
    groups=np.asarray(groups).astype(str); rng=np.random.default_rng(seed); unique=np.unique(groups)
    quota=max(1,max_rows//max(1,len(unique))); selected=[]
    for group in unique:
        indices=np.flatnonzero(groups==group)
        selected.extend(rng.choice(indices,min(quota,len(indices)),replace=False).tolist())
    if len(selected)<min(max_rows,len(groups)):
        remaining=np.setdiff1d(np.arange(len(groups)),np.asarray(selected,int),assume_unique=False)
        selected.extend(rng.choice(remaining,min(max_rows-len(selected),len(remaining)),replace=False).tolist())
    return np.asarray(selected[:max_rows],int)


def fit_explanation_scale(z,residual,run_groups,k,shrinkage,cap):
    """Learn a positive explanation-only scale using training residuals.

    The scale is fixed before conformal calibration. Consequently, calibrating
    |Y-f(X)|/s(X) on an untouched calibration set retains ordinary marginal
    split-conformal validity under exchangeability, regardless of whether the
    learned explanation scale is useful.
    """
    z=np.asarray(z,float); residual=np.asarray(residual,float)+1e-8
    weights=run_balanced_weights(run_groups)
    global_median=weighted_quantile(residual,.5,weights)
    edges=np.linspace(0,1,k+1); bins=np.digitize(z,edges[1:-1])
    ratios=np.ones(k,float)
    for group in range(k):
        mask=bins==group
        if mask.sum()>=CFG.ec_min_bin_rows:
            local=weighted_quantile(residual[mask],.5,weights[mask])
            raw=max(local/max(global_median,1e-8),1e-4)
            ratios[group]=np.exp(shrinkage*np.log(raw))
    ratios=np.clip(ratios,CFG.ec_scale_floor,cap)
    # Identifiability normalization does not change the final interval after
    # conformal calibration, but makes scale ratios comparable across seeds.
    ratios=ratios/max(weighted_quantile(ratios[bins],.5,weights),1e-8)
    ratios=np.clip(ratios,CFG.ec_scale_floor,cap)
    return {"k":int(k),"shrinkage":float(shrinkage),"cap":float(cap),
            "edges":edges.tolist(),"ratios":ratios.tolist()}


def apply_explanation_scale(z,params):
    edges=np.asarray(params["edges"],float); ratios=np.asarray(params["ratios"],float)
    return ratios[np.digitize(np.asarray(z,float),edges[1:-1])]


def worst_signal_error(reference,test,y,lo,hi):
    strata=dict(physical_strata(reference,test))
    key=f"{PRIMARY_SIGNAL_FIELD}-tertile"
    if key not in strata: return abs(np.mean((y>=lo)&(y<=hi))-CFG.nominal_coverage)
    labels=strata[key]; errors=[]
    for label in np.unique(labels):
        mask=labels==label
        errors.append(abs(np.mean((y[mask]>=lo[mask])&(y[mask]<=hi[mask]))-CFG.nominal_coverage))
    return float(max(errors))


def select_ec_hyperparameters(train_frame,seed,model_name):
    """Nested selection using only the outer-training runs."""
    fit_idx,cal_idx,val_idx,_,_=group_split(train_frame,seed+31000)
    fit_df=train_frame.iloc[fit_idx].copy(); cal_df=train_frame.iloc[cal_idx].copy(); val_df=train_frame.iloc[val_idx].copy()
    prep=build_preprocessor(fit_df[FEATURES]); Xfit=prep.fit_transform(fit_df[FEATURES]).astype(np.float32)
    Xcal=prep.transform(cal_df[FEATURES]).astype(np.float32); Xval=prep.transform(val_df[FEATURES]).astype(np.float32)
    yfit=fit_df.delay.to_numpy(float); ycal=cal_df.delay.to_numpy(float); yval=val_df.delay.to_numpy(float)
    model=build_model(model_name,seed+32000).fit(Xfit,yfit)
    pcal=model.predict(Xcal); pval=model.predict(Xval)
    rng=np.random.default_rng(seed+33000); background=Xfit[rng.choice(len(Xfit),min(64,len(Xfit)),replace=False)]
    ref_idx=run_balanced_reference_indices(fit_df.RunGroup.to_numpy(),CFG.max_explanation_reference_rows,seed+33100)
    phi_fit=explain_strength(model_name,model,Xfit[ref_idx],background)
    phi_cal=explain_strength(model_name,model,Xcal,background); phi_val=explain_strength(model_name,model,Xval,background)
    ecdf=EmpiricalCDF().fit(phi_fit); zfit=ecdf.transform(phi_fit); zcal=ecdf.transform(phi_cal); zval=ecdf.transform(phi_val)
    _,fit_oof_residual=oof_scale(Xfit,yfit,fit_df.RunGroup.to_numpy(),seed+34000,model_name)
    cal_residual=np.abs(ycal-pcal); q0=conformal_quantile(cal_residual)
    standard=interval_metrics(yval,pval-q0,pval+q0)
    standard_worst=worst_signal_error(fit_df,val_df,yval,pval-q0,pval+q0)
    standard_objective=standard_worst + .25*(standard["Winkler"]/max(standard["Winkler"],1e-8))
    audit=[{"Candidate":"Standard fallback","K":1,"Shrinkage":0.0,"Cap":1.0,
            "Coverage":standard["Coverage"],"CoverageError":standard["CoverageError"],
            "WorstRSRPError":standard_worst,"Winkler":standard["Winkler"],
            "Objective":standard_objective}]
    candidates=[]
    for k in CFG.ec_k_grid:
        for shrinkage in CFG.ec_shrinkage_grid:
            for cap in CFG.ec_cap_grid:
                params=fit_explanation_scale(zfit,fit_oof_residual[ref_idx],fit_df.RunGroup.to_numpy()[ref_idx],k,shrinkage,cap)
                sc=apply_explanation_scale(zcal,params); sv=apply_explanation_scale(zval,params)
                q=conformal_quantile(cal_residual/sc); lo,hi=pval-q*sv,pval+q*sv
                metrics=interval_metrics(yval,lo,hi); worst=worst_signal_error(fit_df,val_df,yval,lo,hi)
                objective=worst + .25*(metrics["Winkler"]/max(standard["Winkler"],1e-8))
                record={"Candidate":"EC-NCS","K":k,"Shrinkage":shrinkage,"Cap":cap,
                        "Coverage":metrics["Coverage"],"CoverageError":metrics["CoverageError"],
                        "WorstRSRPError":worst,"Winkler":metrics["Winkler"],"Objective":objective}
                audit.append(record); candidates.append((objective,params,record))
    candidates.sort(key=lambda item:item[0]); best_objective,best_params,best_record=candidates[0]
    relative_gain=(standard_objective-best_objective)/max(abs(standard_objective),1e-8)
    # Coverage guardrail and minimum gain are fixed prospectively. If either
    # fails, explanations are ignored and the method becomes Standard CP.
    use_ec=(relative_gain>=CFG.ec_min_validation_gain and
            best_record["Coverage"]>=CFG.nominal_coverage-.02)
    selected=(best_params if use_ec else {"k":1,"shrinkage":0.0,"cap":1.0,
              "edges":[0.0,1.0],"ratios":[1.0]})
    decision={"SelectedEC":bool(use_ec),"RelativeValidationGain":relative_gain,
              "SelectedK":selected["k"],"SelectedShrinkage":selected["shrinkage"],
              "SelectedCap":selected["cap"],"StandardObjective":standard_objective,
              "BestECObjective":best_objective}
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return selected,decision,pd.DataFrame(audit)


def cqr(Xtr,ytr,Xca,yca,Xte,seed):
    models=[]
    for alpha,offset in [(ALPHA/2,0),(1-ALPHA/2,1)]:
        m=GradientBoostingRegressor(loss="quantile",alpha=alpha,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed+offset).fit(Xtr,ytr); models.append(m)
    lca,uca=models[0].predict(Xca),models[1].predict(Xca); score=np.maximum(lca-yca,yca-uca); q=conformal_quantile(score)
    return models[0].predict(Xte)-q,models[1].predict(Xte)+q


def physical_strata(train,test):
    columns=[]; names=[]
    for signal_field in VALID_SIGNAL_FIELDS:
        valid=pd.to_numeric(train[signal_field],errors="coerce").dropna()
        edges=np.unique(valid.quantile([1/3,2/3]).to_numpy()) if len(valid) else np.array([])
        if len(edges)==2:
            values=pd.to_numeric(test[signal_field],errors="coerce").fillna(valid.median()).to_numpy()
            columns.append(np.digitize(values,edges)); names.append(f"{signal_field}-tertile")
    columns += [test[c].astype(str).to_numpy() for c in ["Scenario","NetworkMode"] if c in test]
    names += [c for c in ["Scenario","NetworkMode"] if c in test]
    return list(zip(names,columns))


def evaluate_intervals(intervals,y,reference_train,test,seed,model,protocol):
    overall=[]; groups=[]
    for method,(lo,hi) in intervals.items():
        overall.append({"Seed":seed,"Model":model,"Protocol":protocol,"Method":method,**interval_metrics(y,lo,hi)})
        for group_type,labels in physical_strata(reference_train,test):
            for label in sorted(pd.unique(labels)):
                mask=labels==label
                if mask.sum(): groups.append({"Seed":seed,"Model":model,"Protocol":protocol,"Method":method,
                    "GroupType":group_type,"Group":str(label),**interval_metrics(y[mask],lo[mask],hi[mask])})
    return overall,groups


def bootstrap_ci(frame,keys,metrics):
    rng=np.random.default_rng(2027); rows=[]
    for key,group in frame.groupby(keys,dropna=False):
        key=(key,) if not isinstance(key,tuple) else key
        for metric in metrics:
            values=group.groupby("Seed")[metric].mean().dropna().to_numpy()
            if not len(values): continue
            means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(CFG.bootstrap_resamples)])
            rows.append({**dict(zip(keys,key)),"Metric":metric,"Mean":values.mean(),"CI95Lower":np.quantile(means,.025),"CI95Upper":np.quantile(means,.975),"Seeds":len(values)})
    return pd.DataFrame(rows)


def paired_tests(frame,reference="Standard CP",metrics=("CoverageError","MeanWidth","Winkler"),
                 group_columns=("Protocol","Model")):
    rows=[]
    for key,block in frame.groupby(list(group_columns),dropna=False):
        key=(key,) if not isinstance(key,tuple) else key
        key_fields=dict(zip(group_columns,key))
        for method in sorted(set(block.Method)-{reference}):
            for metric in metrics:
                pivot=block[block.Method.isin([reference,method])].pivot_table(index="Seed",columns="Method",values=metric,aggfunc="mean").dropna()
                if len(pivot)<5: continue
                delta=pivot[method]-pivot[reference]
                try: p=float(wilcoxon(delta).pvalue)
                except ValueError: p=1.0
                rows.append({**key_fields,"Reference":reference,"Method":method,"Metric":metric,
                             "MeanPairedDifference":delta.mean(),"WilcoxonP":p,"Seeds":len(delta)})
    out=pd.DataFrame(rows)
    if len(out):
        order=np.argsort(out.WilcoxonP.to_numpy()); adjusted=np.empty(len(out)); running=0
        for rank,index in enumerate(order):
            running=max(running,min(1,(len(out)-rank)*out.WilcoxonP.iloc[index])); adjusted[index]=running
        out["HolmAdjustedP"]=adjusted
    return out


# %% Main ten-seed experiment
overall_rows=[]; group_rows=[]; prediction_rows=[]; split_rows=[]; latency_rows=[]; stress_rows=[]; sample_rows=[]
operational_rows=[]; degradation_rows=[]; ablation_rows=[]
tuning_rows=[]; selection_rows=[]
for seed in CFG.seeds:
    print(f"\n===== CICV5G seed {seed} ====="); set_seed(seed)
    tr_idx,ca_idx,te_idx,group_table,balance_score=group_split(ordinary,seed)
    train,cal,test=ordinary.iloc[tr_idx].copy(),ordinary.iloc[ca_idx].copy(),ordinary.iloc[te_idx].copy()
    assert_nonempty_group_split(train,cal,test,f"ordinary seed {seed}")
    split_rows.append({"Seed":seed,"TrainRuns":train.RunGroup.nunique(),"CalibrationRuns":cal.RunGroup.nunique(),"TestRuns":test.RunGroup.nunique(),
                       "TrainN":len(train),"CalibrationN":len(cal),"TestN":len(test),"BalanceScore":balance_score,"GroupOverlap":0})
    prep=build_preprocessor(train[FEATURES]); Xtr=prep.fit_transform(train[FEATURES]).astype(np.float32)
    Xca=prep.transform(cal[FEATURES]).astype(np.float32); Xte=prep.transform(test[FEATURES]).astype(np.float32)
    ytr=train.delay.to_numpy(float); yca=cal.delay.to_numpy(float); yte=test.delay.to_numpy(float)
    for model_name in CFG.models:
        selected_params,selection,tuning=select_ec_hyperparameters(train,seed,model_name)
        selection_rows.append({"Seed":seed,"Model":model_name,"Protocol":"ordinary nested development",**selection})
        tuning.insert(0,"Model",model_name); tuning.insert(0,"Seed",seed); tuning.insert(2,"Protocol","ordinary nested development"); tuning_rows.append(tuning)
        start=time.perf_counter(); model=build_model(model_name,seed).fit(Xtr,ytr); train_seconds=time.perf_counter()-start
        pca=model.predict(Xca); start=time.perf_counter(); pte=model.predict(Xte); predict_seconds=time.perf_counter()-start
        prediction_rows.append({"Seed":seed,"Model":model_name,"MAE":mean_absolute_error(yte,pte),"RMSE":mean_squared_error(yte,pte)**.5,"R2":r2_score(yte,pte),"N":len(yte)})
        rng=np.random.default_rng(seed); ref_idx=run_balanced_reference_indices(train.RunGroup.to_numpy(),CFG.max_explanation_reference_rows,seed+100)
        background=Xtr[rng.choice(len(Xtr),min(64,len(Xtr)),replace=False)]
        start=time.perf_counter(); phi_tr=explain_strength(model_name,model,Xtr[ref_idx],background); ref_seconds=time.perf_counter()-start
        start=time.perf_counter(); phi_ca=explain_strength(model_name,model,Xca,background); phi_te=explain_strength(model_name,model,Xte,background); explain_seconds=time.perf_counter()-start
        ecdf=EmpiricalCDF().fit(phi_tr); ztr,zca,zte=ecdf.transform(phi_tr),ecdf.transform(phi_ca),ecdf.transform(phi_te)
        residual=np.abs(yca-pca); q=conformal_quantile(residual); intervals={"Standard CP":(pte-q,pte+q)}
        scale,oof_residual=oof_scale(Xtr,ytr,train.RunGroup.to_numpy(),seed,model_name); sca=np.maximum(scale.predict(Xca),1e-6); ste=np.maximum(scale.predict(Xte),1e-6)
        qn=conformal_quantile(residual/sca); intervals["OOF Normalized CP"]=(pte-qn*ste,pte+qn*ste)
        if model_name=="Random Forest": intervals["CQR"]=cqr(Xtr,ytr,Xca,yca,Xte,seed)
        ec_scale_params=fit_explanation_scale(ztr,oof_residual[ref_idx],train.RunGroup.to_numpy()[ref_idx],
            selected_params["k"],selected_params["shrinkage"],selected_params["cap"])
        sec=apply_explanation_scale(zca,ec_scale_params); setest=apply_explanation_scale(zte,ec_scale_params)
        qe=conformal_quantile(residual/sec)
        # This is one prespecified validation-gated policy. A rejected EC
        # candidate returns Standard CP, but it remains the same evaluated
        # method and must not be split into post-selection subgroups.
        ec_name="Validation-gated EC-NCS"
        intervals[ec_name]=(pte-qe*setest,pte+qe*setest)
        null_rng=np.random.default_rng(seed+9000); nztr=null_rng.random(len(ztr)); nca=null_rng.random(len(zca)); nte=null_rng.random(len(zte))
        null_params=fit_explanation_scale(nztr,oof_residual[ref_idx],train.RunGroup.to_numpy()[ref_idx],
            selected_params["k"],selected_params["shrinkage"],selected_params["cap"])
        snc=apply_explanation_scale(nca,null_params); snt=apply_explanation_scale(nte,null_params)
        qnull=conformal_quantile(residual/snc); intervals["Null-signal Selected"]=(pte-qnull*snt,pte+qnull*snt)
        rows,groups=evaluate_intervals(intervals,yte,train,test,seed,model_name,"ordinary unseen runs"); overall_rows+=rows; group_rows+=groups
        if model_name=="Random Forest":
            # Illustrative only: fixed from the training labels, never called a
            # formal vehicle-control deadline or 5G SLA.
            threshold=float(np.quantile(ytr,.90)); violations=yte>threshold
            for method,(lo,hi) in {"Point prediction":(pte,pte),**intervals}.items():
                decision_value=pte if method=="Point prediction" else hi
                accepted=decision_value<=threshold
                operational_rows.append({"Seed":seed,"Model":model_name,"Method":method,"Threshold":threshold,
                    "ThresholdSource":"training-derived illustrative 90th percentile",
                    "AcceptanceRate":float(accepted.mean()),
                    "FalseSafeRateAmongAccepted":float((accepted&violations).sum()/max(1,accepted.sum())),
                    "ViolationsCaught":float(((~accepted)&violations).sum()/max(1,violations.sum())),"AcceptedN":int(accepted.sum())})

            # Mechanism diagnostics apply only when the training-only gate
            # actually activates explanation conditioning. Fallback seeds are
            # part of the primary policy but are not meaningful degradation or
            # K-sensitivity experiments.
            if selection["SelectedEC"]:
                rank_ca=np.random.default_rng(seed+5000).random(len(zca)); rank_te=np.random.default_rng(seed+5001).random(len(zte))
                standard_width=float(np.mean(intervals["Standard CP"][1]-intervals["Standard CP"][0]))
                for level in [0,.25,.50,.75,1.0]:
                    dc,dt=sec.copy(),setest.copy(); constant=float(np.median(sec))
                    if level>0: dc[rank_ca<level]=constant; dt[rank_te<level]=constant
                    if level==1: dc.fill(constant); dt.fill(constant)
                    dq=conformal_quantile(residual/dc); dr=dq*dt
                    if level==1 and not np.allclose(dr,q,atol=1e-10,rtol=1e-10): raise AssertionError("Full degradation must equal Standard CP")
                    dm=interval_metrics(yte,pte-dr,pte+dr)
                    degradation_rows.append({"Seed":seed,"Model":model_name,"Degradation":level,
                        "DiagnosticPopulation":"training-gate activated seeds only",
                        "WidthRatioToStandard":dm["MeanWidth"]/standard_width,"ExactStandardReversion":bool(level==1),**dm})
                for k in [2,3,4,5,6]:
                    kp=fit_explanation_scale(ztr,oof_residual[ref_idx],train.RunGroup.to_numpy()[ref_idx],k,
                        selected_params["shrinkage"],selected_params["cap"])
                    kca,kte=apply_explanation_scale(zca,kp),apply_explanation_scale(zte,kp)
                    kq=conformal_quantile(residual/kca); kr=kq*kte
                    ablation_rows.append({"Seed":seed,"Model":model_name,"Ablation":"explanation_groups","Value":k,
                        "DiagnosticPopulation":"training-gate activated seeds only",**interval_metrics(yte,pte-kr,pte+kr)})
        latency_rows.append({"Seed":seed,"Model":model_name,"TrainSeconds":train_seconds,"PredictionMsPerSample":1000*predict_seconds/len(Xte),
                             "ExplanationReferenceSeconds":ref_seconds,"ExplanationTestAndCalibrationSeconds":explain_seconds,
                             "ExplanationMsPerEvaluatedSample":1000*explain_seconds/(len(Xca)+len(Xte)),
                             "EndToEndTestMsPerSample":1000*(predict_seconds+explain_seconds*len(Xte)/(len(Xca)+len(Xte)))/len(Xte)})
        if CFG.run_w2s_stress and len(w2s):
            Xs=prep.transform(w2s[FEATURES]).astype(np.float32); ys=w2s.delay.to_numpy(float); ps=model.predict(Xs); phis=explain_strength(model_name,model,Xs,background); zs=ecdf.transform(phis)
            stress_intervals={"Standard CP":(ps-q,ps+q),"OOF Normalized CP":(ps-qn*np.maximum(scale.predict(Xs),1e-6),ps+qn*np.maximum(scale.predict(Xs),1e-6))}
            ss=apply_explanation_scale(zs,ec_scale_params); stress_intervals[ec_name]=(ps-qe*ss,ps+qe*ss)
            rows,_=evaluate_intervals(stress_intervals,ys,train,w2s,seed,model_name,"W2S shift stress test"); stress_rows+=rows
            if seed==CFG.seeds[0] and model_name=="Random Forest" and W2S_SIGNAL_FIELD is not None:
                for method,(lo,hi) in stress_intervals.items():
                    sample_rows.append(pd.DataFrame({"RunGroup":w2s.RunGroup,"RowInRun":w2s.RowInRun,"Method":method,"Y":ys,"Prediction":ps,
                        "Lower":lo,"Upper":hi,"Width":hi-lo,
                        "SignalField":W2S_SIGNAL_FIELD,
                        "SignalValue":pd.to_numeric(w2s.get(W2S_SIGNAL_FIELD),errors="coerce"),
                        "ExplanationPercentile":zs}))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Preserve expensive completed seeds even if a later stage is interrupted.
    for checkpoint_name, checkpoint_rows in {
        "checkpoint_overall_metrics.csv":overall_rows,
        "checkpoint_physical_group_metrics.csv":group_rows,
        "checkpoint_prediction_metrics.csv":prediction_rows,
        "checkpoint_split_audit.csv":split_rows,
        "checkpoint_latency.csv":latency_rows,
        "checkpoint_w2s_stress_metrics.csv":stress_rows,
        "checkpoint_operational_diagnostic.csv":operational_rows,
        "checkpoint_explanation_degradation.csv":degradation_rows,
        "checkpoint_hyperparameter_ablations.csv":ablation_rows,
        "checkpoint_selection_audit.csv":selection_rows,
    }.items():
        save_partial_csv(checkpoint_name,checkpoint_rows)
    if tuning_rows:
        pd.concat(tuning_rows,ignore_index=True).to_csv(
            OUTDIR/"checkpoint_nested_tuning_audit.csv",index=False)

# %% Reserved transmission-frequency confirmation
# Five fixed outer folds make every complete run a test unit exactly once. The
# paired confirmatory tests therefore operate on independent run-level metrics,
# not on overlapping random-seed test sets (which would be pseudoreplication).
confirm_rows=[]; confirm_run_rows=[]; confirm_split_rows=[]
if frequency_confirm.RunGroup.nunique() >= 2*CFG.confirmation_folds:
    group_table,outer_fold_balance_score=make_balanced_confirmation_folds(
        frequency_confirm,CFG.confirmation_folds,2027)
    group_table.to_csv(OUTDIR/"reserved_frequency_fold_assignment.csv",index=False)
    print("Reserved fold run counts:",group_table.Fold.value_counts().sort_index().to_dict())
    for fold in range(CFG.confirmation_folds):
        seed=2027+fold
        print(f"\n===== RESERVED FREQUENCY CONFIRMATION fold {fold+1}/{CFG.confirmation_folds} =====")
        set_seed(seed)
        test_groups=set(group_table.loc[group_table.Fold==fold,"RunGroup"])
        remainder=frequency_confirm[~frequency_confirm.RunGroup.isin(test_groups)].copy()
        train_idx,cal_idx,inner_assignment,balance_score=train_calibration_group_split(remainder,seed)
        train=remainder.iloc[train_idx].copy()
        cal=remainder.iloc[cal_idx].copy()
        test=frequency_confirm[frequency_confirm.RunGroup.isin(test_groups)].copy()
        assert_nonempty_group_split(train,cal,test,f"reserved confirmation fold {fold+1}")
        overlap=(set(train.RunGroup)&set(cal.RunGroup)) | (set(train.RunGroup)&set(test.RunGroup)) | (set(cal.RunGroup)&set(test.RunGroup))
        confirm_split_rows.append({"Fold":fold,"TrainRuns":train.RunGroup.nunique(),"CalibrationRuns":cal.RunGroup.nunique(),
            "TestRuns":test.RunGroup.nunique(),"TrainN":len(train),"CalibrationN":len(cal),"TestN":len(test),
            "OuterFoldBalanceScore":outer_fold_balance_score,
            "BalanceScore":balance_score,"GroupOverlap":len(overlap)})
        inner_assignment.assign(OuterFold=fold).to_csv(
            OUTDIR/f"reserved_frequency_inner_assignment_fold_{fold+1}.csv",index=False)
        prep=build_preprocessor(train[FEATURES]); Xtr=prep.fit_transform(train[FEATURES]).astype(np.float32)
        Xca=prep.transform(cal[FEATURES]).astype(np.float32); Xte=prep.transform(test[FEATURES]).astype(np.float32)
        ytr=train.delay.to_numpy(float); yca=cal.delay.to_numpy(float); yte=test.delay.to_numpy(float)
        for model_name in CFG.models:
            selected_params,selection,tuning=select_ec_hyperparameters(train,seed,model_name)
            selection_rows.append({"Seed":seed,"Model":model_name,"Protocol":"reserved frequency confirmation",**selection})
            tuning.insert(0,"Model",model_name); tuning.insert(0,"Seed",seed); tuning.insert(2,"Protocol","reserved frequency confirmation"); tuning_rows.append(tuning)
            model=build_model(model_name,seed).fit(Xtr,ytr); pca=model.predict(Xca); pte=model.predict(Xte)
            rng=np.random.default_rng(seed); ref_idx=run_balanced_reference_indices(train.RunGroup.to_numpy(),CFG.max_explanation_reference_rows,seed+100)
            background=Xtr[rng.choice(len(Xtr),min(64,len(Xtr)),replace=False)]
            phi_tr=explain_strength(model_name,model,Xtr[ref_idx],background); phi_ca=explain_strength(model_name,model,Xca,background); phi_te=explain_strength(model_name,model,Xte,background)
            ecdf=EmpiricalCDF().fit(phi_tr); ztr,zca,zte=ecdf.transform(phi_tr),ecdf.transform(phi_ca),ecdf.transform(phi_te)
            residual=np.abs(yca-pca); q=conformal_quantile(residual); intervals={"Standard CP":(pte-q,pte+q)}
            scale,oof_residual=oof_scale(Xtr,ytr,train.RunGroup.to_numpy(),seed,model_name)
            sca=np.maximum(scale.predict(Xca),1e-6); ste=np.maximum(scale.predict(Xte),1e-6); qn=conformal_quantile(residual/sca)
            intervals["OOF Normalized CP"]=(pte-qn*ste,pte+qn*ste)
            if model_name=="Random Forest": intervals["CQR"]=cqr(Xtr,ytr,Xca,yca,Xte,seed)
            params=fit_explanation_scale(ztr,oof_residual[ref_idx],train.RunGroup.to_numpy()[ref_idx],selected_params["k"],selected_params["shrinkage"],selected_params["cap"])
            sec,setest=apply_explanation_scale(zca,params),apply_explanation_scale(zte,params); qe=conformal_quantile(residual/sec)
            ec_name="Validation-gated EC-NCS"
            intervals[ec_name]=(pte-qe*setest,pte+qe*setest)
            rows,_=evaluate_intervals(intervals,yte,train,test,fold,model_name,"reserved frequency confirmation")
            confirm_rows+=rows
            for run_group,run_indices in test.groupby("RunGroup").groups.items():
                pos=test.index.get_indexer(list(run_indices)); pos=pos[pos>=0]
                for method,(lo,hi) in intervals.items():
                    confirm_run_rows.append({"Seed":run_group,"Fold":fold,"Model":model_name,
                        "Protocol":"reserved frequency confirmation","Method":method,
                        **interval_metrics(yte[pos],lo[pos],hi[pos])})
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        save_partial_csv("checkpoint_reserved_frequency_fold_metrics.csv",confirm_rows)
        save_partial_csv("checkpoint_reserved_frequency_run_metrics.csv",confirm_run_rows)
        save_partial_csv("checkpoint_reserved_frequency_split_audit.csv",confirm_split_rows)
        save_partial_csv("checkpoint_selection_audit.csv",selection_rows)
        if tuning_rows:
            pd.concat(tuning_rows,ignore_index=True).to_csv(
                OUTDIR/"checkpoint_nested_tuning_audit.csv",index=False)
else:
    print("WARNING: insufficient complete frequency-stress runs for reserved confirmation.")

overall=pd.DataFrame(overall_rows); by_group=pd.DataFrame(group_rows); predictions=pd.DataFrame(prediction_rows)
splits=pd.DataFrame(split_rows); latency=pd.DataFrame(latency_rows); stress=pd.DataFrame(stress_rows)
operational=pd.DataFrame(operational_rows); degradation=pd.DataFrame(degradation_rows); ablations=pd.DataFrame(ablation_rows)
confirmation=pd.DataFrame(confirm_rows); confirmation_by_run=pd.DataFrame(confirm_run_rows); confirmation_splits=pd.DataFrame(confirm_split_rows)
tuning_audit=pd.concat(tuning_rows,ignore_index=True) if tuning_rows else pd.DataFrame()
selection_audit=pd.DataFrame(selection_rows)
if sample_rows: stress_samples=pd.concat(sample_rows,ignore_index=True)
else: stress_samples=pd.DataFrame()
# The validation gate is one method. These assertions prevent accidental
# conditioning on whether EC was activated when forming primary estimates.
for model_name in CFG.models:
    gate_rows=overall[(overall.Model==model_name)&(overall.Method=="Validation-gated EC-NCS")]
    if gate_rows.Seed.nunique()!=len(CFG.seeds):
        raise AssertionError(f"{model_name}: validation-gated policy does not contain all ordinary seeds")
if len(confirmation_by_run):
    expected_runs=frequency_confirm.RunGroup.nunique()
    for model_name in CFG.models:
        for method in ("Standard CP","Validation-gated EC-NCS"):
            block=confirmation_by_run[(confirmation_by_run.Model==model_name)&
                                      (confirmation_by_run.Method==method)]
            if block.Seed.nunique()!=expected_runs or len(block)!=expected_runs:
                raise AssertionError(f"{model_name}/{method}: expected {expected_runs} paired runs, found {len(block)}")
ci=bootstrap_ci(overall,["Protocol","Model","Method"],["Coverage","CoverageError","MeanWidth","Winkler"])
tests=paired_tests(pd.concat([overall,stress],ignore_index=True) if len(stress) else overall)
confirmation_ci=(bootstrap_ci(confirmation_by_run,["Protocol","Model","Method"],["Coverage","CoverageError","MeanWidth","Winkler"])
                 if len(confirmation_by_run) else pd.DataFrame())
# Prespecified confirmatory family: EC-NCS versus Standard CP, coverage error
# and Winkler only. Holm correction is intentionally confined to this locked
# family rather than diluted by exploratory baselines and plots.
if len(confirmation_by_run):
    confirmation_tests=paired_tests(confirmation_by_run[confirmation_by_run.Method.isin(
        ["Standard CP","Validation-gated EC-NCS"])],
        metrics=("CoverageError","Winkler"))
else: confirmation_tests=pd.DataFrame()

# Worst physical-group diagnostics calculated seed-by-seed, without using them for method selection.
worst=[]
for key,block in by_group.groupby(["Seed","Model","Protocol","Method","GroupType"]):
    worst.append({**dict(zip(["Seed","Model","Protocol","Method","GroupType"],key)),
                  "WorstGroupCoverage":block.Coverage.min(),"WorstGroupCoverageError":np.max(np.abs(block.Coverage-CFG.nominal_coverage)),
                  "MeanGroupWinkler":block.Winkler.mean()})
worst=pd.DataFrame(worst); worst_ci=bootstrap_ci(worst,["Model","Protocol","Method","GroupType"],["WorstGroupCoverage","WorstGroupCoverageError","MeanGroupWinkler"])
worst_tests=paired_tests(worst,metrics=("WorstGroupCoverageError","MeanGroupWinkler"),
                         group_columns=("Protocol","Model","GroupType"))
degradation_ci=(bootstrap_ci(degradation,["Model","Degradation"],["Coverage","MeanWidth","WidthRatioToStandard","Winkler"])
                if len(degradation) else pd.DataFrame())
operational_ci=bootstrap_ci(operational,["Model","Method"],["AcceptanceRate","FalseSafeRateAmongAccepted","ViolationsCaught"])

for name,frame in {"overall_metrics.csv":overall,"physical_group_metrics.csv":by_group,"worst_group_metrics.csv":worst,
                   "prediction_metrics.csv":predictions,"split_audit.csv":splits,"latency.csv":latency,
                   "w2s_stress_metrics.csv":stress,"confidence_intervals.csv":ci,"worst_group_confidence_intervals.csv":worst_ci,
                   "paired_tests_holm.csv":tests,"worst_group_paired_tests_holm.csv":worst_tests,
                   "operational_diagnostic.csv":operational,"operational_diagnostic_ci.csv":operational_ci,
                   "explanation_degradation.csv":degradation,"explanation_degradation_ci.csv":degradation_ci,
                   "hyperparameter_ablations.csv":ablations,"file_audit.csv":file_audit,
                   "physical_field_audit.csv":physical_field_audit,
                   "run_physical_field_audit.csv":run_physical_audit,
                   "nested_tuning_audit.csv":tuning_audit,"selection_audit.csv":selection_audit,
                   "reserved_frequency_fold_metrics.csv":confirmation,"reserved_frequency_run_metrics.csv":confirmation_by_run,
                   "reserved_frequency_split_audit.csv":confirmation_splits,
                   "reserved_frequency_confidence_intervals.csv":confirmation_ci,
                   "reserved_frequency_paired_tests_holm.csv":confirmation_tests}.items(): frame.to_csv(OUTDIR/name,index=False)
overall[overall.Method.isin(["Standard CP","OOF Normalized CP","CQR","Validation-gated EC-NCS"])].to_csv(
    OUTDIR/"primary_overall_metrics.csv",index=False)
confirmation_by_run[confirmation_by_run.Method.isin(
    ["Standard CP","OOF Normalized CP","CQR","Validation-gated EC-NCS"])].to_csv(
    OUTDIR/"primary_reserved_run_metrics.csv",index=False)
if CFG.save_sample_predictions and len(stress_samples): stress_samples.to_parquet(OUTDIR/"supplementary/w2s_sample_predictions.parquet",index=False)

# %% Figures
palette={"Standard CP":"#4C78A8","OOF Normalized CP":"#F58518","CQR":"#54A24B",
         "Validation-gated EC-NCS":"#72B7B2",
         "Null-signal Selected":"#B279A2"}
primary_methods=["Standard CP","OOF Normalized CP","CQR","Validation-gated EC-NCS"]
rf=overall[(overall.Model=="Random Forest") & (overall.Method.isin(primary_methods))]
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
sns.pointplot(data=rf,x="Method",y="Coverage",errorbar=("ci",95),palette=palette,ax=axes[0]); axes[0].axhline(CFG.nominal_coverage,color="black",ls="--"); axes[0].tick_params(axis="x",rotation=30); axes[0].set_title("Unseen-run marginal coverage")
sns.pointplot(data=rf,x="Method",y="Winkler",errorbar=("ci",95),palette=palette,ax=axes[1]); axes[1].tick_params(axis="x",rotation=30); axes[1].set_title("Interval quality (lower is better)")
fig.suptitle("CICV5G real-network validation: Random Forest, ten seeds"); fig.tight_layout(); fig.savefig(OUTDIR/"figures/fig_cicv5g_overall.pdf",bbox_inches="tight"); fig.savefig(OUTDIR/"figures/fig_cicv5g_overall.png",dpi=300,bbox_inches="tight"); plt.show()

rf_worst=worst[(worst.Model=="Random Forest") &
               (worst.GroupType==f"{PRIMARY_SIGNAL_FIELD}-tertile") &
               (worst.Method.isin(primary_methods))]
plt.figure(figsize=(8,4.5)); sns.pointplot(data=rf_worst,x="Method",y="WorstGroupCoverageError",errorbar=("ci",95),palette=palette)
plt.xticks(rotation=30); plt.ylabel(f"Worst {PRIMARY_SIGNAL_FIELD}-tertile coverage error"); plt.title("Physical signal-quality robustness (ten unseen-run splits)"); plt.tight_layout(); plt.savefig(OUTDIR/"figures/fig_cicv5g_worst_signal.pdf",bbox_inches="tight"); plt.savefig(OUTDIR/"figures/fig_cicv5g_worst_signal.png",dpi=300,bbox_inches="tight"); plt.show()

if len(stress):
    rs=stress[(stress.Model=="Random Forest") & (stress.Method.isin(primary_methods))]
    plt.figure(figsize=(8,4.5)); sns.pointplot(data=rs,x="Method",y="Coverage",errorbar=("ci",95),palette=palette)
    plt.axhline(CFG.nominal_coverage,color="black",ls="--"); plt.xticks(rotation=25); plt.ylabel("Empirical coverage")
    plt.title("W2S strong-to-weak shift stress test\n(no conformal guarantee under distribution shift)"); plt.tight_layout(); plt.savefig(OUTDIR/"figures/fig_w2s_shift.pdf",bbox_inches="tight"); plt.savefig(OUTDIR/"figures/fig_w2s_shift.png",dpi=300,bbox_inches="tight"); plt.show()

if len(confirmation):
    rc=confirmation_by_run[(confirmation_by_run.Model=="Random Forest") &
                           (confirmation_by_run.Method.isin(primary_methods))]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    sns.pointplot(data=rc,x="Method",y="Coverage",errorbar=("ci",95),palette=palette,ax=axes[0]); axes[0].axhline(CFG.nominal_coverage,color="black",ls="--"); axes[0].tick_params(axis="x",rotation=28); axes[0].set_title("Reserved-run coverage")
    sns.pointplot(data=rc,x="Method",y="Winkler",errorbar=("ci",95),palette=palette,ax=axes[1]); axes[1].tick_params(axis="x",rotation=28); axes[1].set_title("Reserved-run Winkler")
    fig.suptitle("Locked transmission-frequency confirmation (20 independent run units)"); fig.tight_layout()
    fig.savefig(OUTDIR/"figures/fig_reserved_frequency_confirmation.pdf",bbox_inches="tight"); fig.savefig(OUTDIR/"figures/fig_reserved_frequency_confirmation.png",dpi=300,bbox_inches="tight"); plt.show()

if len(degradation):
    plt.figure(figsize=(7,4.2)); sns.lineplot(data=degradation,x="Degradation",y="WidthRatioToStandard",marker="o",errorbar=("ci",95))
    plt.axhline(1,color="black",ls="--"); plt.xlabel("Fraction of explanation values replaced by a constant"); plt.ylabel("Mean width / Standard CP width")
    n_active=degradation.Seed.nunique(); plt.title(f"Explanation-mechanism degradation (RF; {n_active} gate-activated seeds)")
    plt.tight_layout(); plt.savefig(OUTDIR/"supplementary/fig_cicv5g_degradation.pdf",bbox_inches="tight"); plt.savefig(OUTDIR/"supplementary/fig_cicv5g_degradation.png",dpi=300,bbox_inches="tight"); plt.show()

if len(ablations):
    plt.figure(figsize=(7,4.2)); sns.lineplot(data=ablations,x="Value",y="Winkler",marker="o",errorbar=("ci",95))
    plt.xlabel("Number of explanation strata K"); plt.ylabel("Winkler score"); plt.title(f"EC-NCS K sensitivity ({ablations.Seed.nunique()} gate-activated seeds)")
    plt.tight_layout(); plt.savefig(OUTDIR/"supplementary/fig_stratum_ablation.pdf",bbox_inches="tight"); plt.savefig(OUTDIR/"supplementary/fig_stratum_ablation.png",dpi=300,bbox_inches="tight"); plt.show()

if len(stress_samples):
    selected_run=stress_samples.RunGroup.value_counts().index[0]
    trace=stress_samples[stress_samples.RunGroup==selected_run].copy(); maximum=max(1,trace.RowInRun.max())
    trace["ProgressBin"]=pd.cut(trace.RowInRun/maximum,bins=np.linspace(0,1,21),include_lowest=True,labels=False)
    trace_summary=trace.groupby(["Method","ProgressBin"],as_index=False).agg(Width=("Width","mean"),SignalValue=("SignalValue","mean"))
    methods=[m for m in ["Standard CP","Validation-gated EC-NCS"] if m in set(trace_summary.Method)]
    fig,axis=plt.subplots(figsize=(8,4.4))
    sns.lineplot(data=trace_summary[trace_summary.Method.isin(methods)],x="ProgressBin",y="Width",hue="Method",marker="o",ax=axis)
    axis.set_xlabel("Normalized W2S trace progress (20 bins)"); axis.set_ylabel("Mean interval width")
    second=axis.twinx(); signal_trace=trace_summary.groupby("ProgressBin",as_index=False).SignalValue.mean(); second.plot(signal_trace.ProgressBin,signal_trace.SignalValue,color="gray",ls="--",alpha=.65); second.set_ylabel(f"Mean {W2S_SIGNAL_FIELD} (source units)")
    axis.set_title("Example W2S transition: uncertainty allocation and signal quality")
    fig.tight_layout(); fig.savefig(OUTDIR/"supplementary/fig_w2s_transition_example.pdf",bbox_inches="tight"); fig.savefig(OUTDIR/"supplementary/fig_w2s_transition_example.png",dpi=300,bbox_inches="tight"); plt.show()

# %% Tables, integrity report, and archive
summary=(overall.groupby(["Model","Method"],as_index=False).agg(Coverage=("Coverage","mean"),CoverageSD=("Coverage","std"),MeanWidth=("MeanWidth","mean"),Winkler=("Winkler","mean")))
summary.to_excel(OUTDIR/"table_cicv5g_overall.xlsx",index=False); by_group.to_excel(OUTDIR/"table_cicv5g_physical_groups.xlsx",index=False)
worst.to_excel(OUTDIR/"table_cicv5g_worst_groups.xlsx",index=False); tests.to_excel(OUTDIR/"table_cicv5g_paired_tests.xlsx",index=False)
worst_tests.to_excel(OUTDIR/"table_cicv5g_worst_group_tests.xlsx",index=False)
latency.to_excel(OUTDIR/"table_cicv5g_latency.xlsx",index=False); summary.to_latex(OUTDIR/"table_cicv5g_overall.tex",index=False,float_format="%.3f")
if len(confirmation): confirmation.to_excel(OUTDIR/"table_reserved_frequency_fold_metrics.xlsx",index=False)
if len(confirmation_tests): confirmation_tests.to_excel(OUTDIR/"table_reserved_frequency_confirmatory_tests.xlsx",index=False)

expected={(seed,model) for seed in CFG.seeds for model in CFG.models}; observed=set(zip(overall.Seed,overall.Model))
expected_confirmation={(fold,model) for fold in range(CFG.confirmation_folds) for model in CFG.models}
observed_confirmation=(set(zip(confirmation.Seed,confirmation.Model)) if len(confirmation) else set())
integrity={"script_revision":SCRIPT_REVISION,"dataset_doi":"10.5281/zenodo.17475688","source_repository":"https://github.com/zxr805/CICV5G",
           "rows_loaded":len(all_data),"ordinary_rows":len(ordinary),"w2s_rows":len(w2s),
           "reserved_frequency_rows":len(frequency_confirm),"ordinary_runs":ordinary.RunGroup.nunique(),
           "w2s_runs":w2s.RunGroup.nunique(),"reserved_frequency_runs":frequency_confirm.RunGroup.nunique(),
           "features":FEATURES,"target":"measured V2N2V RTT delay","seeds":list(CFG.seeds),"all_seed_model_runs_complete":observed==expected,
           "group_overlap_max":int(splits.GroupOverlap.max()),"development_protocol":"ordinary nested run-disjoint evaluation",
           "confirmatory_protocol":"five balanced outer folds; every reserved frequency run tested exactly once",
           "all_confirmation_fold_model_runs_complete":observed_confirmation==expected_confirmation,
           "confirmation_group_overlap_max":int(confirmation_splits.GroupOverlap.max()) if len(confirmation_splits) else None,
           "validation_gated_method_label":"Validation-gated EC-NCS",
           "ordinary_gate_activation":json.loads(selection_audit[selection_audit.Protocol=="ordinary nested development"].groupby("Model").SelectedEC.agg(["sum","count"]).to_json()),
           "physical_signal_primary":PRIMARY_SIGNAL_FIELD,"w2s_trace_signal":W2S_SIGNAL_FIELD,
           "physical_schema_audit_passed":bool(len(VALID_SIGNAL_FIELDS)>0),
           "confirmatory_test_unit":"complete measurement run","w2s_status":"distribution-shift stress test; no conformal guarantee",
           "test_labels_used_for_selection":False,"ec_scale_fixed_before_outer_calibration":True,
           "confirmatory_significant_after_holm":json.loads(confirmation_tests[confirmation_tests.HolmAdjustedP<.05].to_json(orient="records")) if len(confirmation_tests) else []}
(OUTDIR/"integrity_and_claims.json").write_text(json.dumps(integrity,indent=2))
(OUTDIR/"README_RESULTS.txt").write_text(
    "Ordinary-run estimates use a nested run-disjoint evaluation protocol.\n"
    "Confirmation uses reserved transmission-frequency runs in five balanced folds; every run is tested once.\n"
    "Validation-gated EC-NCS is one policy across activated and fallback splits; primary tests never condition on activation.\n"
    "Every confirmatory Standard-vs-gated comparison contains all reserved complete-run units.\n"
    "Physical radio fields are range-audited; implausible subset/field combinations are invalidated, never relabelled.\n"
    "EC hyperparameters and the explanation scale are learned from training only and fixed before outer calibration.\n"
    "The normalized score |Y-f(X)|/s(X) is calibrated on untouched outer calibration labels.\n"
    "W2S is a distribution-shift stress test and has no marginal conformal guarantee.\n"
    "RSRP tertiles are train-derived physical diagnostics, not labels selected from residuals.\n"
    "Only Holm-adjusted p<0.05 supports a statistical superiority statement.\n"
    "A negative result must be reported; do not tune methods on these test outputs.\n")
archive=shutil.make_archive("/content/cicv5g_ec_ncs_nested_results","zip",OUTDIR)
print("Results archive:",archive); display(summary); display(confirmation_tests)