"""One-sided (upper-bound) analysis for delay-budget admission on the reserved CICV5G runs.

Uses /home/claude/rerun/extended.pkl (frozen RF, signed OOF predictions, full attribution vectors).
All scale models are fitted on training runs only; conformal quantiles on the calibration runs only.
"""
import pickle, json, sys
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr

E = pickle.load(open('/home/claude/rerun/extended.pkl', 'rb'))
B = 31.0
LEVELS = np.round(np.arange(0.80, 0.996, 0.005), 3)  # one-sided confidence levels


def cq(scores, level):
    s = np.sort(np.asarray(scores, float)); n = len(s)
    k = int(np.ceil((n + 1) * level)) - 1
    return s[min(max(k, 0), n - 1)]


def run_w(groups):
    u, inv, c = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / c[inv]


def wq(v, q, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum()
    return v[min(np.searchsorted(c, q), len(v) - 1)]


def ec_scale(z_fit, target, groups, z_apply_list, k=5, shrink=1.0, cap=3.0, floor=0.5, q=0.5):
    w = run_w(groups); gm = wq(target, q, w)
    edges = np.linspace(0, 1, k + 1); b = np.digitize(z_fit, edges[1:-1]); r = np.ones(k)
    for j in range(k):
        m = b == j
        if m.sum() >= 100:
            r[j] = np.exp(shrink * np.log(max(wq(target[m], q, w[m]) / max(gm, 1e-8), 1e-4)))
    r = np.clip(r, floor, cap)
    return [r[np.digitize(z, edges[1:-1])] for z in z_apply_list]


def ecdf(fit):
    s = np.sort(fit)
    return lambda x: np.searchsorted(s, x, side='right') / len(s)


out = {}
for f, d in E.items():
    ytr, oof, gtr = d['ytr'], d['oof'], d['gtr']
    rtr = ytr - oof                       # signed OOF residual (positive = underestimate)
    rca = d['yca'] - d['pca']; pte = d['pte']
    strength = lambda P: np.abs(P).mean(1)
    F = ecdf(strength(d['Ptr'][d['ref_idx']]))
    ztr, zca, zte = F(strength(d['Ptr'])), F(strength(d['Pca'])), F(strength(d['Pte']))
    seed = 2027 + f
    # ---------- scales (training only)
    S = {}
    S['Standard'] = (np.ones(len(rca)), np.ones(len(pte)))
    # two-sided EC scale (abs residual, median), as in paper but fixed hyperparameters
    S['EC |r|'] = tuple(ec_scale(ztr, np.abs(rtr), gtr, [zca, zte]))
    # one-sided EC scale: upper tail of the signed residual in each explanation bin
    pos = np.maximum(rtr, 0) + 1e-6
    S['EC upper'] = tuple(ec_scale(ztr, pos, gtr, [zca, zte], q=0.9))
    # OOF normalized (features), abs target as in paper
    et = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10, random_state=seed + 200, n_jobs=-1)
    et.fit(d['Xtr'], np.abs(rtr) + 1e-6)
    S['OOF |r|'] = (np.maximum(et.predict(d['Xca']), 1e-3), np.maximum(et.predict(d['Xte']), 1e-3))
    et2 = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10, random_state=seed + 201, n_jobs=-1)
    et2.fit(d['Xtr'], pos)
    S['OOF upper'] = (np.maximum(et2.predict(d['Xca']), 1e-2), np.maximum(et2.predict(d['Xte']), 1e-2))
    # full attribution vector as input to the scale model (explanations without compression)
    et3 = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10, random_state=seed + 202, n_jobs=-1)
    et3.fit(d['Ptr'], np.abs(rtr) + 1e-6)
    S['EC-vector |r|'] = (np.maximum(et3.predict(d['Pca']), 1e-3), np.maximum(et3.predict(d['Pte']), 1e-3))
    et4 = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10, random_state=seed + 203, n_jobs=-1)
    et4.fit(d['Ptr'], pos)
    S['EC-vector upper'] = (np.maximum(et4.predict(d['Pca']), 1e-2), np.maximum(et4.predict(d['Pte']), 1e-2))
    # ---------- upper bounds per level
    U = {}
    for m, (sca, ste) in S.items():
        sc = rca / sca
        U[m] = np.array([pte + cq(sc, L) * ste for L in LEVELS])
    # Mondrian by transmission frequency (network-aware baseline); falls back to pooled if freq absent
    Um = []
    for L in LEVELS:
        u = pte + cq(rca, L)
        for fr in np.unique(d['fte']):
            m = d['fca'] == fr
            if m.sum() > 50:
                u = np.where(d['fte'] == fr, pte + cq(rca[m], L), u)
        Um.append(u)
    U['Mondrian freq'] = np.array(Um)
    # CQR upper: GBR 0.95 quantile + one-sided conformal correction
    gb = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=250, max_depth=3,
                                   learning_rate=.04, random_state=seed + 1).fit(d['Xtr'], ytr)
    uca, ute = gb.predict(d['Xca']), gb.predict(d['Xte'])
    U['CQR upper'] = np.array([ute + cq(d['yca'] - uca, L) for L in LEVELS])
    out[f] = dict(U={m: u.astype(np.float32) for m, u in U.items()}, y=d['yte'], run=d['gte'], freq=d['fte'],
                  pte=pte, zte=zte, rte=d['yte'] - pte, zca=zca, rca=rca, fca=d['fca'], gca=d['gca'],
                  ztr=ztr, rtr=rtr, gtr=gtr, ftr=d['ftr'], ptr=oof, sel=bool(d['selection']['SelectedEC']),
                  scales={m: s[1].astype(np.float32) for m, s in S.items()})
    print('fold', f, 'done', flush=True)
pickle.dump(dict(out=out, LEVELS=LEVELS, B=B), open('/home/claude/rerun/onesided.pkl', 'wb'))
