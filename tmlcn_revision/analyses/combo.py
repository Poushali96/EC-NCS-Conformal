"""Does the explanation scale add information beyond CQR / OOF?  (plug-in test)

Upper bound for base method X and scale s:  u = c_X(x) + q * s(x),  where q is the conformal
quantile of (y - c_X)/s on the calibration runs (one-sided) or of |y - c_X|/s (two-sided for c=point).
c_X: point prediction (Standard / OOF / EC) or the CQR 0.95-quantile model.
s:   1, OOF feature scale, explanation scale (training-only, binned), or their product.
"""
import pickle, numpy as np
from scipy.stats import spearmanr

B = 31.0
P = pickle.load(open('/home/claude/rerun/packets.pkl', 'rb'))
PC = pickle.load(open('/home/claude/rerun/packets_cal.pkl', 'rb'))
E = pickle.load(open('/home/claude/rerun/extended.pkl', 'rb'))
LEV = np.round(np.arange(0.80, 0.999, 0.0025), 4)
folds = sorted(P)


def cq(s, L):
    s = np.sort(s); n = len(s); k = int(np.ceil((n + 1) * L)) - 1
    return s[min(max(k, 0), n - 1)]


def run_w(g):
    u, inv, c = np.unique(g, return_inverse=True, return_counts=True); return 1.0 / c[inv]


def wq(v, q, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[min(np.searchsorted(c, q), len(v) - 1)]


def ec_scale(zf, t, g, za, k=5, shrink=1.0, cap=3.0, floor=0.5):
    w = run_w(g); gm = wq(t, .5, w); e = np.linspace(0, 1, k + 1); b = np.digitize(zf, e[1:-1]); r = np.ones(k)
    for j in range(k):
        m = b == j
        if m.sum() >= 100: r[j] = max(wq(t[m], .5, w[m]) / max(gm, 1e-8), 1e-4) ** shrink
    r = np.clip(r, floor, cap); return [r[np.digitize(z, e[1:-1])] for z in za]


def ecdf(fit):
    s = np.sort(fit); return lambda x: np.searchsorted(s, x, side='right') / len(s)


fam = {}; diag = []
for f in folds:
    d, c, x = P[f], PC[f], E[f]
    st = lambda M: np.abs(M).mean(1)
    F = ecdf(st(x['Ptr'][x['ref_idx']])); ztr, zca, zte = F(st(x['Ptr'])), F(st(x['Pca'])), F(st(x['Pte']))
    rtr = x['ytr'] - x['oof']
    # explanation scale for point-prediction residuals
    eca, ete = ec_scale(ztr, np.abs(rtr), x['gtr'], [zca, zte])
    one_c, one_t = np.ones(len(zca)), np.ones(len(zte))
    centres = {'point': (c['pca'], d['pte']), 'CQR': (c['uca'], c['ute'])}
    scales = {'1': (one_c, one_t), 'OOF': (c['sca'], c['ste']), 'EC': (eca, ete),
              'OOF*EC': (c['sca'] * eca, c['ste'] * ete)}
    for cn, (cc, ct) in centres.items():
        for sn, (sc, stt) in scales.items():
            sco = (c['ycal'] - cc) / sc  # one-sided upper scores
            fam.setdefault(f'{cn}|{sn}', []).append(np.array([ct + cq(sco, L) * stt for L in LEV]))
    # diagnostic: does the explanation scale rank test |residual| beyond the OOF scale and frequency?
    r = np.abs(d['y'] - d['pte']); band = (d['pte'] > 24) & (d['pte'] <= 31)
    diag.append(dict(f=f, rho_ec=spearmanr(ete, r)[0], rho_oof=spearmanr(c['ste'], r)[0],
                     rho_ec_band=spearmanr(ete[band], r[band])[0] if band.sum() > 50 else np.nan,
                     rho_oof_band=spearmanr(c['ste'][band], r[band])[0] if band.sum() > 50 else np.nan))
U = {m: np.concatenate(v, axis=1) for m, v in fam.items()}
y = np.concatenate([P[f]['y'] for f in folds]); run = np.concatenate([P[f]['run'] for f in folds]); late = y > B
runs = np.unique(run)


def adm_at(u, l, T):
    acc = u <= B; a = acc.mean(1); v = (acc & l).sum(1) / np.maximum(acc.sum(1), 1); ok = v <= T
    return a[ok].max() if ok.any() else 0.0


print('admitted share (%) at matched late rate among admitted, pooled 20 runs')
for m, u in U.items():
    print(f'{m:14s}', '  '.join(f'T={T}: {adm_at(u, late, T)*100:5.1f}' for T in (0.01, 0.015, 0.02, 0.03)))
print('\nleave-one-run-out at T=2% (min,max of differences, pp)')
pairs = [('CQR|EC', 'CQR|1'), ('CQR|OOF*EC', 'CQR|OOF'), ('point|OOF*EC', 'point|OOF'), ('point|EC', 'point|OOF'), ('CQR|OOF', 'CQR|1')]
for a, b in pairs:
    dd = []
    for r in runs:
        idx = np.flatnonzero(run != r); dd.append(adm_at(U[a][:, idx], late[idx], .02) - adm_at(U[b][:, idx], late[idx], .02))
    dd = np.array(dd) * 100; print(f'{a:14s} - {b:12s}: min {dd.min():+5.1f} max {dd.max():+5.1f} mean {dd.mean():+5.1f}')
for dg in diag: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dg.items()})
pickle.dump(dict(U=U, LEV=LEV, y=y, run=run), open('/home/claude/rerun/combo.pkl', 'wb'))
