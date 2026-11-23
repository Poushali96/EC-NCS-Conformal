"""Robustness of the matched-violation comparison.
1) paper-exact scales (gated EC, OOF, two-sided CQR) used for an upper bound, level swept
2) leave-one-run-out
3) per-run comparison
4) hyperparameter sensitivity of the explanation scale
5) near-deadline residual diagnostic
"""
import pickle, numpy as np, json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
B = 31.0
P = pickle.load(open('/home/claude/rerun/packets.pkl', 'rb'))
PC = pickle.load(open('/home/claude/rerun/packets_cal.pkl', 'rb'))
E = pickle.load(open('/home/claude/rerun/extended.pkl', 'rb'))
O = pickle.load(open('/home/claude/rerun/onesided.pkl', 'rb'))['out']
LEV = np.round(np.arange(0.80, 0.996, 0.0025), 4)
folds = sorted(P)


def cq(s, L):
    s = np.sort(s); n = len(s); k = int(np.ceil((n + 1) * L)) - 1
    return s[min(max(k, 0), n - 1)]


def upper_family(f, kind, sca=None, ste=None):
    d, c = P[f], PC[f]; pte = d['pte']
    if kind == 'abs':  # two-sided scores |r|/s, upper end p + q s
        sc = np.abs(c['ycal'] - c['pca']) / sca
    elif kind == 'signed':
        sc = (c['ycal'] - c['pca']) / sca
    return np.array([pte + cq(sc, L) * ste for L in LEV])


def cqr_family(f, kind):
    c = P[f]; k = PC[f]
    if kind == 'abs':
        sc = np.maximum(k['lca'] - k['ycal'], k['ycal'] - k['uca'])
        return np.array([k['ute'] + cq(sc, L) for L in LEV])
    sc = k['ycal'] - k['uca']
    return np.array([k['ute'] + cq(sc, L) for L in LEV])


fam = {}
for kind in ('abs', 'signed'):
    fam[f'Standard CP ({kind})'] = [upper_family(f, kind, np.ones(len(PC[f]['ycal'])), np.ones(len(P[f]['pte']))) for f in folds]
    fam[f'OOF normalized CP ({kind})'] = [upper_family(f, kind, PC[f]['sca'], PC[f]['ste']) for f in folds]
    # gated EC: folds where gate rejected use scale 1 (exact fallback)
    fam[f'EC-NCS gated ({kind})'] = [upper_family(f, kind, PC[f]['sec'] if E[f]['selection']['SelectedEC'] else np.ones(len(PC[f]['ycal'])),
                                                  PC[f]['setest'] if E[f]['selection']['SelectedEC'] else np.ones(len(P[f]['pte']))) for f in folds]
    fam[f'CQR ({kind})'] = [cqr_family(f, kind) for f in folds]
U = {m: np.concatenate(v, axis=1) for m, v in fam.items()}
y = np.concatenate([P[f]['y'] for f in folds]); run = np.concatenate([P[f]['run'] for f in folds])
freq = np.concatenate([PC[f]['freq_test'] for f in folds]); late = y > B
runs = np.unique(run); ri = {r: np.flatnonzero(run == r) for r in runs}


def adm_at(u, l, T):
    acc = u <= B; a = acc.mean(1); v = (acc & l).sum(1) / np.maximum(acc.sum(1), 1)
    ok = v <= T
    return a[ok].max() if ok.any() else 0.0


res = {}
print('=== paper-exact methods, admitted share at matched late rate ===')
for m, u in U.items():
    res[m] = {T: adm_at(u, late, T) for T in (0.01, 0.015, 0.02, 0.03)}
    print(f'{m:30s}', '  '.join(f'T={T}: {v*100:5.1f}' for T, v in res[m].items()))

print('\n=== leave-one-run-out, T=2%, abs scores: min/max over 20 omissions ===')
keys = ['Standard CP (abs)', 'OOF normalized CP (abs)', 'CQR (abs)', 'EC-NCS gated (abs)']
loro = {m: [] for m in keys}
for r in runs:
    idx = np.flatnonzero(run != r)
    for m in keys: loro[m].append(adm_at(U[m][:, idx], late[idx], 0.02))
for m in keys:
    v = np.array(loro[m]) * 100; print(f'{m:28s} min {v.min():5.1f} max {v.max():5.1f}')
for b in ('Standard CP (abs)', 'OOF normalized CP (abs)', 'CQR (abs)'):
    dd = np.array(loro['EC-NCS gated (abs)']) - np.array(loro[b])
    print(f'EC - {b}: min {dd.min()*100:+.1f} max {dd.max()*100:+.1f} pp')

print('\n=== by frequency at matched late rate (T = 2% within the frequency group) ===')
for fr in (10, 33, 100):
    idx = np.flatnonzero(freq == fr)
    print(fr, '  '.join(f'{m.split(" (")[0]}: {adm_at(U[m][:, idx], late[idx], 0.02)*100:5.1f}' for m in keys),
          f' late base {late[idx].mean()*100:.1f}%')
for fr in (100,):
    idx = np.flatnonzero(freq == fr)
    for T in (0.03, 0.05):
        print(fr, f'T={T}', '  '.join(f'{m.split(" (")[0]}: {adm_at(U[m][:, idx], late[idx], T)*100:5.1f}' for m in keys))

# folds whose calibration lacks 100 Hz: 0 and 3
print('\n=== 100 Hz test runs whose fold has no 100 Hz in calibration (folds 0,3) ===')
fold_of = np.concatenate([[f] * len(P[f]['y']) for f in folds])
for fs in ((0, 3), (3,), (1, 2, 4)):
    idx = np.flatnonzero(np.isin(fold_of, fs) & (freq == 100))
    print(fs, 'n', len(idx), 'late base %.1f%%' % (late[idx].mean() * 100))
    i95 = int(np.argmin(abs(LEV - 0.95)))
    for m in keys:
        acc = U[m][i95, idx] <= B
        print(f'   {m:28s} @0.95 admitted {acc.mean()*100:5.1f}% late among adm {((acc&late[idx]).sum()/max(acc.sum(),1))*100:5.1f}%'
              f'   adm@T=5% {adm_at(U[m][:, idx], late[idx], .05)*100:5.1f}')
pickle.dump(dict(U=U, LEV=LEV, y=y, run=run, freq=freq, fold=fold_of), open('/home/claude/rerun/paper_families.pkl', 'wb'))
