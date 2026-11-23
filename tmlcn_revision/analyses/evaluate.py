import pickle, numpy as np, sys
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
D = pickle.load(open('/home/claude/rerun/onesided.pkl', 'rb')); O, L, B = D['out'], D['LEVELS'], D['B']
folds = sorted(O)
cat = lambda k: np.concatenate([O[f][k] for f in folds])
y, run, freq, pte = cat('y'), cat('run'), cat('freq'), cat('pte')
methods = list(O[0]['U'])
U = {m: np.concatenate([O[f]['U'][m] for f in folds], axis=1) for m in methods}
late = y > B
runs = np.unique(run); ri = {r: np.flatnonzero(run == r) for r in runs}


def curve(m, idx=None):
    u = U[m] if idx is None else U[m][:, idx]; l = late if idx is None else late[idx]
    acc = u <= B
    adm = acc.mean(1); v = (acc & l).sum(1) / np.maximum(acc.sum(1), 1)
    return adm, v


def adm_at(m, target, idx=None):
    adm, v = curve(m, idx)
    ok = v <= target
    return adm[ok].max() if ok.any() else 0.0


print('=== level 0.95 one-sided ===')
i95 = int(np.argmin(abs(L - 0.95)))
for m in methods:
    adm, v = curve(m)
    print(f'{m:18s} admitted {adm[i95]*100:5.1f}%  late {v[i95]*100:5.2f}%   | adm@late<=1% {adm_at(m,.01)*100:5.1f}  <=2% {adm_at(m,.02)*100:5.1f}')

print('\n=== matched-violation bootstrap (admitted share at late<=T), 400 run resamples ===')
rng = np.random.default_rng(1)
for T in (0.01, 0.02):
    bs = {m: [] for m in methods}
    for _ in range(400):
        idx = np.concatenate([ri[r] for r in rng.choice(runs, len(runs))])
        for m in methods: bs[m].append(adm_at(m, T, idx))
    bs = {m: np.array(v) for m, v in bs.items()}
    for m in methods:
        print(f'T={T} {m:18s} {np.mean(bs[m])*100:5.1f} [{np.quantile(bs[m],.025)*100:5.1f},{np.quantile(bs[m],.975)*100:5.1f}]', end='')
        for b in ('Standard', 'OOF |r|', 'CQR upper'):
            dd = bs[m] - bs[b]; print(f' | vs {b}: {np.mean(dd)*100:+5.1f} P>0={np.mean(dd>0):.2f}', end='')
        print()
pickle.dump(dict(U=U, y=y, run=run, freq=freq, pte=pte), open('/home/claude/rerun/pooled.pkl', 'wb'))
