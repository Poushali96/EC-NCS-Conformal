"""Deployable rule: choose the level on the calibration runs (late share among admitted <= T), apply to test."""
import pickle, numpy as np
exec(open('combo.py').read().split("fam = {}; diag = []")[0])
LEVd = LEV
res = {}; per_run = {}
names = ['point|1','point|OOF','point|EC','CQR|1','CQR|OOF','CQR|EC']
for T in (0.01, 0.02):
    Ut = {m: [] for m in names}
    for f in folds:
        d, c, x = P[f], PC[f], E[f]
        st = lambda M: np.abs(M).mean(1)
        F = ecdf(st(x['Ptr'][x['ref_idx']])); ztr, zca, zte = F(st(x['Ptr'])), F(st(x['Pca'])), F(st(x['Pte']))
        eca, ete = ec_scale(ztr, np.abs(x['ytr'] - x['oof']), x['gtr'], [zca, zte])
        cen = {'point': (c['pca'], d['pte']), 'CQR': (c['uca'], c['ute'])}
        sca = {'1': (np.ones(len(zca)), np.ones(len(zte))), 'OOF': (c['sca'], c['ste']), 'EC': (eca, ete)}
        latec = c['ycal'] > B
        for m in names:
            cn, sn = m.split('|'); (cc, ct), (sc, stt) = cen[cn], sca[sn]
            sco = (c['ycal'] - cc) / sc
            # smallest level whose calibration late share <= T (most admission), computed on calibration runs
            best = None
            for L in LEVd:
                q = cq(sco, L); uc = cc + q * sc; acc = uc <= B
                if acc.sum() and (acc & latec).sum() / acc.sum() <= T: best = q; break
            if best is None: best = np.inf
            Ut[m].append(ct + best * stt)
    y = np.concatenate([P[f]['y'] for f in folds]); run = np.concatenate([P[f]['run'] for f in folds]); late = y > B
    runs = np.unique(run); ri = {r: np.flatnonzero(run == r) for r in runs}
    print(f'--- T = {T}: level chosen on calibration runs, evaluated on the 20 reserved runs')
    A = {}
    for m in names:
        u = np.concatenate(Ut[m]); acc = u <= B; A[m] = acc
        print(f'{m:10s} admitted {acc.mean()*100:5.1f}%  late among admitted {(acc&late).sum()/max(acc.sum(),1)*100:5.2f}%')
    rng = np.random.default_rng(7)
    for a, b in [('point|EC','point|OOF'), ('point|EC','point|1'), ('CQR|EC','CQR|1'), ('CQR|EC','CQR|OOF')]:
        da, dv = [], []
        for _ in range(1000):
            idx = np.concatenate([ri[r] for r in rng.choice(runs, len(runs))])
            aa, bb, l = A[a][idx], A[b][idx], late[idx]
            da.append(aa.mean() - bb.mean()); dv.append((aa & l).sum()/max(aa.sum(),1) - (bb & l).sum()/max(bb.sum(),1))
        da, dv = np.array(da)*100, np.array(dv)*100
        print(f'   {a} - {b}: admitted {da.mean():+5.1f} [{np.quantile(da,.025):+5.1f},{np.quantile(da,.975):+5.1f}]  late {dv.mean():+5.2f} [{np.quantile(dv,.025):+5.2f},{np.quantile(dv,.975):+5.2f}]')
    res[T] = {m: (float(A[m].mean()), float((A[m]&late).sum()/max(A[m].sum(),1))) for m in names}
    per_run[T] = {m: [(float(A[m][ri[r]].mean()), int((A[m][ri[r]]&late[ri[r]]).sum())) for r in runs] for m in names}
import json; json.dump(dict(res=res, per_run=per_run), open('/home/claude/rerun/deploy.json','w'))
