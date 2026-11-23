"""Pre-specified checks requested in review.
(1) operating level chosen on calibration drives only (2% late target, B=31 ms), evaluated on untouched drives;
(2) same 5-bin estimator driven by predicted-delay rank or RSRP rank instead of explanation rank.
Symmetric (|y-c|/s) and one-sided ((y-c)/s) scores. Cases: 2/4/8 training drives (3 draws), no-100 Hz drives."""
import pickle, numpy as np, json
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
E=pickle.load(open('/home/claude/rerun/extended.pkl','rb')); B=31.0; T=0.02
LEV=np.round(np.arange(0.01,0.9995,0.005),4)
def run_w(g): u,i,c=np.unique(g,return_inverse=True,return_counts=True); return 1.0/c[i]
def wq(v,q,w): o=np.argsort(v); v,w=v[o],w[o]; c=np.cumsum(w)/w.sum(); return v[min(np.searchsorted(c,q),len(v)-1)]
def bin_scale(zf,t,g,za,k=5,cap=2.0,floor=.5):
    w=run_w(g); gm=wq(t,.5,w); e=np.linspace(0,1,k+1); b=np.digitize(zf,e[1:-1]); r=np.ones(k)
    for j in range(k):
        m=b==j
        if m.sum()>=100: r[j]=max(wq(t[m],.5,w[m])/max(gm,1e-8),1e-4)
    r=np.clip(r,floor,cap); r=r/max(wq(r[b],.5,w),1e-8); r=np.clip(r,floor,cap)
    return [r[np.digitize(z,e[1:-1])] for z in za]
def ecdf(fit): s=np.sort(fit); return lambda x: np.searchsorted(s,x,side='right')/len(s)
st=lambda M: np.abs(M).mean(1)
CASES=[('drives',2),('drives',4),('drives',8),('no100',0)]
rows=[]; front={}
for case,n in CASES:
    for f in sorted(E):
        d=E[f]; runs=np.unique(d['gtr']); rng=np.random.default_rng(100+f)
        for di in ([0,1,2] if case=='drives' else [0]):
            m=np.isin(d['gtr'],rng.choice(runs,n,replace=False)) if case=='drives' else d['ftr']!=100
            seed=2027+f+di; r=np.abs(d['ytr'][m]-d['oof'][m]); g=d['gtr'][m]
            sig={}
            F=ecdf(st(d['Ptr'][m])); sig['EC-NCS']=(F(st(d['Ptr'][m])),F(st(d['Pca'])),F(st(d['Pte'])))
            F=ecdf(d['oof'][m]); sig['Pred-rank bins']=(F(d['oof'][m]),F(d['pca']),F(d['pte']))
            F=ecdf(d['Xtr'][m,0]); sig['RSRP-rank bins']=(F(d['Xtr'][m,0]),F(d['Xca'][:,0]),F(d['Xte'][:,0]))
            S={'Standard CP':(np.ones(len(d['yca'])),np.ones(len(d['yte'])))}
            for k,(zf,zc,zt) in sig.items(): S[k]=tuple(bin_scale(zf,r,g,[zc,zt]))
            et=ExtraTreesRegressor(200,min_samples_leaf=10,random_state=seed,n_jobs=-1).fit(d['Xtr'][m],r+1e-6)
            S['OOF normalized CP']=(np.maximum(et.predict(d['Xca']),1e-3),np.maximum(et.predict(d['Xte']),1e-3))
            gl=GradientBoostingRegressor(loss='quantile',alpha=.05,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed).fit(d['Xtr'][m],d['ytr'][m])
            gh=GradientBoostingRegressor(loss='quantile',alpha=.95,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed+1).fit(d['Xtr'][m],d['ytr'][m])
            yc,yt,pc,pt=d['yca'],d['yte'],d['pca'],d['pte']; lc,uc,ut=gl.predict(d['Xca']),gh.predict(d['Xca']),gh.predict(d['Xte'])
            fam={}
            for k,(sc,stt) in S.items():
                fam[(k,'sym')]=(pc,sc,pt,stt,np.abs(yc-pc)/sc); fam[(k,'one')]=(pc,sc,pt,stt,(yc-pc)/sc)
            one=np.ones; fam[('CQR','sym')]=(uc,one(len(yc)),ut,one(len(yt)),np.maximum(lc-yc,yc-uc)); fam[('CQR','one')]=(uc,one(len(yc)),ut,one(len(yt)),yc-uc)
            latec,latet=yc>B,yt>B
            for key,(cc,sc,ct,stt,s) in fam.items():
                s=np.sort(s); nn=len(s); qs=s[np.minimum(np.ceil((nn+1)*LEV).astype(int)-1,nn-1)]
                Uc=cc[None,:]+qs[:,None]*sc[None,:]; Ut=ct[None,:]+qs[:,None]*stt[None,:]
                ac=Uc<=B; vc=(ac&latec).sum(1)/np.maximum(ac.sum(1),1); ok=np.flatnonzero((vc<=T)&(ac.sum(1)>0))
                i=ok[0] if len(ok) else None
                acc=Ut[i]<=B if i is not None else np.zeros(len(yt),bool)
                for rr in np.unique(d['gte']):
                    j=d['gte']==rr
                    rows.append((case,n,f,di,key[0],key[1],rr,int(d['fte'][j][0]),int(j.sum()),int(acc[j].sum()),int((acc[j]&latet[j]).sum())))
                at=Ut<=B; fk=(case,n,di)+key; c0=front.get(fk,np.zeros((3,len(LEV))))
                c0+=np.array([at.sum(1),(at&latet).sum(1),np.full(len(LEV),len(yt))]); front[fk]=c0
    print('done',case,n,flush=True)
import pandas as pd
R=pd.DataFrame(rows,columns=['case','n','fold','draw','method','score','run','freq','N','adm','late'])
R.to_csv('/home/claude/rerun/reviewcheck.csv',index=False)
def summ(sub):
    return 100*sub.adm.sum()/sub.N.sum(), 100*sub.late.sum()/max(sub.adm.sum(),1)
meths=['Standard CP','EC-NCS','Pred-rank bins','RSRP-rank bins','OOF normalized CP','CQR']
print('\n(1) CALIBRATION-CHOSEN level, target 2%: admitted% / late%   [100 Hz late%]')
for case,n in CASES:
    for sc in ('sym','one'):
        sub=R[(R.case==case)&(R.n==n)&(R.score==sc)]
        print(case,n,sc,' | '.join(f"{mm[:10]} {summ(sub[sub.method==mm])[0]:4.1f}/{summ(sub[sub.method==mm])[1]:4.2f} [{summ(sub[(sub.method==mm)&(sub.freq==100)])[1]:4.1f}]" for mm in meths))
# run bootstrap CI for EC minus each, one-sided, admitted and late
print('\nrun-bootstrap 95% CI, one-sided scores, EC-NCS minus method (admitted pp, late pp)')
rng=np.random.default_rng(3)
for case,n in CASES:
    sub=R[(R.case==case)&(R.n==n)&(R.score=='one')]
    agg=sub.groupby(['method','run'])[['N','adm','late']].sum()
    runs=sub.run.unique()
    for b in ('OOF normalized CP','CQR','Pred-rank bins','RSRP-rank bins'):
        da,dl=[],[]
        for _ in range(1000):
            pick=rng.choice(runs,len(runs))
            e=agg.loc['EC-NCS'].loc[pick].sum(); o=agg.loc[b].loc[pick].sum()
            da.append(100*(e.adm/e.N-o.adm/o.N)); dl.append(100*(e.late/max(e.adm,1)-o.late/max(o.adm,1)))
        print(case,n,'EC -',b,'adm %+.1f [%+.1f,%+.1f]  late %+.2f [%+.2f,%+.2f]'%(np.mean(da),*np.quantile(da,[.025,.975]),np.mean(dl),*np.quantile(dl,[.025,.975])))
print('\n(2) OFFLINE frontier: admitted% at 2% late (mean over draws)')
def adm_at(c): a=c[0]/c[2]; v=c[1]/np.maximum(c[0],1); ok=v<=T; return a[ok].max() if ok.any() else 0
for case,n in CASES:
    for sc in ('sym','one'):
        dr=[0,1,2] if case=='drives' else [0]
        print(case,n,sc,'  '.join(f"{mm[:10]} {np.mean([adm_at(front[(case,n,di,mm,sc)]) for di in dr])*100:5.1f}" for mm in meths))
