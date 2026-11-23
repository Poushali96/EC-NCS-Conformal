"""Pre-specified test: few drives to learn the error scale. Frozen RF; scale/quantile models learned from n training drives.
Fixed settings (not tuned): EC k=5, shrinkage 1, cap 2, floor 0.5 (median |OOF residual| per explanation bin);
OOF: ExtraTrees(200, leaf 10) on features -> |OOF residual|; CQR: GBR quantiles 0.05/0.95 (250, depth 3, lr .04).
Calibration runs and conformal step unchanged. Metrics on the 20 reserved runs: coverage, Winkler, admitted at 2% late (31 ms)."""
import pickle, numpy as np, json, sys
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
E=pickle.load(open('/home/claude/rerun/extended.pkl','rb')); A=0.1; B=31.0
LEV=np.round(np.arange(0.01,0.9995,0.005),4)
def cq(s,L):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*L))-1; return s[min(max(k,0),n-1)]
def run_w(g): u,i,c=np.unique(g,return_inverse=True,return_counts=True); return 1.0/c[i]
def wq(v,q,w): o=np.argsort(v); v,w=v[o],w[o]; c=np.cumsum(w)/w.sum(); return v[min(np.searchsorted(c,q),len(v)-1)]
def ec_scale(zf,t,g,za,k=5,cap=2.0,floor=.5):
    w=run_w(g); gm=wq(t,.5,w); e=np.linspace(0,1,k+1); b=np.digitize(zf,e[1:-1]); r=np.ones(k)
    for j in range(k):
        m=b==j
        if m.sum()>=100: r[j]=max(wq(t[m],.5,w[m])/max(gm,1e-8),1e-4)
    r=np.clip(r,floor,cap); r=r/max(wq(r[b],.5,w),1e-8); r=np.clip(r,floor,cap)
    return [r[np.digitize(z,e[1:-1])] for z in za]
def ecdf(fit): s=np.sort(fit); return lambda x: np.searchsorted(s,x,side='right')/len(s)
st=lambda M: np.abs(M).mean(1)
res=[]; curves={}
for n_dr in (2,4,8,'all'):
    for f in sorted(E):
        d=E[f]; runs=np.unique(d['gtr']); rng=np.random.default_rng(100+f)
        draws=[runs] if n_dr=='all' else [rng.choice(runs,n_dr,replace=False) for _ in range(3)]
        for di,sub in enumerate(draws):
            m=np.isin(d['gtr'],sub); seed=2027+f+di
            r=np.abs(d['ytr'][m]-d['oof'][m])
            F=ecdf(st(d['Ptr'][m])); zs,zc,zt=F(st(d['Ptr'][m])),F(st(d['Pca'])),F(st(d['Pte']))
            eca,ete=ec_scale(zs,r,d['gtr'][m],[zc,zt])
            et=ExtraTreesRegressor(200,min_samples_leaf=10,random_state=seed,n_jobs=-1).fit(d['Xtr'][m],r+1e-6)
            oca,ote=np.maximum(et.predict(d['Xca']),1e-3),np.maximum(et.predict(d['Xte']),1e-3)
            gl=GradientBoostingRegressor(loss='quantile',alpha=.05,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed).fit(d['Xtr'][m],d['ytr'][m])
            gh=GradientBoostingRegressor(loss='quantile',alpha=.95,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed+1).fit(d['Xtr'][m],d['ytr'][m])
            lca,uca,lte,ute=gl.predict(d['Xca']),gh.predict(d['Xca']),gl.predict(d['Xte']),gh.predict(d['Xte'])
            y,yc,p,pc=d['yte'],d['yca'],d['pte'],d['pca']
            one_c,one_t=np.ones(len(yc)),np.ones(len(y))
            fam={}
            for name,(sc,stt) in {'Standard CP':(one_c,one_t),'OOF normalized CP':(oca,ote),'EC-NCS':(eca,ete)}.items():
                s=np.abs(yc-pc)/sc; fam[name]=(p,stt,s)
            s=np.maximum(lca-yc,yc-uca); fam['CQR']=None
            for name,v in fam.items():
                if name=='CQR':
                    q=cq(s,1-A); lo,hi=lte-q,ute+q; Ufam=np.array([ute+cq(s,L) for L in LEV])
                else:
                    c0,sct,sco=v; q=cq(sco,1-A); lo,hi=c0-q*sct,c0+q*sct; Ufam=np.array([c0+cq(sco,L)*sct for L in LEV])
                for rr in np.unique(d['gte']):
                    i=d['gte']==rr; yy=y[i]; w=hi[i]-lo[i]
                    wink=np.mean(w+(2/A)*(lo[i]-yy)*(yy<lo[i])+(2/A)*(yy-hi[i])*(yy>hi[i]))
                    res.append(dict(n=str(n_dr),fold=f,draw=di,run=rr,method=name,cov=float(np.mean((yy>=lo[i])&(yy<=hi[i]))),wink=float(wink)))
                acc=Ufam<=B; late=y>B; k=(str(n_dr),di,name); c0=curves.get(k,np.zeros((3,len(LEV)))); c0+=np.array([acc.sum(1),(acc&late).sum(1),np.full(len(LEV),len(y))]); curves[k]=c0
    print('done',n_dr,flush=True)
import pandas as pd
R=pd.DataFrame(res)
# per draw: mean over the 20 runs, then mean/sd over draws
g=R.groupby(['n','draw','method']).agg(cov=('cov','mean'),wink=('wink','mean')).reset_index()
print(g.groupby(['n','method'])[['cov','wink']].agg(['mean','std']).round(3))
# paired run-level comparison EC vs others (pooled over draws)
for n in ('2','4','8','all'):
    piv=R[R.n==n].pivot_table(index=['draw','fold','run'],columns='method',values='wink')
    for b in ('Standard CP','OOF normalized CP','CQR'):
        dd=piv['EC-NCS']-piv[b]; print(n,'EC - ',b,'mean Winkler diff %.2f  EC better in %.0f%% of run-draws'%(dd.mean(),100*(dd<0).mean()))
def adm_at(c,T=.02):
    a=c[0]/c[2]; v=c[1]/np.maximum(c[0],1); ok=v<=T; return a[ok].max() if ok.any() else 0
print('admitted share at 2% late (mean over draws):')
for n in ('2','4','8','all'):
    print(n,'  '.join(f"{m}: {np.mean([adm_at(curves[(n,di,m)]) for di in range(1 if n=="all" else 3)])*100:5.1f}" for m in ('Standard CP','OOF normalized CP','EC-NCS','CQR')))
R.to_csv('/home/claude/rerun/fewdrives.csv',index=False)
