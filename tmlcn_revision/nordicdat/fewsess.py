import numpy as np, pandas as pd
src=open('/home/claude/nb/rerun_reserved.py').read().split("packet_store_cal = {}")[0]
src=src.replace('group_table, outer_score = g["make_balanced_confirmation_folds"](frequency_confirm, CFG.confirmation_folds, 2027)','').replace('group_table.to_csv(OUT / "fold_assignment.csv", index=False)','')
exec(src)
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
d=pd.read_csv('/mnt/user-data/uploads/8f39e070-nordicdat.csv',low_memory=False).sort_values('timestamp')
d=d[(d.delay>0)&(d.delay<=2000)&(d.service_status!=-1)].copy()
d['RunGroup']='S'+pd.Series(np.r_[0,np.cumsum(np.diff(d.timestamp.values)>60)],index=d.index).astype(str)
c=d.RunGroup.value_counts(); d=d[d.RunGroup.isin(c[c>=500].index)].copy()
for col in ['band','ran','operator']: d[col]=d[col].astype(str)
d=d.rename(columns={'rsrp':'RSRP','sinr':'SINR'}); d['NetworkMode']=d['ran']; d['FrequencyHz']=1; d['Scenario']='NordicDat'
FEATS=['rsrq','rssi','RSRP','SINR','velocity_abs','acceleration_abs','velocity_long','acceleration_lat','acceleration_long','heading','elevation','band','ran','operator']
g['FEATURES']=FEATS; g['VALID_SIGNAL_FIELDS']=['RSRP','SINR']; g['PRIMARY_SIGNAL_FIELD']='RSRP'; d=d.reset_index(drop=True)
rng0=np.random.default_rng(2027); sess=d.RunGroup.unique(); rng0.shuffle(sess); fold_of={s:i%5 for i,s in enumerate(sess)}
A=0.1
def cq(s,L=0.9): s=np.sort(s); n=len(s); return s[min(int(np.ceil((n+1)*L))-1,n-1)]
def run_w(g_): u,i,c=np.unique(g_,return_inverse=True,return_counts=True); return 1.0/c[i]
def wq(v,q,w): o=np.argsort(v); v,w=v[o],w[o]; c=np.cumsum(w)/w.sum(); return v[min(np.searchsorted(c,q),len(v)-1)]
def bins(zf,t,gg,za,k=5,cap=2.0,floor=.5):
    w=run_w(gg); gm=max(wq(t,.5,w),1e-8); e=np.linspace(0,1,k+1); b=np.digitize(zf,e[1:-1]); r=np.ones(k)
    for j in range(k):
        m=b==j
        if m.sum()>=100: r[j]=max(wq(t[m],.5,w[m])/gm,1e-4)
    r=np.clip(r,floor,cap); r=r/max(wq(r[b],.5,w),1e-8); r=np.clip(r,floor,cap); return [r[np.digitize(z,e[1:-1])] for z in za]
def ecdf(f): s=np.sort(f); return lambda x: np.searchsorted(s,x,side='right')/len(s)
def wink(y,lo,hi): return (hi-lo)+(2/A)*(lo-y)*(y<lo)+(2/A)*(y-hi)*(y>hi)
rows=[]; mn="Random Forest"
for fold in range(5):
    seed=2027+fold; g["set_seed"](seed)
    test=d[d.RunGroup.map(fold_of)==fold].copy(); rem=d[d.RunGroup.map(fold_of)!=fold].copy()
    ti,ci,_,_=g["train_calibration_group_split"](rem,seed); train=rem.iloc[ti].copy(); cal=rem.iloc[ci].copy()
    prep=g["build_preprocessor"](train[FEATS]); Xtr=prep.fit_transform(train[FEATS]).astype(np.float32)
    Xca=prep.transform(cal[FEATS]).astype(np.float32); Xte=prep.transform(test[FEATS]).astype(np.float32)
    ytr,yca,yte=train.delay.to_numpy(float),cal.delay.to_numpy(float),test.delay.to_numpy(float); gtr=train.RunGroup.to_numpy()
    model=g["build_model"](mn,seed).fit(Xtr,ytr); pca,pte=model.predict(Xca),model.predict(Xte)
    _,oofres=g["oof_scale"](Xtr,ytr,gtr,seed,mn)
    from sklearn.model_selection import GroupKFold
    oofp=np.full(len(ytr),np.nan)
    for k_,(a_,b_) in enumerate(GroupKFold(3).split(Xtr,groups=gtr)):
        m2=g["build_model"](mn,seed+100+k_).fit(Xtr[a_],ytr[a_]); oofp[b_]=m2.predict(Xtr[b_])
    bg=Xtr[:64]; str_tr,str_ca,str_te=(g["explain_strength"](mn,model,X,bg) for X in (Xtr,Xca,Xte))
    runs=np.unique(gtr); rng=np.random.default_rng(100+fold)
    one_c,one_t=np.ones(len(yca)),np.ones(len(yte))
    q=cq(np.abs(yca-pca)); base={'Standard CP':(pte-q,pte+q)}
    for n in (2,4):
        for di in range(3):
            m=np.isin(gtr,rng.choice(runs,n,replace=False)); r=oofres[m]; ints=dict(base)
            for name,(sf,sc_,st_) in {'EC-NCS':(str_tr,str_ca,str_te),'Pred-rank scale':(oofp,pca,pte)}.items():
                F=ecdf(sf[m]); a,b=bins(F(sf[m]),r,gtr[m],[F(sc_),F(st_)]); qq=cq(np.abs(yca-pca)/a); ints[name]=(pte-qq*b,pte+qq*b)
            et=ExtraTreesRegressor(200,min_samples_leaf=10,random_state=seed+di,n_jobs=-1).fit(Xtr[m],r+1e-6)
            a,b=np.maximum(et.predict(Xca),1e-3),np.maximum(et.predict(Xte),1e-3); qq=cq(np.abs(yca-pca)/a); ints['OOF normalized CP']=(pte-qq*b,pte+qq*b)
            gl=GradientBoostingRegressor(loss='quantile',alpha=.05,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed).fit(Xtr[m],ytr[m])
            gh=GradientBoostingRegressor(loss='quantile',alpha=.95,n_estimators=250,max_depth=3,learning_rate=.04,random_state=seed+1).fit(Xtr[m],ytr[m])
            s=np.maximum(gl.predict(Xca)-yca,yca-gh.predict(Xca)); qq=cq(s); ints['CQR']=(gl.predict(Xte)-qq,gh.predict(Xte)+qq)
            for rg in np.unique(test.RunGroup):
                i=test.RunGroup.values==rg
                for mm,(lo,hi) in ints.items(): rows.append(dict(n=n,fold=fold,draw=di,run=rg,method=mm,W=float(wink(yte[i],lo[i],hi[i]).mean()),cov=float(((yte[i]>=lo[i])&(yte[i]<=hi[i])).mean())))
    print('fold',fold,flush=True)
pd.DataFrame(rows).to_csv('/home/claude/nordic/fewsess.csv',index=False); print('saved',flush=True)
