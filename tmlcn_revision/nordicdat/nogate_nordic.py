import sys, json, pickle, numpy as np, pandas as pd
src=open('/home/claude/nb/rerun_reserved.py').read().replace('cicv5g_pipeline.py','cicv5g_pipeline_nogate.py').split("packet_store_cal = {}")[0]
src=src.replace('group_table, outer_score = g["make_balanced_confirmation_folds"](frequency_confirm, CFG.confirmation_folds, 2027)','')
src=src.replace('group_table.to_csv(OUT / "fold_assignment.csv", index=False)','')
exec(src)
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon

d=pd.read_csv('/mnt/user-data/uploads/8f39e070-nordicdat.csv',low_memory=False).sort_values('timestamp')
d=d[(d.delay>0)&(d.delay<=2000)&(d.service_status!=-1)].copy()
d['RunGroup']='S'+pd.Series(np.r_[0,np.cumsum(np.diff(d.timestamp.values)>60)],index=d.index).astype(str)
c=d.RunGroup.value_counts(); d=d[d.RunGroup.isin(c[c>=500].index)].copy()
for col in ['band','ran','operator']: d[col]=d[col].astype(str)
d=d.rename(columns={'rsrp':'RSRP','sinr':'SINR'}); d['NetworkMode']=d['ran']; d['FrequencyHz']=1; d['Scenario']='NordicDat'
FEATS=['rsrq','rssi','RSRP','SINR','velocity_abs','acceleration_abs','velocity_long','acceleration_lat','acceleration_long','heading','elevation','band','ran','operator']
g['FEATURES']=FEATS; g['VALID_SIGNAL_FIELDS']=['RSRP','SINR']; g['PRIMARY_SIGNAL_FIELD']='RSRP'
d=d.reset_index(drop=True); print('rows',len(d),'sessions',d.RunGroup.nunique(),flush=True)
orig=g['explain_strength']
def pred_sig(name,model,X,bg): return model.predict(X)
rng=np.random.default_rng(2027); sess=d.RunGroup.unique(); rng.shuffle(sess); fold_of={s:i%5 for i,s in enumerate(sess)}
pk=[]; rows=[]; mn="Random Forest"
for fold in [2]:
    seed=2027+fold; g["set_seed"](seed)
    test=d[d.RunGroup.map(fold_of)==fold].copy(); rem=d[d.RunGroup.map(fold_of)!=fold].copy()
    ti,ci,_,_=g["train_calibration_group_split"](rem,seed); train=rem.iloc[ti].copy(); cal=rem.iloc[ci].copy()
    prep=g["build_preprocessor"](train[FEATS]); Xtr=prep.fit_transform(train[FEATS]).astype(np.float32)
    Xca=prep.transform(cal[FEATS]).astype(np.float32); Xte=prep.transform(test[FEATS]).astype(np.float32)
    ytr,yca,yte=train.delay.to_numpy(float),cal.delay.to_numpy(float),test.delay.to_numpy(float)
    model=g["build_model"](mn,seed).fit(Xtr,ytr); pca,pte=model.predict(Xca),model.predict(Xte)
    ints={}; q=g["conformal_quantile"](np.abs(yca-pca)); ints['Standard CP']=(pte-q,pte+q)
    scale,oofres=g["oof_scale"](Xtr,ytr,train.RunGroup.to_numpy(),seed,mn)
    sca,ste=np.maximum(scale.predict(Xca),1e-6),np.maximum(scale.predict(Xte),1e-6); qn=g["conformal_quantile"](np.abs(yca-pca)/sca)
    ints['OOF normalized CP']=(pte-qn*ste,pte+qn*ste)
    ints['CQR']=g["cqr"](Xtr,ytr,Xca,yca,Xte,seed)
    ref=g["run_balanced_reference_indices"](train.RunGroup.to_numpy(),CFG.max_explanation_reference_rows,seed+100); bg=Xtr[:64]
    sel={}
    for name,fn in (('EC-NCS',orig),('Pred-rank scale',pred_sig)):
        g['explain_strength']=fn; g["set_seed"](seed)
        params,selection,_=g["select_ec_hyperparameters"](train,seed,mn); sel[name]=selection['SelectedEC']
        s_tr,s_ca,s_te=(fn(mn,model,X,bg) for X in (Xtr[ref],Xca,Xte))
        e=g["EmpiricalCDF"]().fit(s_tr); ztr,zca,zte=e.transform(s_tr),e.transform(s_ca),e.transform(s_te)
        p=g["fit_explanation_scale"](ztr,oofres[ref],train.RunGroup.to_numpy()[ref],params["k"],params["shrinkage"],params["cap"])
        a0,b0=g["apply_explanation_scale"](zca,p),g["apply_explanation_scale"](zte,p); q0=g["conformal_quantile"](np.abs(yca-pca)/a0)
        ints[name+' (no gate)']=(pte-q0*b0,pte+q0*b0)
        if selection['SelectedEC']:
            p=g["fit_explanation_scale"](ztr,oofres[ref],train.RunGroup.to_numpy()[ref],params["k"],params["shrinkage"],params["cap"])
            a,b=g["apply_explanation_scale"](zca,p),g["apply_explanation_scale"](zte,p)
        else: a,b=np.ones(len(yca)),np.ones(len(yte))
        qe=g["conformal_quantile"](np.abs(yca-pca)/a); ints[name]=(pte-qe*b,pte+qe*b)
    g['explain_strength']=orig
    pk.append(pd.DataFrame(dict(fold=fold,run=test.RunGroup.values,y=yte,ran=test.ran.values,rsrp=test.RSRP.values,speed=test.velocity_abs.values,
        rsrp_t1=np.quantile(train.RSRP,1/3),rsrp_t2=np.quantile(train.RSRP,2/3),**{f'lo|{m}':np.asarray(v[0]) for m,v in ints.items()},**{f'hi|{m}':np.asarray(v[1]) for m,v in ints.items()})))
    for rg,ix in test.groupby('RunGroup').groups.items():
        pos=test.index.get_indexer(list(ix))
        for m,(lo,hi) in ints.items(): rows.append(dict(fold=fold,run=rg,method=m,**g["interval_metrics"](yte[pos],np.asarray(lo)[pos],np.asarray(hi)[pos])))
    print('fold',fold,'gate',sel,flush=True)
R=pd.DataFrame(rows); R.to_csv('/home/claude/nordic/nogate_fold2.csv',index=False); pd.concat(pk).to_pickle('/home/claude/nordic/nogate_fold2.pkl'); print('saved',flush=True)
print(R.groupby('method')[['Coverage','Winkler']].mean().round(3))
