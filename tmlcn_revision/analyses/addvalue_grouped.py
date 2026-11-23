"""Does the attribution summary add value beyond predicted delay and frequency? (full data, exploratory)"""
import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
E=pickle.load(open('/home/claude/rerun/extended.pkl','rb')); A=0.1
def cq(s,L): s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*L))-1; return s[min(max(k,0),n-1)]
st=lambda M: np.abs(M).mean(1)
rows=[]
for f,d in E.items():
    rng=np.random.default_rng(f); r=np.abs(d['ytr']-d['oof'])
    a_tr,a_ca,a_te=st(d['Ptr']),st(d['Pca']),st(d['Pte'])
    # shuffle attribution within predicted-delay deciles x frequency (control)
    def shuf(a,p,fq):
        a=a.copy(); bins=np.digitize(p,np.quantile(p,np.linspace(0,1,11)[1:-1]))
        for b in np.unique(bins):
            for v in np.unique(fq):
                i=np.flatnonzero((bins==b)&(fq==v)); a[i]=a[rng.permutation(i)]
        return a
    G=[[0,1],[2,3],[4]]  # prespecified groups: radio (RSRP,SINR), mobility (velocity,heading), traffic (frequency)
    grp=lambda P: np.c_[[np.abs(P[:,c]).sum(1) for c in G]].T
    g_tr,g_ca,g_te=grp(d['Ptr']),grp(d['Pca']),grp(d['Pte'])
    feats={'pred+freq':lambda p,fq,a:np.c_[p,fq],
           'pred+freq+attr':lambda p,fq,a:np.c_[p,fq,a]}
    sets={'pred+freq':(d['oof'],d['ftr'],a_tr),'pred+freq+attr':(d['oof'],d['ftr'],a_tr),
          'pred+freq+attr(shuffled)':(d['oof'],d['ftr'],shuf(a_tr,d['oof'],d['ftr']))}
    sets={'pred+freq':(d['oof'],d['ftr'],a_tr),'pred+freq+grouped attr':(d['oof'],d['ftr'],g_tr)}
    for name,(p,fq,a) in sets.items():
        fn=feats['pred+freq'] if name=='pred+freq' else feats['pred+freq+attr']
        acal=g_ca if 'grouped' in name else a_ca
        ate=g_te if 'grouped' in name else a_te
        et=ExtraTreesRegressor(200,min_samples_leaf=50,random_state=f,n_jobs=-1).fit(fn(p,fq,a),r+1e-6)
        sc=np.maximum(et.predict(fn(d['pca'],d['fca'],acal)),1e-3); s_te=np.maximum(et.predict(fn(d['pte'],d['fte'],ate)),1e-3)
        q=cq(np.abs(d['yca']-d['pca'])/sc,1-A); lo,hi=d['pte']-q*s_te,d['pte']+q*s_te; y=d['yte']
        for rr in np.unique(d['gte']):
            i=d['gte']==rr; w=hi[i]-lo[i]; yy=y[i]
            rows.append(dict(fold=f,run=rr,freq=d['fte'][i][0],model=name,cov=np.mean((yy>=lo[i])&(yy<=hi[i])),
                 wink=np.mean(w+(2/A)*(lo[i]-yy)*(yy<lo[i])+(2/A)*(yy-hi[i])*(yy>hi[i])),
                 upper_miss=np.mean(yy>hi[i])))
R=pd.DataFrame(rows); R.to_csv('/home/claude/rerun/addvalue_grouped.csv',index=False)
print(R.groupby('model')[['cov','wink','upper_miss']].mean().round(3))
print(R[R.freq==100].groupby('model')[['cov','wink','upper_miss']].mean().round(3))
pv=R.pivot_table(index='run',columns='model',values='wink')
from scipy.stats import wilcoxon
for b in ('pred+freq',):
    dd=pv['pred+freq+grouped attr']-pv[b]; print('attr minus',b,'mean %.2f, better in %d/20 runs, Wilcoxon p=%.3f'%(dd.mean(),(dd<0).sum(),wilcoxon(dd).pvalue))
