import pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
A=0.1
def wink(y,lo,hi): return (hi-lo)+(2/A)*(lo-y)*(y<lo)+(2/A)*(y-hi)*(y>hi)
def paired(df,a,b,by=None):
    out=[]
    for key,sub in ([('all',df)] if by is None else df.groupby(by)):
        P=sub.groupby('run')[[a,b]].mean(); P=P[P.index.isin(sub.run.unique())]
        d=P[a]-P[b]; nz=(d.abs()>1e-9).sum()
        p=wilcoxon(d[d.abs()>1e-9]).pvalue if nz>=6 else np.nan
        out.append((key,len(P),100*d.mean()/P[b].mean(),int((d<-1e-9).sum()),int((d>1e-9).sum()),p))
    return out
def show(t,rows):
    for k,n,rel,w,l,p in rows: print(f'  {t:28s} {str(k):14s} runs {n:2d}  {rel:+6.1f}%  better {w:2d} worse {l:2d}  p={p:.3f}')
# ---------------- NordicDat
K=pd.read_pickle('/home/claude/nordic/packets.pkl')
for m in ['Standard CP','EC-NCS','EC-NCS (no gate)','Pred-rank scale','Pred-rank scale (no gate)','OOF normalized CP','CQR']:
    K['W|'+m]=wink(K.y,K['lo|'+m],K['hi|'+m])
K['rsrp_t']=np.where(K.rsrp<=K.rsrp_t1,'low',np.where(K.rsrp<=K.rsrp_t2,'mid','high')); K['moving']=np.where(K.speed>1,'moving','stationary')
print('NordicDat: gate on vs off (vs Standard CP)')
show('EC no gate - Std',paired(K,'W|EC-NCS (no gate)','W|Standard CP')); show('EC gated - Std',paired(K,'W|EC-NCS','W|Standard CP'))
show('EC no gate - Std, fold 2',paired(K[K.fold==2],'W|EC-NCS (no gate)','W|Standard CP'))
print('NordicDat: EC vs predicted-delay scale by condition (prespecified groups, exploratory)')
for by in (None,'ran','rsrp_t','moving'): show('EC - Pred '+str(by),paired(K,'W|EC-NCS','W|Pred-rank scale',by))
# ---------------- CICV5G
P=pickle.load(open('/home/claude/rerun/packets.pkl','rb')); PC=pickle.load(open('/home/claude/rerun/packets_cal.pkl','rb')); E=pickle.load(open('/home/claude/rerun/extended.pkl','rb'))
rows=[]
for f in sorted(P):
    d,c,e=P[f],PC[f],E[f]; sel=e['selection']['SelectedEC']; y=d['y']; pte=d['pte']; pca=c['pca']
    q0=np.quantile(np.abs(c['ycal']-pca),0.9,method='higher')
    def ci(sc,st):
        s=np.sort(np.abs(c['ycal']-pca)/sc); n=len(s); q=s[min(int(np.ceil((n+1)*0.9))-1,n-1)]; return pte-q*st,pte+q*st
    lo0,hi0=ci(np.ones(len(pca)),np.ones(len(pte))); lo1,hi1=ci(c['sec'],c['setest'])
    t1,t2=np.quantile(e['Xtr'][:,0],[1/3,2/3]); r=e['Xte'][:,0]
    rows.append(pd.DataFrame({'fold':f,'run':d['run'],'freq':c['freq_test'],'rsrp_t':np.where(r<=t1,'low',np.where(r<=t2,'mid','high')),
        'W|Standard CP':wink(y,lo0,hi0),'W|EC-NCS (no gate)':wink(y,lo1,hi1),'W|EC-NCS':wink(y,lo1,hi1) if sel else wink(y,lo0,hi0)}))
C=pd.concat(rows)
print('CICV5G: gate on vs off (vs Standard CP)')
show('EC no gate - Std',paired(C,'W|EC-NCS (no gate)','W|Standard CP')); show('EC gated - Std',paired(C,'W|EC-NCS','W|Standard CP'))
show('EC no gate - Std, fold 0',paired(C[C.fold==0],'W|EC-NCS (no gate)','W|Standard CP'))
print('CICV5G: EC vs Std by condition'); 
for by in ('freq','rsrp_t'): show('EC - Std '+by,paired(C,'W|EC-NCS','W|Standard CP',by))
