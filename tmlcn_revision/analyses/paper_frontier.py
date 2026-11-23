"""Frontier (admitted vs late-among-admitted at B=31 ms, confidence level swept) for the paper's exact methods,
plus a deployable rule (level chosen on the calibration runs). Writes numbers and the figure."""
import pickle, numpy as np, json, sys
sys.path.insert(0,'/home/claude/work')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from build_figures import C, COL_W
B=31.0
P=pickle.load(open('/home/claude/rerun/packets.pkl','rb')); PC=pickle.load(open('/home/claude/rerun/packets_cal.pkl','rb')); E=pickle.load(open('/home/claude/rerun/extended.pkl','rb'))
LEV=np.round(np.arange(0.01,0.9995,0.0025),4); folds=sorted(P)
def cq(s,L):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*L))-1; return s[min(max(k,0),n-1)]
names=['Standard CP','OOF normalized CP','CQR','Validation-gated EC-NCS']
U={m:[] for m in names}; Ucal={m:[] for m in names}
for f in folds:
    d,c=P[f],PC[f]; sel=E[f]['selection']['SelectedEC']; oc,ot=np.ones(len(c['ycal'])),np.ones(len(d['pte']))
    sc={'Standard CP':(oc,ot),'OOF normalized CP':(c['sca'],c['ste']),'Validation-gated EC-NCS':(c['sec'],c['setest']) if sel else (oc,ot)}
    for m in names:
        if m=='CQR':
            s=np.maximum(c['lca']-c['ycal'],c['ycal']-c['uca']); qs=[cq(s,L) for L in LEV]
            U[m].append(np.array([c['ute']+q for q in qs])); Ucal[m].append(np.array([c['uca']+q for q in qs]))
        else:
            a,b=sc[m]; s=np.abs(c['ycal']-c['pca'])/a; qs=[cq(s,L) for L in LEV]
            U[m].append(np.array([d['pte']+q*b for q in qs])); Ucal[m].append(np.array([c['pca']+q*a for q in qs]))
y=np.concatenate([P[f]['y'] for f in folds]); run=np.concatenate([P[f]['run'] for f in folds]); late=y>B; runs=np.unique(run)
UU={m:np.concatenate(U[m],1) for m in names}
def curve(u,l): acc=u<=B; return acc.mean(1),(acc&l).sum(1)/np.maximum(acc.sum(1),1)
def adm_at(u,l,T):
    a,v=curve(u,l); ok=v<=T; return a[ok].max() if ok.any() else 0.0
num={'frontier':{},'loro':{},'deploy':{}}
for m in names:
    num['frontier'][m]={str(T):adm_at(UU[m],late,T) for T in (0.01,0.02,0.03)}
    print(f'{m:25s}',num['frontier'][m])
for a,b in [('Validation-gated EC-NCS','Standard CP'),('Validation-gated EC-NCS','OOF normalized CP'),('Validation-gated EC-NCS','CQR'),('OOF normalized CP','Standard CP'),('CQR','Standard CP')]:
    for T in (0.02,0.03):
        dd=np.array([adm_at(UU[a][:,run!=r],late[run!=r],T)-adm_at(UU[b][:,run!=r],late[run!=r],T) for r in runs])*100
        num['loro'][f'{a}|{b}|{T}']=[float(dd.min()),float(dd.max()),float((adm_at(UU[a],late,T)-adm_at(UU[b],late,T))*100)]
        print(a,'-',b,T,num['loro'][f'{a}|{b}|{T}'])
# deployable: per fold choose the most permissive level whose calibration late share among admitted <= T
for T in (0.01,0.02):
    num['deploy'][str(T)]={}
    for m in names:
        acc_all=[]
        for k,f in enumerate(folds):
            c=PC[f]; lc=c['ycal']>B; pick=None
            for i in range(len(LEV)):
                acc=Ucal[m][k][i]<=B
                if acc.sum() and (acc&lc).sum()/acc.sum()<=T: pick=i; break
            acc_all.append(U[m][k][pick]<=B if pick is not None else np.zeros(len(P[f]['y']),bool))
        acc=np.concatenate(acc_all); num['deploy'][str(T)][m]=[float(acc.mean()),float((acc&late).sum()/max(acc.sum(),1))]
        print('deploy',T,m,num['deploy'][str(T)][m])
json.dump(num,open('/home/claude/rerun/paper_frontier.json','w'),indent=1)
# ---------------- figure
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'font.size':7.5,'pdf.fonttype':42})
col={'Standard CP':C['Standard CP'],'OOF normalized CP':C['OOF normalized CP'],'CQR':C['CQR'],'Validation-gated EC-NCS':C['EC-NCS']}
fig,ax=plt.subplots(figsize=(COL_W,COL_W*0.78))
i90=int(np.argmin(abs(LEV-0.90)))
for m in names:
    a,v=curve(UU[m],late)
    o=np.argsort(v); ax.plot(v*100,a*100,color=col[m],lw=1.4 if 'EC' in m else 1.0,label=m.replace('Validation-gated ','')) 
    ax.plot(v[i90]*100,a[i90]*100,'o',ms=4,color=col[m],mec='white',mew=0.6,zorder=5)
ax.axvline(2,color='0.6',lw=0.6,ls=(0,(2,2)))
ax.text(2.08,8,'2% late',color='0.4',fontsize=6.5)
ax.set_xlim(0,6); ax.set_ylim(0,100)
ax.set_xlabel('Admitted packets that miss the 31 ms budget (%)'); ax.set_ylabel('Packets admitted (%)')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.grid(alpha=0.25,lw=0.4)
ax.legend(frameon=False,fontsize=6.5,loc='lower right')
fig.tight_layout(pad=0.3); fig.savefig('/home/claude/work/figures/cicv5g_admission_frontier.pdf'); fig.savefig('/tmp/frontier.png',dpi=200)
