import pickle, numpy as np, json
B=31.0
P=pickle.load(open('/home/claude/rerun/packets.pkl','rb')); PC=pickle.load(open('/home/claude/rerun/packets_cal.pkl','rb')); E=pickle.load(open('/home/claude/rerun/extended.pkl','rb'))
LEV=np.round(np.arange(0.70,0.999,0.001),4); folds=sorted(P)
def cq(s,L):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*L))-1; return s[min(max(k,0),n-1)]
fam={}
for f in folds:
    d,c=P[f],PC[f]; sel=E[f]['selection']['SelectedEC']
    one_c,one_t=np.ones(len(c['ycal'])),np.ones(len(d['pte']))
    ec=(c['sec'],c['setest']) if sel else (one_c,one_t)
    # paper methods: two-sided scores, upper end used by the controller
    for name,(sc,st) in {'Standard CP':(one_c,one_t),'OOF normalized CP':(c['sca'],c['ste']),'Validation-gated EC-NCS':ec}.items():
        s=np.abs(c['ycal']-c['pca'])/sc; fam.setdefault(name,[]).append(np.array([d['pte']+cq(s,L)*st for L in LEV]))
    s=np.maximum(c['lca']-c['ycal'],c['ycal']-c['uca']); fam.setdefault('CQR',[]).append(np.array([c['ute']+cq(s,L) for L in LEV]))
    # CQR with the EC-NCS scale (same gated scale, no new tuning)
    s=np.maximum(c['lca']-c['ycal'],c['ycal']-c['uca'])/ec[0]
    fam.setdefault('CQR + EC-NCS scale',[]).append(np.array([c['ute']+cq(s,L)*ec[1] for L in LEV]))
    s=np.maximum(c['lca']-c['ycal'],c['ycal']-c['uca'])/c['sca']
    fam.setdefault('CQR + OOF scale',[]).append(np.array([c['ute']+cq(s,L)*c['ste'] for L in LEV]))
U={m:np.concatenate(v,1) for m,v in fam.items()}
y=np.concatenate([P[f]['y'] for f in folds]); run=np.concatenate([P[f]['run'] for f in folds]); late=y>B; runs=np.unique(run)
def curve(u,l): acc=u<=B; return acc.mean(1),(acc&l).sum(1)/np.maximum(acc.sum(1),1)
def adm_at(u,l,T):
    a,v=curve(u,l); ok=v<=T; return a[ok].max() if ok.any() else 0.0
out={'T':{}, 'loro':{}}
for m,u in U.items():
    out['T'][m]={T:adm_at(u,late,T) for T in (0.01,0.015,0.02,0.03)}
    print(f'{m:26s}',' '.join(f'{T}:{v*100:5.1f}' for T,v in out['T'][m].items()))
for a,b in [('Validation-gated EC-NCS','OOF normalized CP'),('Validation-gated EC-NCS','Standard CP'),('CQR + EC-NCS scale','CQR'),('CQR + OOF scale','CQR'),('Validation-gated EC-NCS','CQR')]:
    for T in (0.01,0.02):
        dd=[]
        for r in runs:
            i=np.flatnonzero(run!=r); dd.append(adm_at(U[a][:,i],late[i],T)-adm_at(U[b][:,i],late[i],T))
        dd=np.array(dd)*100; out['loro'][f'{a} - {b} @{T}']=dd.tolist()
        print(f'{a} - {b} T={T}: min {dd.min():+.1f} max {dd.max():+.1f} full {(adm_at(U[a],late,T)-adm_at(U[b],late,T))*100:+.1f}')
curves={m:[c.tolist() for c in curve(u,late)] for m,u in U.items()}
json.dump(dict(out=out,curves=curves,LEV=LEV.tolist()),open('/home/claude/rerun/frontier.json','w'))
