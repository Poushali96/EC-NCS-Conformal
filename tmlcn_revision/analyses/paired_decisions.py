import pickle, numpy as np, json
P=pickle.load(open('/home/claude/rerun/packets.pkl','rb')); PC=pickle.load(open('/home/claude/rerun/packets_cal.pkl','rb'))
y=np.concatenate([P[f]['y'] for f in P]); run=np.concatenate([P[f]['run'] for f in P])
pte=np.concatenate([P[f]['pte'] for f in P])
freq=np.concatenate([PC[f]['freq_test'] for f in P])
U={m:np.concatenate([P[f]['intervals'][m][1] for f in P]) for m in P[0]['intervals']}
B=float(np.quantile(P[0]['ytrain'],0.9)); print('B',B,'n',len(y))
s=U['Standard CP']<=B; e=U['Validation-gated EC-NCS']<=B; late=y>B
def tab(idx):
    S,E,L=s[idx],e[idx],late[idx]
    return dict(n=int(idx.sum()), gain_ok=int((~S&E&~L).sum()), gain_late=int((~S&E&L).sum()),
                drop_ok=int((S&~E&~L).sum()), drop_late=int((S&~E&L).sum()),
                both_admit=int((S&E).sum()), both_admit_late=int((S&E&L).sum()),
                scp_adm=int(S.sum()), scp_late=int((S&L).sum()), ec_adm=int(E.sum()), ec_late=int((E&L).sum()))
out={'all':tab(np.ones(len(y),bool))}
for f in np.unique(freq): out[f'{f}Hz']=tab(freq==f)
for k,v in out.items(): print(k,v)
# bootstrap CIs on four counts as shares of all packets
runs=np.unique(run); ri={r:np.flatnonzero(run==r) for r in runs}; rng=np.random.default_rng(2027); bs=[]
for _ in range(1000):
    idx=np.concatenate([ri[r] for r in rng.choice(runs,len(runs))]); S,E,L=s[idx],e[idx],late[idx]; n=len(idx)
    bs.append([(~S&E&~L).sum()/n,(~S&E&L).sum()/n,(S&~E&~L).sum()/n,(S&~E&L).sum()/n])
bs=np.array(bs); ci=np.quantile(bs,[.025,.975],axis=0).T; print('CI shares',ci*100)
# per-run: how many runs have net more safe admissions and fewer late
pr=[]
for r in runs:
    i=ri[r]; S,E,L=s[i],e[i],late[i]; pr.append(((~S&E&~L).sum(),(~S&E&L).sum(),(S&~E&~L).sum(),(S&~E&L).sum(), (E&L).sum()-(S&L).sum(), E.sum()-S.sum()))
pr=np.array(pr); print('runs where EC admits more:',(pr[:,5]>0).sum(),'runs where EC has fewer late admits:',(pr[:,4]<0).sum(), 'equal late',(pr[:,4]==0).sum())
# real example: pairs with same point pred ~ around 24-26
for lab,cond in [('ec_rejects_late',s&~e&late),('both_admit_ok',s&e&~late)]:
    idx=np.flatnonzero(cond&(abs(pte-25)<1))
    print(lab,len(idx))
    if len(idx): j=idx[np.argsort(abs(pte[idx]-25))[:3]]; print([(round(pte[k],2),round(y[k],2),round(U['Standard CP'][k],2),round(U['Validation-gated EC-NCS'][k],2),freq[k]) for k in j])
json.dump({'B':B,'tables':out,'ci_share_pct':(ci*100).tolist()},open('/home/claude/rerun/paired_decisions.json','w'),indent=1,default=int)
fold=np.concatenate([[f]*len(P[f]['y']) for f in P])
ec=U['Validation-gated EC-NCS']; sc=U['Standard CP']
for f in P: print('fold',f,'identical' if np.allclose(ec[fold==f],sc[fold==f]) else 'adaptive', 'runs',np.unique(run[fold==f]).tolist())
diff_runs=[r for r in runs if not np.allclose(ec[run==r],sc[run==r])]; print('runs with any change',len(diff_runs))
k0=np.flatnonzero(s&~e&late&(abs(pte-25)<0.05)); print('late cands folds',np.unique(fold[k0],return_counts=True))
for f in np.unique(fold[k0]):
    c=np.flatnonzero((fold==f)&s&e&~late&(ec<sc-1.5)&(abs(pte-25)<0.05))
    print('fold',f,'narrow ok cands',len(c), [(round(pte[k],2),y[k],round(sc[k],2),round(ec[k],2),freq[k]) for k in c[:3]])
    l=k0[fold[k0]==f]; print('   late', [(round(pte[k],2),y[k],round(sc[k],2),round(ec[k],2),freq[k]) for k in l[:3]])
# per run late change among changed runs
for r in diff_runs:
    i=run==r; print(r, freq[i][0], 'adm',int(s[i].sum()),'->',int(e[i].sum()),'late',int((s&late)[i].sum()),'->',int((e&late)[i].sum()))
print('----')
for f in [1,2,3,4]:
    c=np.flatnonzero((fold==f)&s&e&~late&(ec<sc-0.5)&(abs(pte-25)<0.3))
    l=np.flatnonzero((fold==f)&s&~e&late&(abs(pte-25)<0.3))
    print(f,len(c),len(l))
    if len(c) and len(l):
        a=c[np.argmin(abs(pte[c]-25))]; b=l[np.argmin(abs(pte[l]-pte[a]))]
        for k in (a,b): print('   pred %.2f y %.1f scp %.2f ec %.2f freq %d run %s'%(pte[k],y[k],sc[k],ec[k],freq[k],run[k].split('frequency/')[1]))
print('====')
best=[]
for f in [1,2,3,4]:
    for sd in (True,False):
        A=np.flatnonzero((fold==f)&(s==sd)&e&~late&(y<B-3))
        Bb=np.flatnonzero((fold==f)&(s==sd)&~e&late&(y>B+3))
        if len(A)==0 or len(Bb)==0: continue
        oa=np.argsort(pte[A]); pa=pte[A][oa]
        for k in Bb:
            j=np.searchsorted(pa,pte[k]); 
            for jj in (j-1,j):
                if 0<=jj<len(pa) and abs(pa[jj]-pte[k])<0.02: best.append((abs(pa[jj]-pte[k]),f,sd,A[oa[jj]],k))
print(len(best))
seen=set()
for d,f,sd,a,b in sorted(best,key=lambda t:(abs(pte[t[3]]-25),t[0]))[:40]:
    key=(f,sd,round(pte[a],1))
    if key in seen: continue
    seen.add(key)
    print('fold',f,'scp_admits',sd)
    for k in (a,b): print('   pred %.2f y %.1f scp %.2f ec %.2f freq %d run %s'%(pte[k],y[k],sc[k],ec[k],freq[k],run[k].split('frequency/')[1]))
