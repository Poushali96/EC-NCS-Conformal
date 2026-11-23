"""How many calibration runs, and which transmission frequencies, does a 2% late-share target need to transfer?
No retraining: within each fold, the frozen predictor's 7 unseen runs (3 calibration + 4 test) are re-split
into k calibration runs and 7-k test runs, over all combinations. Operating level chosen on calibration only."""
import pickle, itertools, json, numpy as np
B=31.0; T=0.02
P=pickle.load(open('/home/claude/rerun/packets.pkl','rb')); PC=pickle.load(open('/home/claude/rerun/packets_cal.pkl','rb'))
E=pickle.load(open('/home/claude/rerun/extended.pkl','rb'))
LEV=np.round(np.arange(0.01,0.9995,0.005),4)
M=['Standard CP','OOF normalized CP','Validation-gated EC-NCS','CQR']
rows=[]
for f in sorted(P):
    d,c=P[f],PC[f]; sel=E[f]['selection']['SelectedEC']
    # stack the 7 runs: centre (point or CQR upper), lower-for-CQR, scales, y, run, freq
    y=np.r_[c['ycal'],d['y']]; run=np.r_[c['run_cal'],d['run']]; fq=np.r_[c['freq_cal'],c['freq_test']]
    p=np.r_[c['pca'],d['pte']]; one=np.ones(len(y))
    sc={'Standard CP':one,'OOF normalized CP':np.r_[c['sca'],c['ste']],
        'Validation-gated EC-NCS':np.r_[c['sec'],c['setest']] if sel else one}
    lo,hi=np.r_[c['lca'],c['lte']],np.r_[c['uca'],c['ute']]
    score={m:np.abs(y-p)/sc[m] for m in sc}; score['CQR']=np.maximum(lo-y,y-hi)
    def upper(m,q): return hi+q if m=='CQR' else p+q*sc[m]
    runs=np.unique(run); idx={r:np.flatnonzero(run==r) for r in runs}; rf={r:fq[idx[r][0]] for r in runs}
    late=y>B
    for k in range(1,7):
        for cal in itertools.combinations(runs,k):
            ci=np.concatenate([idx[r] for r in cal]); tests=[r for r in runs if r not in cal]
            cfreq={rf[r] for r in cal}
            for m in M:
                s=np.sort(score[m][ci]); n=len(s)
                qs=s[np.minimum(np.ceil((n+1)*LEV).astype(int)-1,n-1)]
                q=None
                for qq in qs:  # most permissive level meeting T on calibration runs
                    a=upper(m,qq)[ci]<=B
                    if a.sum() and (a&late[ci]).sum()/a.sum()<=T: q=qq; break
                for r in tests:
                    i=idx[r]; a=(upper(m,q)[i]<=B) if q is not None else np.zeros(len(i),bool)
                    rows.append((f,k,m,int(rf[r] in cfreq),rf[r],len(i),int(a.sum()),int((a&late[i]).sum())))
R=np.array(rows,dtype=object)
def agg(mask):
    n,a,l=R[mask,5].sum(),R[mask,6].sum(),R[mask,7].sum(); return a/n*100, l/max(a,1)*100
print('k = calibration runs.  cells: admitted% / late-among-admitted%   (target 2%)')
for cov,lab in ((None,'all test runs'),(1,'test freq IN calibration'),(0,'test freq MISSING')):
    print('\n--',lab)
    print('k  '+'  '.join(f'{m[:12]:>16s}' for m in M))
    for k in range(1,7):
        line=f'{k}  '
        for m in M:
            mk=(R[:,1]==k)&(R[:,2]==m)&((R[:,3]==cov) if cov is not None else True)
            line+=f'  {agg(mk)[0]:6.1f} / {agg(mk)[1]:5.2f}' if mk.any() else '        -       '
        print(line)
# share of (split, test run) pairs whose late share <= 3% (per run reliability), k=6, freq in cal
print('\nshare of test runs (k=1..6, freq in cal) with late-among-admitted <= 3%:')
for m in M:
    out=[]
    for k in range(1,7):
        mk=(R[:,1]==k)&(R[:,2]==m)&(R[:,3]==1); a=R[mk,6].astype(float); l=R[mk,7].astype(float)
        out.append(np.mean(l<=0.03*np.maximum(a,1)))
    print(f'{m:25s}',' '.join(f'{v:.2f}' for v in out))
print('\nby test frequency, k=3 vs k=6, freq in cal:')
for fr in (10,33,100):
    print(fr,' '.join(f'{m[:8]} k3 {agg((R[:,1]==3)&(R[:,2]==m)&(R[:,3]==1)&(R[:,4]==fr))[1]:.2f} k6 {agg((R[:,1]==6)&(R[:,2]==m)&(R[:,3]==1)&(R[:,4]==fr))[1]:.2f}' for m in M))
pickle.dump(rows,open('/home/claude/rerun/calsize_rows.pkl','wb'))
