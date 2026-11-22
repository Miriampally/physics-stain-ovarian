import json, numpy as np, sys
sys.path.insert(0,"../code")
import histo_utils as hu
S="../results/"
R={43:json.load(open(S+"seed43_no_radiomics/results.json")),42:json.load(open(S+"seed42_no_radiomics/results.json"))}
V=["A_baseline","H_stainaug","C_stainOD+consist","N_reinhard"]
conds=["clean","clean_sub","bright0.8","bright0.65","bright0.5","wb","wb0.2","hscale1.3","hscale2.0","escale0.7","escale0.4","jpeg30","noise"]
def f(seed,v,c): return np.array([x['slide']['balanced_acc'] for x in R[seed][v][c]])
out=[]
out.append("| Condition | C seed42 | C seed43 | C pooled | Reinhard pooled | C-Reinhard (10 folds) | folds C>Reinhard | C-A | folds C>A | C-H | folds C>H |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
rows={}
for c in conds:
    cp=np.concatenate([f(s,"C_stainOD+consist",c) for s in (42,43)])
    rp=np.concatenate([f(s,"N_reinhard",c) for s in (42,43)])
    ap=np.concatenate([f(s,"A_baseline",c) for s in (42,43)])
    hp=np.concatenate([f(s,"H_stainaug",c) for s in (42,43)])
    rows[c]=dict(C=cp.mean(),R=rp.mean(),A=ap.mean(),H=hp.mean(),wR=int((cp>rp).sum()),wA=int((cp>ap).sum()),wH=int((cp>hp).sum()),
      C42=f(42,"C_stainOD+consist",c).mean(),C43=f(43,"C_stainOD+consist",c).mean(), n=len(cp))
    r=rows[c]
    out.append(f"| {c} | {r['C42']:.3f} | {r['C43']:.3f} | {r['C']:.3f} | {r['R']:.3f} | {r['C']-r['R']:+.3f} | {r['wR']}/{r['n']} | {r['C']-r['A']:+.3f} | {r['wA']}/{r['n']} | {r['C']-r['H']:+.3f} | {r['wH']}/{r['n']} |")
print("\n".join(out))
# per-seed wins vs reinhard
for s in (42,43):
    print("seed",s,{c:int((f(s,"C_stainOD+consist",c)>f(s,"N_reinhard",c)).sum()) for c in conds})
# bootstrap clean seed 42
df=hu.load_slide_predictions(S+"seed42_no_radiomics/slide_predictions.csv")
print(df.variant.unique(), df.slide.nunique())
print(hu.bootstrap_table_md(df,V,"A_baseline"))
print(hu.bootstrap_table_md(df,V,"N_reinhard"))
json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in rows.items()},open("pooled_norad.json","w"),indent=1)
