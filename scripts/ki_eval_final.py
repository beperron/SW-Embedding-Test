import glob, json, os
import numpy as np, pandas as pd
from swrd_eval import config, corpus as C
from swrd_eval.embedders import build
EMB=config.EMB_DIR
ids=np.load(EMB/'ids.npy')
corp=pd.read_parquet('runs/corpus.parquet'); corp['yr']=pd.to_numeric(corp['publication_year'],errors='coerce')
w=corp[(corp.yr>=1989)&(corp.yr<=2025)].copy()
w['tl']=w['title'].astype(str).str.strip().str.lower(); w['_l']=w['abstract'].astype(str).str.len()
keep=set()
for (tl,yr),g in w.groupby(['tl','yr']):
    keep.add(int(g.sort_values(['_l','id'],ascending=[False,True])['id'].iloc[0]))
inwin=keep
mask=np.array([int(p) in inwin for p in ids])
row_of={int(p):i for i,p in enumerate(ids)}
kq=pd.read_parquet(config.RUNS_DIR/'ki_queries.parquet')
kq=kq[kq.seed_id.apply(lambda s:int(s) in row_of and int(s) in inwin)].reset_index(drop=True)
queries=kq['query'].tolist(); src=np.array([row_of[int(s)] for s in kq.seed_id])
print(len(kq),'bounded KI queries',flush=True)
def mets(rk):
    rk=np.asarray(rk,float)
    return dict(S1=float((rk==1).mean()),S10=float((rk<=10).mean()),MRR=float(np.where(rk<=10,1.0/rk,0).mean()),R100=float((rk<=100).mean()))
def ranks(D,Q):
    D=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-12); Q=Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-12)
    S=D@Q.T; S[~mask,:]=-1e9
    ss=S[src,np.arange(S.shape[1])]
    return (S>ss).sum(axis=0)+1
rows=[]
for m in config.enabled_dense():
    k=m['key']; npy=EMB/f'{k}.npy'
    if not npy.exists(): continue
    try:
        e=build(m); Q=e.encode_queries(queries,batch_size=m.get('batch_size',64)).astype(np.float32)
        rows.append({'system':k,**mets(ranks(np.load(npy).astype(np.float32),Q))}); print(rows[-1],flush=True); del e,Q
    except Exception as ex: print('skip',k,str(ex)[:100],flush=True)
try:
    from openai import OpenAI; cli=OpenAI()
    for k,mod in [('openai-3-small','text-embedding-3-small'),('openai-3-large','text-embedding-3-large')]:
        vecs=[]
        for i in range(0,len(queries),256):
            vecs.extend(d.embedding for d in cli.embeddings.create(model=mod,input=queries[i:i+256]).data)
        rows.append({'system':k,**mets(ranks(np.load(EMB/f'{k}.npy').astype(np.float32),np.array(vecs,dtype=np.float32)))}); print(rows[-1],flush=True)
except Exception as ex: print('openai skip',str(ex)[:100],flush=True)
pd.DataFrame(rows).sort_values('S1',ascending=False).to_parquet(config.RUNS_DIR/'ki_metrics_final.parquet',index=False)
print('done',flush=True)
