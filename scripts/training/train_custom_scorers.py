#!/usr/bin/env python3
"""Train frozen-ESM2 custom toxicity, hemolysis and stability scorers."""
from __future__ import annotations
import hashlib, json, pickle, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, EsmModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'outputs/custom_scorers_stage_a_v1'; OUT=ROOT/'outputs/custom_scorers_stage_b_v1'; OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'external/models/facebook/esm2_t30_150M_UR50D'; AA='ACDEFGHIKLMNPQRSTVWY'

def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def cluster_map(name):
 d=pd.read_csv(DATA/f'{name}.fasta',sep='\t',header=None) if False else None
 t=pd.read_csv(DATA/f'mmseqs/{name}_cluster.tsv',sep='\t',header=None,names=['rep','member'])
 return dict(zip(t.member,t.rep))

def embed(seqs):
 keys=np.array([hashlib.sha256(s.encode()).hexdigest() for s in seqs]); cache=OUT/f"esm2_embeddings_{hashlib.sha256('|'.join(keys).encode()).hexdigest()[:16]}.npz"
 if cache.exists():
  z=np.load(cache,allow_pickle=True)
  if np.array_equal(z['keys'],keys): return z['X']
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); mod=EsmModel.from_pretrained(MODEL,local_files_only=True).to(device).eval()
 out=[]
 with torch.inference_mode():
  for i in range(0,len(seqs),8):
   b={k:v.to(device) for k,v in tok(list(seqs[i:i+32]),return_tensors='pt',padding=True, truncation=True).items()}
   h=mod(**b).last_hidden_state; mask=b['attention_mask'].bool(); mask[:,0]=False
   for j in range(h.shape[0]):
    n=int(mask[j].sum()); out.append((h[j][mask[j]].mean(0) if n else h[j,0]).cpu().numpy())
 X=np.asarray(out,dtype='float32'); np.savez_compressed(cache,keys=keys,X=X); return X

def load():
 tox=pd.read_csv(DATA/'toxicity_normalized.csv'); tox=tox[tox.canonical].copy(); tox['cluster']=tox.apply(lambda r: cluster_map('toxicity_train').get(r.sequence_sha256,r.sequence_sha256) if r.split=='train' else r.sequence_sha256,axis=1)
 he=pd.read_csv(DATA/'hemolysis_normalized.csv'); he=he[he.canonical].copy(); he['cluster']=he.apply(lambda r: cluster_map('hemolysis_cross_val').get(r.sequence_sha256,r.sequence_sha256) if r.split=='cross_val' else r.sequence_sha256,axis=1)
 st=pd.read_csv(DATA/'stability_normalized.csv'); st=st[st.formal_inclusion].copy(); st=st.groupby('sequence_sha256',as_index=False).agg(sequence=('sequence','first'),target=('stability_target_log10_hours','median')); st['cluster']=st.sequence_sha256.map(cluster_map('stability_formal'))
 return tox,he,st

def cls(frame, split, labelcol, name, seed=42):
 d=frame[frame.split==split].copy(); X=embed(d.sequence.tolist()); y=d[labelcol].to_numpy(); g=d.cluster.to_numpy(); cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed); oof=np.zeros(len(d)); rows=[]; best=[]
 for k,(tr,va) in enumerate(cv.split(X,y,g)):
  scores=[]
  for C in [1e-4,1e-3,1e-2,.1,1,10,100]:
   p=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=C,max_iter=3000,class_weight='balanced' if name=='hemolysis' else None))]); p.fit(X[tr],y[tr]); pr=p.predict_proba(X[va])[:,1]; scores.append((average_precision_score(y[va],pr),-C,p))
  _,_,p=max(scores,key=lambda z:(z[0],z[1])); oof[va]=p.predict_proba(X[va])[:,1]; best.append(p)
 auc=roc_auc_score(y,oof); ap=average_precision_score(y,oof); thr=np.quantile(oof[y==1],.05); sens=((oof[y==1]>=thr).mean()); spec=((oof[y==0]<thr).mean())
 full=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=3000,class_weight='balanced' if name=='hemolysis' else None))]); full.fit(X,y)
 pickle.dump(full,open(OUT/f'{name}_model.pkl','wb')); np.savez_compressed(OUT/f'{name}_oof.npz',y=y,score=oof,threshold=thr)
 return {'task':name,'n':len(d),'auroc':auc,'auprc':ap,'threshold':float(thr),'sensitivity':float(sens),'specificity':float(spec)}

def stability(st,seed=42):
 X=embed(st.sequence.tolist()); y=st.target.to_numpy(); g=st.cluster.to_numpy(); cv=GroupKFold(5); oof=np.zeros(len(st))
 for tr,va in cv.split(X,y,g):
  best=None
  for a in [1e-4,1e-3,1e-2,.1,1,10,100,1000,10000]:
   p=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=a))]); p.fit(X[tr],y[tr]); e=mean_absolute_error(y[va],p.predict(X[va])); best=(e,p) if best is None or e<best[0] else best
  oof[va]=best[1].predict(X[va])
 full=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=1))]); full.fit(X,y); pickle.dump(full,open(OUT/'stability_model.pkl','wb')); np.savez_compressed(OUT/'stability_oof.npz',y=y,score=oof)
 return {'task':'stability','n':len(y),'spearman':float(spearmanr(y,oof).statistic),'mae':float(mean_absolute_error(y,oof))}

if __name__=='__main__':
 tox,he,st=load(); res=[cls(tox,'train','label','toxicity'),cls(he,'cross_val','label','hemolysis'),stability(st)]; (OUT/'metrics.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
