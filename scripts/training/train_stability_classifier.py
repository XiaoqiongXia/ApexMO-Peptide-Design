#!/usr/bin/env python3
"""Frozen 1-hour blood-stability classifier from the approved plan amendment."""
from __future__ import annotations
import glob, json, pickle, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, matthews_corrcoef, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'outputs/custom_scorers_stage_a_v1'; OUT=ROOT/'outputs/custom_scorers_stage_c_stability_classifier_v1'; OUT.mkdir(parents=True,exist_ok=True)
EMB=ROOT/'outputs/custom_scorers_stage_b_v1'

def load_data():
    raw=pd.read_csv(DATA/'stability_normalized.csv'); raw=raw[raw.formal_inclusion].copy()
    raw['species_raw']=raw['Test Sample'].astype(str).str.strip().str.lower()
    raw['species']=np.select([raw.species_raw.str.contains('human'),raw.species_raw.str.contains('rat'),raw.species_raw.str.contains('mouse|mice')],['human','rat','mouse'],'unknown')
    raw['matrix']=np.select([raw.species_raw.str.contains('plasma'),raw.species_raw.str.contains('serum')],['plasma','serum'],'unknown')
    raw['stable_label_1h']=pd.to_numeric(raw['T1/2(hour)'],errors='coerce').ge(1).astype(int)
    # Condition-aware median aggregation.
    d=raw.groupby(['sequence_sha256','sequence','species','matrix'],as_index=False).agg(
        target_hours=('T1/2(hour)','median'), stable_label_1h=('stable_label_1h','first'))
    d['weight']=1/d.groupby('sequence_sha256')['sequence_sha256'].transform('size')
    clusters=pd.read_csv(DATA/'mmseqs/stability_formal_cluster.tsv',sep='\t',header=None,names=['rep','member'])
    cmap=dict(zip(clusters.member,clusters.rep)); d['cluster']=d.sequence_sha256.map(cmap).fillna(d.sequence_sha256)
    # Existing stability embeddings were generated from this sequence-level order.
    seq_order=(raw[['sequence_sha256','sequence']].drop_duplicates('sequence_sha256').sort_values('sequence_sha256').reset_index(drop=True))
    X=np.concatenate([np.load(p) for p in sorted(glob.glob(str(EMB/'stab_batch_*.npy')))])
    if len(X)!=len(seq_order): raise RuntimeError(f'embedding rows {len(X)} != sequence reference {len(seq_order)}')
    em=pd.DataFrame(X,index=seq_order.sequence_sha256)
    d=d.join(em,on='sequence_sha256',how='left',rsuffix='_emb')
    if d.iloc[:,-640:].isna().any().any(): raise RuntimeError('missing stability embeddings')
    return d, X

def design(frame, cols):
    # Explicitly scale continuous ESM+length and one-hot conditions inside fold.
    n=640; emb_cols=list(range(n)); cont=emb_cols+['length']; cat=['species','matrix']
    prep=ColumnTransformer([('cont',StandardScaler(),cont),('cat',OneHotEncoder(handle_unknown='ignore'),cat)],remainder='drop')
    return prep

def make_frame(d):
    X=np.asarray(d.iloc[:,-640:].to_numpy(),dtype=float); out=pd.DataFrame(X,index=d.index,columns=[f'e{i}' for i in range(640)])
    out['length']=d.sequence.str.len().to_numpy(); out['species']=d.species.to_numpy(); out['matrix']=d.matrix.to_numpy(); out['aa_comp']=d.sequence.map(lambda s:[s.count(a)/len(s) for a in 'ACDEFGHIKLMNPQRSTVWY']).to_numpy()
    return out

def fit_model(X, y, weights, C, cw):
    p=Pipeline([('prep',ColumnTransformer([('cont',StandardScaler(),[f'e{i}' for i in range(640)]+['length']),('cat',OneHotEncoder(handle_unknown='ignore'),['species','matrix'])])),('model',LogisticRegression(C=C,class_weight=cw,max_iter=5000))])
    p.fit(X,y,model__sample_weight=weights); return p

def evaluate_seed(d, X, seed, prefix):
    y=d.stable_label_1h.to_numpy(); g=d.cluster.to_numpy(); w=d.weight.to_numpy(); o=np.full(len(d),np.nan); chosen=[]
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    for fold,(tr,va) in enumerate(cv.split(X,y,g)):
        best=None
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=seed+1000+fold)
        for C in [1e-4,1e-3,1e-2,.1,1,10,100]:
            for cw in [None,'balanced']:
                aps=[]
                for it,iv in inner.split(X.iloc[tr],y[tr],g[tr]):
                    p=fit_model(X.iloc[tr].iloc[it],y[tr][it],w[tr][it],C,cw); s=p.predict_proba(X.iloc[tr].iloc[iv])[:,1]; aps.append(average_precision_score(y[tr][iv],s,sample_weight=w[tr][iv]))
                score=float(np.mean(aps)); key=(score,-C,0 if cw is None else -1)
                if best is None or key>best[0]: best=(key,C,cw)
        p=fit_model(X.iloc[tr],y[tr],w[tr],best[1],best[2]); o[va]=p.predict_proba(X.iloc[va])[:,1]; chosen.append({'fold':fold,'C':best[1],'class_weight':best[2]})
    # Threshold from this seed's own pooled OOF scores; scores are uncalibrated.
    order=np.unique(o[~np.isnan(o)]); candidates=[]
    for t in order:
        pred=o>=t; pr=precision_score(y,pred,sample_weight=w,zero_division=0); rec=recall_score(y,pred,sample_weight=w,zero_division=0)
        if pr>=.90 and rec>=.20: candidates.append((rec,t,pr))
    threshold=float(max(candidates)[1]) if candidates else None
    pred=(o>=threshold) if threshold is not None else np.zeros(len(y),dtype=bool)
    metrics={'seed':seed,'task':prefix,'n_rows':int(len(d)),'n_sequences':int(d.sequence_sha256.nunique()),'stable_prevalence':float(np.average(y,weights=w)),'auroc':float(roc_auc_score(y,o,sample_weight=w)),'auprc':float(average_precision_score(y,o,sample_weight=w)),'threshold':threshold,'precision':float(precision_score(y,pred,sample_weight=w,zero_division=0)),'recall':float(recall_score(y,pred,sample_weight=w,zero_division=0)),'mcc':float(matthews_corrcoef(y,pred,sample_weight=w)),'confusion_matrix':confusion_matrix(y,pred,sample_weight=w).tolist(),'threshold_available':threshold is not None}
    np.savez_compressed(OUT/f'{prefix}_oof_seed{seed}.npz',y=y,score=o,weight=w)
    return metrics,chosen

if __name__=='__main__':
    d,_=load_data(); X=make_frame(d); results=[]; choices={}
    for seed in [42,123,2025]:
        m,c=evaluate_seed(d,X,seed,'stability_classifier'); results.append(m); choices[str(seed)]=c
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2)); (OUT/'fold_model_choices.json').write_text(json.dumps(choices,indent=2)); (OUT/'data_manifest.json').write_text(json.dumps({'rows':len(d),'unique_sequences':int(d.sequence_sha256.nunique()),'stable_rows_weighted':float(d.groupby('sequence_sha256').stable_label_1h.first().sum()),'stable_label_definition':'T1/2 >= 1 hour','seeds':[42,123,2025]},indent=2)); print(json.dumps(results,indent=2))
