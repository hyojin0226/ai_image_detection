"""Existing train/valid features only: temporal pooling and regularized kernels.

No test predictions, no new collection and no edits to evaluated frozen models.
Hyperparameters are shortlisted by grouped training-only CV, then validation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import train_development_gru as original
from baseline_common import metrics
from video_pipeline import ROOT,file_sha256,save_npz

BASE=ROOT/'checkpoints/development_gru_v1/20261107T084400_716762Z'
OUT=ROOT/'checkpoints/existing_features_improvement_v2'
RELEASE=ROOT/'models/development_improved_v2'
SPECS=[
    {'name':'temporal_linear_1','representation':'temporal','kernel':'linear','alpha':1.,'gamma':0.},
    {'name':'temporal_linear_10','representation':'temporal','kernel':'linear','alpha':10.,'gamma':0.},
    {'name':'temporal_linear_100','representation':'temporal','kernel':'linear','alpha':100.,'gamma':0.},
    {'name':'temporal_rbf_01_02','representation':'temporal','kernel':'rbf','alpha':.1,'gamma':.2},
    {'name':'temporal_rbf_1_02','representation':'temporal','kernel':'rbf','alpha':1.,'gamma':.2},
    {'name':'temporal_rbf_01_05','representation':'temporal','kernel':'rbf','alpha':.1,'gamma':.5},
    {'name':'temporal_rbf_1_05','representation':'temporal','kernel':'rbf','alpha':1.,'gamma':.5},
    {'name':'mean_rbf_01_05','representation':'mean','kernel':'rbf','alpha':.1,'gamma':.5},
]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):original.dump(p,v)


def pooled(sequence,representation):
    a=np.asarray(sequence,np.float64)
    if a.ndim!=3 or a.shape[1:]!=(8,257) or not np.isfinite(a).all():raise ValueError('Expected finite N x 8 x 257 features')
    embedding=a[:,:,:256];score=a[:,:,-1]
    stats=np.stack([score.mean(axis=1),score.std(axis=1),score.min(axis=1),score.max(axis=1)],axis=1)
    parts=[embedding.mean(axis=1)]
    blocks=[256]
    if representation=='temporal':
        parts.extend([embedding.std(axis=1),np.abs(np.diff(embedding,axis=1)).mean(axis=1)])
        blocks.extend([256,256])
    elif representation!='mean':raise ValueError('Unknown representation')
    parts.append(stats);blocks.append(4)
    return np.concatenate(parts,axis=1),blocks


def group_folds(rows,n=3):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['group_id']].append(i)
    result=np.full(len(rows),-1,int);counts=np.zeros((n,2),int)
    for label in [0,1]:
        selected=[]
        for g,ids in groups.items():
            labels={int(rows[i]['label_id']) for i in ids}
            if len(labels)!=1:raise ValueError('A training group mixes labels')
            if labels=={label}:selected.append((g,ids))
        selected.sort(key=lambda v:(-len(v[1]),hashlib.sha256(('20261108'+v[0]).encode()).hexdigest()))
        for group,ids in selected:
            fold=min(range(n),key=lambda k:(counts[k,label],counts[k].sum(),k))
            result[ids]=fold;counts[fold,label]+=len(ids)
    if np.any(result<0) or np.any(counts==0):raise ValueError('Invalid grouped CV allocation')
    for g,ids in groups.items():
        if len(set(result[ids]))!=1:raise ValueError('Group leakage')
    return result


def kernel(a,b,kind,gamma):
    product=a@b.T
    if kind=='linear':return product
    if kind!='rbf':raise ValueError('Unknown kernel')
    distance=np.maximum(np.sum(a*a,axis=1)[:,None]+np.sum(b*b,axis=1)[None,:]-2*product,0)
    return np.exp(-gamma*distance)


def fit(x,y,spec,blocks):
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-3)
    block=np.concatenate([np.full(size,1/np.sqrt(size)) for size in blocks])
    z=np.clip((x-mean)/scale,-10,10)*block
    weights=np.array([len(y)/(2*np.sum(y==c)) for c in y]);sw=np.sqrt(weights)
    k=kernel(z,z,spec['kernel'],spec['gamma'])
    coefficient=sw*np.linalg.solve(sw[:,None]*k*sw[None,:]+spec['alpha']*np.eye(len(y)),sw*(y-.5))
    return {'mean':mean,'scale':scale,'block':block,'support':z,'coefficient':coefficient}


def predict(model,x,spec):
    z=np.clip((x-model['mean'])/model['scale'],-10,10)*model['block']
    residual=kernel(z,model['support'],spec['kernel'],spec['gamma'])@model['coefficient']
    return 1/(1+np.exp(-np.clip(4*residual,-60,60)))


def rank(item):return (-item['metrics']['accuracy'],item['metrics']['false_positive_rate'],item['bce'],item['name'])


def protected_files():
    names=['data/sample/train.csv','data/sample/valid.csv','data/sample/test.csv','models/pretest_selected_v1/frozen.json',
        'models/pretest_selected_v1/video_head.keras','models/pretest_selected_v1/release_lock.json','test_latest_model.bat',
        'checkpoints/adoption_augmented_v1/residual.keras','reports/evaluation/frozen_test1000_v1/results.json',
        'reports/evaluation/frozen_test1000_v1/completion_lock.json','reports/evaluation/frozen_test1000_v1/predictions.csv']
    # Only hashes are read for the old Test files, never scores or labels.
    return {p:file_sha256(ROOT/p) for p in names if (ROOT/p).is_file()}


def execute():
    if OUT.exists():raise ValueError('Preserve existing v2 experiment; inspect results instead of repeating selection')
    cfg=original.configuration();rows,info=original.preflight(cfg);protocol=read(BASE/'protocol.json')
    if protocol['feature_contract']!=info['feature_contract']:raise ValueError('Cached feature contract changed')
    protected=protected_files();OUT.mkdir(parents=True)
    plan={'purpose':'Improve only existing development data; no Test feedback or rescoring',
        'candidates':SPECS,'cv':'3-fold grouped and class-stratified within train; normalization fit separately per fold',
        'shortlist':'Top 2 by out-of-fold balanced accuracy, then FPR then BCE',
        'validation_candidates':'Two shortlisted models plus fixed equal average; compare with unchanged previous validation candidate',
        'selection':'Fixed threshold .5; require accuracy >.795, real FPR<=.18, BCE<=.5080758; rank accuracy then FPR then BCE',
        'fallback':'Retain old model if no qualifying validation improvement',
        'feature_contract':protocol['feature_contract'],'protected_sha256':protected,
        'code_sha256':file_sha256(Path(__file__)),'no_collection':True,'test_predictions_prohibited':True}
    dump(OUT/'plan.json',plan)
    data={};started=time.monotonic()
    for split,part in rows.items():
        features=[]
        for r in part:
            if file_sha256(ROOT/r['file_path'])!=r['sha256']:raise ValueError('Development video changed')
            features.append(original.read_feature(original.cache_path(r,protocol['feature_contract']),r,protocol['feature_contract']))
        data[split]=(np.asarray(features,np.float32),np.array([int(r['label_id']) for r in part],np.float64))
        print('VERIFIED DEVELOPMENT FEATURES',split,len(part),flush=True)
    folds=group_folds(rows['train'])
    dump(OUT/'group_folds.json',[{'video_id':r['video_id'],'group_id':r['group_id'],'fold':int(f)} for r,f in zip(rows['train'],folds)])
    cv=[]
    for spec in SPECS:
        x,blocks=pooled(data['train'][0],spec['representation']);y=data['train'][1];scores=np.empty(len(y))
        for fold in range(3):
            mask=folds!=fold;model=fit(x[mask],y[mask],spec,blocks)
            scores[~mask]=predict(model,x[~mask],spec)
        record={'name':spec['name'],'spec':spec,'metrics':metrics(y,scores),'bce':original.bce(y,scores)}
        cv.append(record);dump(OUT/'training_group_cv.json',cv)
        print('GROUP CV',spec['name'],f'BA={record["metrics"]["accuracy"]:.1%}',f'FPR={record["metrics"]["false_positive_rate"]:.1%}',flush=True)
    shortlist=sorted(cv,key=rank)[:2];dump(OUT/'shortlist_before_validation.json',shortlist)
    scores=[];members=[];choices=[]
    for record in shortlist:
        spec=record['spec'];x,blocks=pooled(data['train'][0],spec['representation']);vx,_=pooled(data['valid'][0],spec['representation'])
        model=fit(x,data['train'][1],spec,blocks);name=spec['name']+'.npz';save_npz(OUT/name,**model)
        with np.load(OUT/name,allow_pickle=False) as saved:restored={k:saved[k].copy() for k in saved.files}
        p=predict(restored,vx,spec);np.testing.assert_allclose(p,predict(model,vx,spec),atol=1e-12)
        scores.append(p);member={'name':spec['name'],'spec':spec,'file':name,'sha256':file_sha256(OUT/name)};members.append(member)
        choices.append({'name':spec['name'],'members':[member],'metrics':metrics(data['valid'][1],p),'bce':original.bce(data['valid'][1],p)})
        dump(OUT/(spec['name']+'_validation.json'),[{'video_id':r['video_id'],'label_id':int(r['label_id']),'score_fake':float(v)} for r,v in zip(rows['valid'],p)])
    average=np.mean(scores,axis=0)
    choices.append({'name':'shortlist_equal_average','members':members,'metrics':metrics(data['valid'][1],average),'bce':original.bce(data['valid'][1],average)})
    dump(OUT/'shortlist_equal_average_validation.json',[{'video_id':r['video_id'],'label_id':int(r['label_id']),'score_fake':float(v)} for r,v in zip(rows['valid'],average)])
    eligible=[c for c in choices if c['metrics']['accuracy']>.795 and c['metrics']['false_positive_rate']<=.18+1e-12 and c['bce']<=.5080758]
    selected=min(eligible,key=rank) if eligible else None
    result={'status':'development_comparison_complete','training_cv':cv,'validation_candidates':choices,'selected':selected,
        'test_evaluated':False,'previous_test_not_independent_for_future_improvements':True,'old_model_replaced':False,
        'elapsed_seconds':time.monotonic()-started,'no_new_media':True,'no_production_accuracy_claim':True}
    if protected_files()!=protected:raise ValueError('Protected data or frozen evaluation changed')
    dump(OUT/'results.json',result)
    if selected:
        if RELEASE.exists():raise ValueError('Preserve existing v2 release')
        RELEASE.mkdir(parents=True)
        import shutil
        for member in selected['members']:shutil.copy2(OUT/member['file'],RELEASE/member['file'])
        frozen={'status':'improved_development_candidate_not_independently_tested','selected':selected,
            'threshold':.5,'feature_contract':protocol['feature_contract'],'protected_sha256':protected,
            'runtime_source_sha256':{n:file_sha256(ROOT/'src'/n) for n in ['improve_existing_features.py','predict_improved_candidate.py',
                'train_development_gru.py','development_video_data.py','image_model_backbone.py','video_pipeline.py']},
            'sampling':'8 uniform centers; full aspect preserving square padding 256 RGB; original image backbone frozen',
            'test_evaluated':False,'adopted':False,'validation_selection_optimism':True,
            'limitations':['No independent test score for this candidate','The previous Test has already been viewed',
                'Eight frames can miss brief events','No paired long-duration or broad new-generator evidence']}
        dump(RELEASE/'frozen.json',frozen)
        dump(RELEASE/'release_lock.json',{'frozen_json_sha256':file_sha256(RELEASE/'frozen.json'),
            'member_sha256':{m['file']:m['sha256'] for m in selected['members']}})
        print('IMPROVED DEVELOPMENT CANDIDATE',selected['name'],'accuracy',selected['metrics']['accuracy'],'FPR',selected['metrics']['false_positive_rate'],'BCE',selected['bce'],flush=True)
    else:print('No qualifying validation improvement; old model retained.',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:execute()
    else:print('Preparation only. --execute runs train/valid improvement; never uses Test predictions.')
