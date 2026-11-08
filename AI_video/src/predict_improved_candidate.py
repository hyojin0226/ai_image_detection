"""Individual-video inference for v2; never evaluates the previously used Test."""
import argparse
import json
from pathlib import Path
import numpy as np
import baseline_common
from video_pipeline import ROOT,file_sha256
from improve_existing_features import pooled,predict

RELEASE=ROOT/'models/development_improved_v2'


def verify():
    lock=json.loads((RELEASE/'release_lock.json').read_text(encoding='utf-8'))
    if file_sha256(RELEASE/'frozen.json')!=lock['frozen_json_sha256']:raise ValueError('Frozen v2 metadata changed')
    frozen=json.loads((RELEASE/'frozen.json').read_text(encoding='utf-8'))
    for name,value in lock['member_sha256'].items():
        if file_sha256(RELEASE/name)!=value:raise ValueError('Frozen v2 weights changed')
    for name,value in frozen['runtime_source_sha256'].items():
        if file_sha256(ROOT/'src'/name)!=value:raise ValueError('Frozen runtime changed: '+name)
    from image_model_backbone import source_identity
    if source_identity()!=frozen['feature_contract']['frozen_image_files']:raise ValueError('Image encoder changed')
    return frozen


def from_features(features):
    frozen=verify();values=[]
    for member in frozen['selected']['members']:
        with np.load(RELEASE/member['file'],allow_pickle=False) as saved:model={k:saved[k].copy() for k in saved.files}
        x,_=pooled(features,member['spec']['representation']);values.append(predict(model,x,member['spec']))
    return np.mean(values,axis=0)


def predict_video(path):
    import tensorflow as tf
    from development_video_data import sample_development_frames
    from image_model_backbone import load_image_backbone
    path=path.resolve()
    if path.is_relative_to((ROOT/'data/sample/test').resolve()):raise ValueError('Previous Test is not an independent new-model test; use development checks only')
    if not path.is_relative_to(ROOT):raise ValueError('Place the clip inside the project')
    if not path.is_file() or not 0<path.stat().st_size<=90*1024*1024:raise ValueError('Expected a video under 90 MB')
    verify();row={'file_path':str(path.relative_to(ROOT)),'sha256':file_sha256(path)}
    frames,times,meta=sample_development_frames(row,8,256);encoder,_=load_image_backbone();values=[]
    for frame in frames:
        feature,real=encoder(tf.convert_to_tensor(frame[None],tf.float32))
        values.append(np.concatenate([feature.numpy()[0],[1-float(real.numpy()[0,0])]]))
    score=float(from_features(np.asarray(values,np.float32)[None])[0])
    return {'candidate':'development_improved_v2','score_fake':score,'threshold':.5,
        'prediction':'fake' if score>=.5 else 'real','timestamps_sec':times.tolist(),
        'duration_sec':float(meta['duration']),'independent_test_performed':False,'adopted':False}


if __name__=='__main__':
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--video',type=Path);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:print(json.dumps({'verified':bool(verify()),'test_evaluated':False}))
    elif args.video:print(json.dumps(predict_video(args.video),ensure_ascii=False,indent=2))
    else:parser.error('Select --verify-only or --video')
