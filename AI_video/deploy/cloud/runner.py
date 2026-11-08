"""One isolated request, original frozen video logic, memory-bounded graph loader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
os.environ['MALLOC_ARENA_MAX']='2'
import ctypes
import gc
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
import image_model_backbone as backbone
from compact_backbone import build
bundle=None
def shared_loader(include_gradcam=False):
    global bundle
    if bundle is None:
        bundle=build(backbone);gc.collect()
        library=ctypes.CDLL(None)
        if hasattr(library,'malloc_trim'):library.malloc_trim(0)
    return bundle if include_gradcam else bundle[:2]
backbone.load_image_backbone=shared_loader
from analyze_64frames import analyze
try:
    analyze(sys.argv[2],root/'.local/result.json')
except Exception as error:
    message=str(error)[:500]
    if sys.argv[2].startswith('https://'):
        message='YouTube에서 영상을 가져오지 못했거나 지원 조건에 맞지 않습니다. 공개 영상도 서버 접근이 제한될 수 있으니 영상 파일로 입력해 주세요.'
    (root/'.local/cloud-error.json').write_text(json.dumps({'error':message},ensure_ascii=False))
    raise
