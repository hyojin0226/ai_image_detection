"""Bounded, streamed cloud inference. Uploaded inputs remain private."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
os.environ['MALLOC_ARENA_MAX']='2'
import base64
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

BASE=Path(__file__).resolve().parents[1]
ORIGIN='https://duhyeon-ai-video.vercel.app'
MAX_BYTES=90*1024*1024
PATH=re.compile(r'inputs/(\d{13})-[a-f0-9]{32}\.mp4')
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=[ORIGIN],allow_methods=['GET','POST'],
                   allow_headers=['Content-Type'])

def event(kind,**values):
    return json.dumps({'type':kind,**values},ensure_ascii=False,allow_nan=False)+'\n'

@app.get('/{path:path}')
def health(path:str):
    return {'status':'online','mode':'segmented64_v1','hosting':'cloud','max_bytes':MAX_BYTES}

def public_result(result):
    result.pop('preview_directory',None)
    for frame in result['frames']:
        image=Path(frame['comparison_path'])
        frame['comparison_url']='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode('ascii')
        for key in ('original_path','comparison_path','gradcam_path'):frame.pop(key,None)
    if result['source'].get('kind')=='file':result['source']['title']='업로드한 영상'
    return result

def execute(payload):
    gate=open('/tmp/ai-video-analysis.lock','a+b')
    pathname=payload.get('pathname') if payload['kind']=='file' else None
    child=None
    acquired=False
    try:
        try:
            fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
        except BlockingIOError:
            yield event('error',message='다른 영상을 분석 중입니다. 잠시 후 다시 시도해 주세요.')
            return
        yield event('progress',message='온라인 서버에서 모델을 준비하고 있습니다.')
        with tempfile.TemporaryDirectory(prefix='ai_video_cloud_') as tmp:
            root=Path(tmp)
            shutil.copytree(BASE/'frozen',root,dirs_exist_ok=True)
            local=root/'.local';local.mkdir(exist_ok=True)
            source=payload.get('url')
            if pathname:
                from vercel.blob import sync as blob
                source=str(local/'upload.mp4')
                yield event('progress',message='비공개 업로드 파일을 확인하고 있습니다.')
                total=0
                with blob.stream_blob(pathname,access='private') as download:
                    if not 0<download.metadata.size<=MAX_BYTES:raise ValueError('90 MB 이하의 영상이 필요합니다.')
                    with open(source,'wb') as output:
                        for chunk in download:
                            total+=len(chunk)
                            if total>MAX_BYTES:raise ValueError('90 MB 제한을 넘었습니다.')
                            output.write(chunk)
                if not total:raise ValueError('비어 있는 영상입니다.')
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
            # Node is used only for the normal public YouTube extractor.
            node=BASE/'vendor/node'
            if node.is_file():env['PATH']=str(node.parent)+os.pathsep+env.get('PATH','')
            with (local/'worker.log').open('w+') as errors:
                child=subprocess.Popen([sys.executable,str(BASE/'runner.py'),str(root),source],
                    stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1,env=env)
                for line in child.stdout:
                    line=line.strip()
                    if line:yield event('progress',message=line[:500])
                code=child.wait(timeout=5)
                if code:
                    errors.seek(0);print(errors.read()[-1800:],flush=True)
                    message='영상 분석을 완료하지 못했습니다. 파일 형식과 길이를 확인해 주세요.'
                    failure=local/'cloud-error.json'
                    if failure.exists():message=json.loads(failure.read_text())['error']
                    if code==-9:message='무료 서버의 메모리 한도를 넘었습니다. 잠시 후 다시 시도해 주세요.'
                    yield event('error',message=message)
                    return
            result=json.loads((local/'result.json').read_text())
            yield event('result',result=public_result(result))
    except GeneratorExit:
        raise
    except Exception as error:
        print(type(error).__name__+': '+str(error)[:700],flush=True)
        yield event('error',message='파일 전송 또는 분석에 실패했습니다. 무료 이용 한도와 입력 영상을 확인해 주세요.')
    finally:
        if child and child.poll() is None:
            child.kill();child.wait()
        if pathname:
            try:
                from vercel.blob import sync as blob
                blob.delete_blob(pathname)
            except Exception as error:print('Input cleanup deferred: '+type(error).__name__,flush=True)
        if acquired:fcntl.flock(gate,fcntl.LOCK_UN)
        gate.close()

@app.post('/{path:path}')
async def analyze(path:str,request:Request):
    if request.headers.get('origin')!=ORIGIN:return JSONResponse({'error':'허용되지 않은 요청입니다.'},status_code=403)
    body=await request.body()
    if len(body)>4096:return JSONResponse({'error':'입력 요청이 너무 큽니다.'},status_code=413)
    try:
        payload=json.loads(body)
        if payload.get('kind')=='file':
            match=PATH.fullmatch(payload.get('pathname',''))
            if not match or not -60_000<int(time.time()*1000)-int(match[1])<900_000:raise ValueError()
        elif payload.get('kind')=='youtube':
            from urllib.parse import urlsplit
            url=payload.get('url','');parts=urlsplit(url)
            if len(url)>2048 or parts.scheme!='https' or parts.hostname not in ('youtube.com','www.youtube.com','m.youtube.com','youtu.be') or parts.username or parts.password or parts.port not in (None,443):raise ValueError()
        else:raise ValueError()
    except (ValueError,TypeError,AttributeError):
        return JSONResponse({'error':'영상 파일이나 공개 Shorts 주소를 다시 입력해 주세요.'},status_code=400)
    return StreamingResponse(execute(payload),media_type='application/x-ndjson',headers={
        'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','X-Accel-Buffering':'no'})
