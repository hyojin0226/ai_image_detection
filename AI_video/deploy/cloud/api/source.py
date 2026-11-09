"""Fetch public YouTube media through its normal downloader; no inference."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,StreamingResponse
from starlette.background import BackgroundTask
BASE=Path(__file__).resolve().parents[1]
ORIGIN='https://duhyeon-ai-video.vercel.app'
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=[ORIGIN],allow_methods=['POST'],allow_headers=['Content-Type'])

def download(folder,url):
    shutil.copytree(BASE/'frozen/src',folder/'src')
    (folder/'.local').mkdir()
    env=dict(os.environ,PATH=str(BASE/'vendor')+os.pathsep+os.environ.get('PATH',''))
    result=subprocess.run([sys.executable,str(BASE/'source-worker.py'),str(folder),url],
        capture_output=True,text=True,env=env,timeout=100)
    if result.returncode:
        print(result.stderr[-1200:],flush=True)
        failure=folder/'download-error.json'
        return None,json.loads(failure.read_text()) if failure.exists() else {'error':'공개 영상을 가져오지 못했습니다.','code':'download_failed'}
    metadata=json.loads((folder/'download.json').read_text())
    path=Path(metadata['file'])
    if not path.resolve().is_relative_to(folder.resolve()) or not 0<path.stat().st_size<=90*1024**2:raise ValueError('Invalid downloaded file')
    return path,None

@app.post('/{path:path}')
async def source(path:str,request:Request):
    if request.headers.get('origin')!=ORIGIN:return JSONResponse({'error':'허용되지 않은 요청입니다.'},status_code=403)
    body=await request.body()
    if len(body)>4096:return JSONResponse({'error':'주소가 너무 깁니다.'},status_code=413)
    try:
        value=json.loads(body)['url'];parts=urlsplit(value)
        if not isinstance(value,str) or len(value)>2048 or parts.scheme!='https' or parts.hostname not in ('youtube.com','www.youtube.com','m.youtube.com','youtu.be') or parts.username or parts.password or parts.port not in (None,443):raise ValueError()
    except (ValueError,KeyError,TypeError,AttributeError):return JSONResponse({'error':'공개 YouTube 영상 주소를 입력해 주세요.'},status_code=400)
    temporary=tempfile.TemporaryDirectory(prefix='browser_youtube_');folder=Path(temporary.name)
    try:
        video,error=await asyncio.to_thread(download,folder,value)
        if error:
            temporary.cleanup();return JSONResponse(error,status_code=422)
        media='video/webm' if video.suffix.lower()=='.webm' else 'video/mp4'
        def stream():
            try:
                with video.open('rb') as file:
                    while chunk:=file.read(65536):yield chunk
            finally:temporary.cleanup()
        return StreamingResponse(stream(),media_type=media,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'},background=BackgroundTask(temporary.cleanup))
    except Exception as error:
        temporary.cleanup();print(type(error).__name__+': '+str(error)[:500],flush=True)
        return JSONResponse({'error':'YouTube 원본을 가져오지 못해 자동 분석을 시작하지 못했습니다.','code':'source_failed'},status_code=502)
