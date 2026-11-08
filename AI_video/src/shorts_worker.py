"""One bounded local job: public YouTube/file -> final model -> safe result JSON."""
import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from video_pipeline import ROOT,extract_frames,file_sha256

MAX_BYTES=90*1024*1024
RELEASE=ROOT/'models/final_v1'

def canonical_youtube_url(value):
    if not isinstance(value,str) or len(value)>2048:raise ValueError('유튜브 Shorts 주소를 입력해주세요.')
    parsed=urlsplit(value.strip())
    if parsed.scheme not in ('https','http') or parsed.username or parsed.password or parsed.port not in (None,80,443):raise ValueError('올바른 유튜브 주소가 아닙니다.')
    host=(parsed.hostname or '').lower()
    if host=='youtu.be':identity=parsed.path.strip('/')
    elif host in ('youtube.com','www.youtube.com','m.youtube.com'):
        if parsed.path.startswith('/shorts/'):identity=parsed.path[len('/shorts/'):].rstrip('/')
        elif parsed.path=='/watch':identity=parse_qs(parsed.query).get('v',[''])[0]
        else:raise ValueError('개별 Shorts 또는 유튜브 영상 주소를 입력해주세요.')
    else:raise ValueError('youtube.com 또는 youtu.be 주소만 지원합니다.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',identity):raise ValueError('영상 주소의 ID가 올바르지 않습니다.')
    return 'https://www.youtube.com/watch?v='+identity

def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temporary.replace(path)

class InputError(Exception):pass

def download(url,folder,progress,*,max_duration_sec=30):
    import yt_dlp
    canonical=canonical_youtube_url(url)
    node=shutil.which('node')
    runtime=ROOT/'configs/local_runtime.json'
    if not node and runtime.exists():
        configured=json.loads(runtime.read_text(encoding='utf-8')).get('node')
        if configured and Path(configured).is_file():node=configured
    last=[0.0]
    def hook(data):
        if data.get('downloaded_bytes',0)>MAX_BYTES:raise InputError('영상 파일이 90 MB 제한을 넘었습니다.')
        now=time.monotonic()
        if now-last[0]>1:
            progress('downloading','유튜브 영상을 가져오고 있습니다.');last[0]=now
    def validate(info,*,incomplete=False):
        duration=info.get('duration')
        if duration is not None and (not isinstance(duration,(int,float)) or not 3<=duration<=max_duration_sec):return f'3~{max_duration_sec}초 영상만 분석할 수 있습니다.'
        if info.get('is_live'):return '실시간 방송은 분석할 수 없습니다.'
        width,height=info.get('width'),info.get('height')
        if width and height and height<=width:return '세로 영상만 분석할 수 있습니다.'
        if info.get('filesize',0) and info['filesize']>MAX_BYTES:return '영상 파일이 90 MB 제한을 넘었습니다.'
    options={'quiet':True,'no_warnings':True,'noprogress':True,'noplaylist':True,'socket_timeout':15,'retries':1,'extractor_retries':1,
             'fragment_retries':1,'concurrent_fragment_downloads':1,'cachedir':False,'max_filesize':MAX_BYTES,
             'outtmpl':str(folder/'input.%(ext)s'),'overwrites':False,'progress_hooks':[hook],
             'format':'bestvideo[height<=1280][ext=mp4]/best[height<=1280][ext=mp4]/bestvideo[height<=1280]/best[height<=1280]',
             'js_runtimes':{'node':{'path':node}} if node else {},'playlistend':1}
    with yt_dlp.YoutubeDL(options) as downloader:
        progress('checking','공개 영상의 길이와 형식을 확인하고 있습니다.')
        info=downloader.extract_info(canonical,download=False)
        if not info or info.get('_type') in ('playlist','multi_video'):raise InputError('개별 영상 주소가 필요합니다.')
        reason=validate(info)
        if reason:raise InputError(reason)
        duration=info.get('duration')
        if not isinstance(duration,(int,float)) or not math.isfinite(duration):raise InputError('영상 길이를 확인할 수 없습니다. 파일로 업로드해주세요.')
        progress('downloading','유튜브 영상을 가져오고 있습니다.')
        # Use the already extracted public media information; no second lookup.
        downloader.process_info(info)
        paths=[p for p in folder.glob('input.*') if p.suffix not in ('.part','.ytdl','.tmp')]
        if len(paths)!=1 or paths[0].stat().st_size>MAX_BYTES:raise InputError('영상을 완전히 가져오지 못했습니다. 파일로 업로드해주세요.')
        return paths[0],{'kind':'youtube','url':canonical,'title':str(info.get('title') or 'YouTube Shorts')[:300]}

def run(folder,url=None):
    folder=folder.resolve();started=time.perf_counter();download_complete=False
    def progress(stage,message):write_json(folder/'state.json',{'status':'running','stage':stage,'message':message})
    try:
        if url:path,source=download(url,folder,progress)
        else:
            path=folder/'input.mp4';source={'kind':'file','title':json.loads((folder/'request.json').read_text(encoding='utf-8')).get('filename','업로드한 영상')}
        download_complete=True
        if not path.exists() or not 0<path.stat().st_size<=MAX_BYTES:raise InputError('영상 파일 크기를 확인해주세요. 최대 90 MB입니다.')
        release=json.loads((RELEASE/'release.json').read_text(encoding='utf-8'))
        progress('preparing','분석할 장면 8개를 고르고 있습니다.')
        try:frames,timestamps,video=extract_frames(path,release['preprocessing'])
        except ValueError as error:raise InputError('3~30초 세로 영상인지 확인해주세요. 손상된 파일은 분석할 수 없습니다.') from error
        for name,digest in release['files_sha256'].items():
            target=(RELEASE/name).resolve()
            if not target.is_relative_to(RELEASE.resolve()) or file_sha256(target)!=digest:raise RuntimeError('Model release integrity failure')
        progress('loading','노트북에서 영상 모델을 준비하고 있습니다.')
        import baseline_common  # CPU thread and runtime defaults before TensorFlow import.
        import tensorflow as tf
        from PIL import Image
        tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
        model=tf.saved_model.load(str(RELEASE/'saved_model'))
        progress('analyzing','선택한 장면과 시간 흐름을 분석하고 있습니다.')
        value=model.signatures['serving_default'](frames_rgb=tf.convert_to_tensor(frames[None]))
        score=float(value['fake_score'].numpy()[0,0]);image_scores=value['frame_fake_scores'].numpy()[0]
        if not math.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid model score')
        previews=[]
        for index,(frame,timestamp,frame_score) in enumerate(zip(frames,timestamps,image_scores)):
            Image.fromarray(frame).save(folder/f'frame_{index}.jpg',quality=88)
            previews.append({'index':index,'timestamp_sec':float(timestamp),'image_fake_score':float(frame_score)})
        result={'status':'completed','release':'final_v1','model':release['model'],'source':source,
                'prediction':'fake' if score>=release['threshold'] else 'real','fake_score':score,'threshold':release['threshold'],
                'score_is_calibrated_probability':False,'experimental':True,'duration_sec':video['duration'],
                'width':video['size'][0],'height':video['size'][1],'frames':previews,'elapsed_sec':round(time.perf_counter()-started,2),
                'limitation':'시험용 모델의 추정입니다. AI 점수는 확률이 아니며 일부 장면만 합성된 영상과 새로운 생성 방식에서는 틀릴 수 있습니다.'}
        write_json(folder/'result.json',result);write_json(folder/'state.json',{'status':'completed','stage':'done','message':'분석을 마쳤습니다.','result':result})
    except InputError as error:write_json(folder/'state.json',{'status':'failed','code':'unsupported_input','message':str(error)})
    except Exception as error:
        print(type(error).__name__+': '+str(error),flush=True)
        message='유튜브에서 영상을 가져오지 못했습니다. 삭제·로그인·지역·일시적 접근 제한이 있는지 확인하거나 영상 파일을 업로드해주세요.' if url and not download_complete else '분석을 완료하지 못했습니다. 모델 파일과 영상 형식을 확인해주세요.'
        write_json(folder/'state.json',{'status':'failed','code':'analysis_failed','message':message})
    finally:
        for candidate in folder.glob('input.*'):
            if candidate.is_file():candidate.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job-dir',type=Path,required=True);parser.add_argument('--url');args=parser.parse_args();run(args.job_dir,args.url)
