"""One normal public-video fetch. No model inference, cookies or proxy."""
import json
import os
from pathlib import Path
import sys
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
from shorts_worker import download
try:
    path,source=download(sys.argv[2],root/'.local',lambda stage,message:None,max_duration_sec=40)
    (root/'download.json').write_text(json.dumps({'file':str(path),'source':source},ensure_ascii=False))
except Exception as error:
    text=str(error)
    restricted=any(x in text.lower() for x in ['not a bot','sign in','403','429','private video','members-only'])
    message='YouTube가 온라인 서버에 원본 영상을 제공하지 않아 주소 자동 분석을 진행할 수 없습니다.' if restricted else 'YouTube 영상을 가져오지 못했습니다. 공개 상태와 3~40초 세로 영상인지 확인해 주세요.'
    (root/'download-error.json').write_text(json.dumps({'error':message,'code':'youtube_access_restricted' if restricted else 'youtube_download_failed'},ensure_ascii=False))
    print(type(error).__name__+': '+text[:900],file=sys.stderr)
    raise SystemExit(1)
