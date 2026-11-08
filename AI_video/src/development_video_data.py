"""Separate development data reader; does not change the existing model runtime."""
import csv
from pathlib import Path
import subprocess
import numpy as np
from video_pipeline import ROOT,probe_video,imageio_ffmpeg,file_sha256


def load_split(split):
    if split not in ['train','valid']:raise ValueError('Development reader accepts train/valid only; test remains sealed')
    with (ROOT/'data/sample'/(split+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        if row['label'] not in ['real','fake'] or int(row['label_id'])!=int(row['label']=='fake'):raise ValueError('Invalid label')
        if row['split']!=split or row.get('dataset_qc_accepted')!='True':raise ValueError('Not an accepted development row')
        path=(ROOT/row['file_path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.exists():raise ValueError('Invalid development media path')
    return rows


def sample_development_frames(row,frames=8,size=224):
    """Decode full pictures with square padding, in either native orientation."""
    path=(ROOT/row['file_path']).resolve()
    if not path.is_relative_to(ROOT.resolve()) or file_sha256(path)!=row['sha256']:raise ValueError('Source changed')
    if not 1<=frames<=64 or not 64<=size<=512:raise ValueError('Unbounded development frame request')
    meta=probe_video(path);duration=float(meta['duration'])
    if not 3<=duration<=40:raise ValueError('Development scope is 3-40 seconds')
    times=(np.arange(frames,dtype=np.float64)+.5)*duration/frames
    result=np.empty((frames,size,size,3),dtype=np.uint8)
    vf=f'scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black,setsar=1'
    for i,timestamp in enumerate(times):
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-threads','1','-ss',f'{timestamp:.6f}',
            '-i',str(path),'-an','-frames:v','1','-vf',vf,'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
        process=subprocess.run(command,capture_output=True,timeout=60)
        if process.returncode or len(process.stdout)!=size*size*3:raise ValueError('Development frame decode failed')
        result[i]=np.frombuffer(process.stdout,dtype=np.uint8).reshape(size,size,3)
    return result,times,meta
