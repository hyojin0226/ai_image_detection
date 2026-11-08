"""Bounded full-frame preparation shared by training and future inference."""
from data_paths import resolve_data_path
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python'))
import imageio_ffmpeg
import numpy as np


def load_config():
    return json.loads((ROOT / 'configs/preprocessing.json').read_text(encoding='utf-8'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def manifest_rows(include_review=False):
    with (ROOT / 'data/metadata/video_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate video_id in manifest')
    allowed = {'approved', 'review'} if include_review else {'approved'}
    selected = [r for r in rows if r['status'] in allowed]
    for row in selected:
        if row['label'] not in ('real', 'fake') or int(row['label_id']) != {'real': 0, 'fake': 1}[row['label']]:
            raise ValueError('Inconsistent label: ' + row['video_id'])
        if not row['group_id']:
            raise ValueError('Missing source group: ' + row['video_id'])
    return selected


def video_path(row):
    path = (resolve_data_path(ROOT / row['file_path'])).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Video path outside project')
    return path


def probe_video(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        return next(reader)
    finally:
        reader.close()


def extract_frames(path, config, *, timestamps=None):
    meta = probe_video(path)
    duration = float(meta['duration'])
    width, height = meta['size']
    if height <= width or not config['min_duration_sec'] <= duration <= config['max_duration_sec']:
        raise ValueError(f"세로 영상이며 길이가 {config['min_duration_sec']}~{config['max_duration_sec']}초여야 합니다.")
    count, size = config['frames_per_video'], config['input_size']
    timestamps = ((np.arange(count, dtype=np.float64) + 0.5) * duration / count
                  if timestamps is None else np.asarray(timestamps,dtype=np.float64))
    if timestamps.ndim!=1 or not len(timestamps) or not np.isfinite(timestamps).all() or np.any((timestamps<0)|(timestamps>=duration)):
        raise ValueError('Invalid frame sampling positions')
    count=len(timestamps)
    frames = np.empty((count, size, size, 3), dtype=np.uint8)
    # One decoded RGB frame at a time; no complete video in memory.
    filter_text = f'scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black,setsar=1'
    for index, timestamp in enumerate(timestamps):
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin',
                   '-threads', str(config['ffmpeg_threads']), '-ss', f'{timestamp:.6f}',
                   '-i', str(path), '-an', '-frames:v', '1', '-vf', filter_text,
                   '-threads', str(config['ffmpeg_threads']), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        result = subprocess.run(command, capture_output=True, timeout=30)
        # At low FPS the last presentation frame remains visible until duration,
        # but seeking after its timestamp can return no frame. Re-seek its start
        # only for an empty successful decode inside that final frame interval.
        fps = float(meta.get('fps', 0))
        if result.returncode == 0 and not result.stdout and fps > 0 and timestamp >= duration - 1 / fps:
            command[command.index('-ss') + 1] = f'{max(0, duration - 1 / fps - .001):.6f}'
            result = subprocess.run(command, capture_output=True, timeout=30)
        if result.returncode or len(result.stdout) != size * size * 3:
            raise ValueError(f'Frame {index} failed at {timestamp:.3f}s: ' + result.stderr.decode(errors='replace')[:200])
        frames[index] = np.frombuffer(result.stdout, dtype=np.uint8).reshape(size, size, 3)
    return frames, timestamps, meta


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def cached_frames(row, config):
    path = ROOT / 'data/cache/processed/frames' / (row['video_id'] + '.npz')
    with np.load(path, allow_pickle=False) as data:
        expected = (config['frames_per_video'], config['input_size'], config['input_size'], 3)
        if str(data['source_sha256']) != row['sha256'] or str(data['config_sha256']) != config_sha256(config):
            raise ValueError('Stale frame cache: ' + row['video_id'])
        frames = data['frames']
        if frames.shape != expected or frames.dtype != np.uint8:
            raise ValueError('Invalid cache shape/dtype')
        return frames, data['timestamps_sec']
