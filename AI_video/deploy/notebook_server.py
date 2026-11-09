"""Bounded web API with duration-based sampling and unchanged frozen weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(__file__).parent / 'site'
MAX_BYTES = 90 * 1024 * 1024
TTL_SECONDS = 3600
FRAME_POLICY = 'duration_v1'


def frame_count_for_duration(duration):
    if not math.isfinite(duration) or not 3 <= duration <= 40:
        raise ValueError('3~40초 영상을 선택하세요.')
    return 16 if duration <= 10 else 32 if duration <= 20 else 48 if duration <= 30 else 64


def select_duration_segments(frames, times, duration, decode, is_black):
    """Keep eight unique nonblack timestamps per segment; never pad duplicates."""
    planned = frame_count_for_duration(duration)
    count = planned // 8
    if len(frames) != planned or len(times) != planned:
        raise ValueError('길이별 계획 프레임 수와 입력이 다릅니다.')
    groups, skipped, black = [], [], []
    for segment in range(count):
        start, end = segment * duration / count, (segment + 1) * duration / count
        candidates = []
        for i in range(segment * 8, (segment + 1) * 8):
            if is_black(frames[i]):
                black.append(float(times[i]))
            else:
                candidates.append((float(times[i]), frames[i]))
        if len(candidates) < 8:
            for j in range(16):
                timestamp = start + (j + .5) * (end - start) / 16
                frame = decode(timestamp)
                if is_black(frame):
                    black.append(float(timestamp))
                else:
                    candidates.append((float(timestamp), frame))
        if len(candidates) < 8:
            skipped.append({'segment': segment + 1, 'start_sec': start, 'end_sec': end,
                            'reason': '검은 화면 제외 후 유효 시점 8개를 확보하지 못해 미분석'})
            continue
        candidates.sort(key=lambda item: item[0])
        chosen = [candidates[int((i + .5) * len(candidates) / 8)] for i in range(8)]
        groups.append({'segment': segment + 1, 'start_sec': start, 'end_sec': end,
                       'times': [item[0] for item in chosen], 'frames': [item[1] for item in chosen]})
    return groups, skipped, black


def analyze_duration_video(source, output):
    """Reuse frozen scoring/Grad-CAM functions without rewriting the old release."""
    sys.path.insert(0, str(ROOT / 'src'))
    import analyze_64frames as core
    from video_pipeline import probe_video
    import tensorflow as tf
    import image_model_backbone as backbone
    import numpy as np

    manifest = core.verify()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    backbone.SOURCE = core.RELEASE / 'image_model'
    started = time.monotonic()
    local = ROOT / '.local'
    previews = local / 'analyzed_frames64'
    previews.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='adaptive_', dir=previews))
    with tempfile.TemporaryDirectory(prefix='adaptive_input_', dir=local) as temporary:
        if source.startswith(('https://', 'http://')):
            from shorts_worker import download
            path, provenance = download(source, Path(temporary), lambda stage, message: core.progress(message), max_duration_sec=40)
        else:
            path = Path(source).expanduser().resolve()
            if path.is_relative_to((ROOT / 'data/sample/test').resolve()):
                raise ValueError('보존된 Test는 개별 분석 입력으로 사용하지 않습니다.')
            if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
                raise ValueError('90 MB 이하의 영상 파일을 선택하세요.')
            provenance = {'kind': 'file', 'title': path.name}
            destination = Path(temporary) / ('input' + path.suffix)
            shutil.copyfile(path, destination)
            path = destination
        planned = frame_count_for_duration(float(probe_video(path)['duration']))
        count = planned // 8
        core.progress(f'영상 길이에 따라 {planned}개 시점을 추출하고 있습니다.')
        source_sha = core.file_sha256(path)
        frames, times, metadata = core.sample_development_frames(
            {'file_path': str(path.relative_to(ROOT)), 'sha256': source_sha}, planned, 256)
        duration = float(metadata['duration'])
        groups, skipped, black = select_duration_segments(frames, times, duration, lambda t: core.decode_at(path, t), core.is_black)
        if not groups:
            raise ValueError('검은 화면 제외 후 분석 가능한 구간이 없습니다. 실제 영상으로 기본 분류하지 않습니다.')
        core.progress(f'유효 프레임 {len(groups) * 8}/{planned} · 이미지 모델을 불러오고 있습니다.')
        encoder, _ = backbone.load_image_backbone()
        bundle = backbone.load_image_backbone(include_gradcam=True)
        sequences = []
        for group in groups:
            sequence = []
            for frame in group['frames']:
                embedding, real = encoder(tf.convert_to_tensor(frame[None], tf.float32))
                sequence.append(np.concatenate([embedding.numpy()[0], [1 - float(real.numpy()[0, 0])]]))
            sequences.append(np.asarray(sequence, np.float32))
            core.progress(f"구간 {group['segment']}/{count} · 프레임 특징 분석 완료")
        sequences = np.stack(sequences)
        members = core.load_members()
        segment_scores = core.score_sequences(sequences, members)
        score = float(np.mean(segment_scores))
        records = core.explain_frames(groups, sequences, segment_scores, members, bundle, folder)
    main, details = core.rank_frames(records)
    result = {
        'status': 'completed', 'model': manifest['release'], 'analysis_mode': 'duration_frames_v1', 'frame_policy': FRAME_POLICY,
        'threshold': .5, 'source': provenance, 'source_sha256': source_sha, 'duration_sec': duration,
        'fake_score': score, 'prediction': 'fake' if score >= .5 else 'real',
        'prediction_ko': 'AI 생성 영상 추정' if score >= .5 else '실제 촬영 영상 추정',
        'planned_frames': planned, 'analyzed_frames': len(records), 'segments_used': len(groups),
        'black_seek_positions_sec': black, 'skipped_segments': skipped,
        'segments': [{'segment': g['segment'], 'start_sec': g['start_sec'], 'end_sec': g['end_sec'], 'fake_score': float(s)}
                     for g, s in zip(groups, segment_scores, strict=True)],
        'frames': records, 'main_frame_indices': main, 'detail_frame_indices': details,
        'preview_directory': str(folder.resolve()), 'elapsed_sec': round(time.monotonic() - started, 1),
        'accuracy_validated_for_this_mode': False, 'score_is_calibrated_probability': False,
        'limitations_ko': '길이별 16·32·48·64프레임 방식의 정확도는 별도 검증 전입니다. '
                          'Grad-CAM은 선택한 분류의 신호이며 합성 사실이나 합성 위치의 증명이 아닙니다.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (folder / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    core.progress('완료 · 주요 프레임 8개와 세부사항을 확인하세요.')
    return result


def youtube_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError('올바른 공개 YouTube 주소를 입력하세요.')
    if (parsed.hostname or '').lower() not in {'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be'}:
        raise ValueError('youtube.com 또는 youtu.be 주소만 지원합니다.')
    return value


def validate_video_header(path):
    with path.open('rb') as stream:
        head = stream.read(12)
    if not (head[4:8] == b'ftyp' or head[:4] == b'\x1aE\xdf\xa3'
            or head[:4] == b'OggS' or (head[:4] == b'RIFF' and head[8:12] == b'AVI ')):
        raise ValueError('MP4, MOV(ftyp), WEBM, OGG 또는 AVI 영상 파일을 입력하세요.')


class Jobs:
    def __init__(self, directory=None, runner=None):
        self.directory = Path(directory or ROOT / '.local/web_jobs').resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock, self.records = threading.Lock(), {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.runner = runner or self.run

    def reserve(self):
        with self.lock:
            if sum(r['state'] in {'uploading', 'queued', 'running'} for r in self.records.values()) >= 2:
                raise OverflowError('다른 영상을 분석 중입니다. 잠시 후 다시 시도하세요.')
            identity = secrets.token_urlsafe(24)
            folder = self.directory / identity
            folder.mkdir()
            record = {'id': identity, 'folder': folder, 'state': 'uploading', 'created': time.time(), 'message': '영상 준비 중'}
            self.records[identity] = record
            return record

    def submit(self, record, source):
        with self.lock:
            record.update(state='queued', message='분석 대기 중')
        self.executor.submit(self.runner, record, source)

    def run(self, record, source):
        with self.lock:
            record.update(state='running', message='영상 길이에 맞춘 프레임 분석 준비 중')
        output = record['folder'] / 'result.json'
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--analyze-duration', '--input', source, '--output', str(output)]
        try:
            with (record['folder'] / 'worker.log').open('w', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, encoding='utf-8', errors='replace', env=env,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        with self.lock:
                            record['message'] = line.strip()[:240]
                    code = process.wait()
            if not output.is_file():
                raise ValueError('분석을 완료하지 못했습니다. 서버 실행 기록을 확인하세요.')
            result = json.loads(output.read_text(encoding='utf-8'))
            if code or result.get('status') != 'completed':
                raise ValueError(result.get('error', '영상 분석 실패'))
            preview = Path(result['preview_directory']).resolve()
            if not preview.is_relative_to((ROOT / '.local/analyzed_frames64').resolve()):
                raise ValueError('Invalid analysis output path')
            # Only worker-generated PNGs may be served. Never return local paths.
            frames, assets = [], {}
            for frame in result['frames']:
                item = {k: frame.get(k) for k in ['index', 'timestamp_sec', 'segment', 'video_score_delta',
                        'gradcam_available', 'strongest_zone', 'target_class', 'note']}
                for field in ['original_path', 'comparison_path', 'gradcam_path']:
                    if not frame.get(field):
                        continue
                    path = Path(frame[field]).resolve()
                    if path.parent != preview or path.suffix != '.png' or not path.is_file():
                        raise ValueError('Invalid frame asset')
                    assets[path.name] = path
                    item[field.replace('_path', '_url')] = '/jobs/' + record['id'] + '/frames/' + path.name
                frames.append(item)
            safe = {k: result[k] for k in ['prediction', 'prediction_ko', 'fake_score', 'threshold', 'duration_sec',
                    'planned_frames', 'frame_policy', 'analyzed_frames', 'segments', 'skipped_segments', 'main_frame_indices', 'detail_frame_indices',
                    'elapsed_sec', 'limitations_ko']}
            safe.update(frames=frames, accuracy_validated_for_this_mode=False)
            with self.lock:
                record.update(state='completed', message='분석 완료', result=safe, assets=assets, preview=preview)
        except Exception as error:
            with self.lock:
                record.update(state='failed', message=str(error)[:500])
        finally:
            (record['folder'] / 'input.mp4').unlink(missing_ok=True)

    def get(self, identity):
        with self.lock:
            record = self.records.get(identity)
            if record and time.time() - record['created'] < TTL_SECONDS:
                return dict(record)
        return None

    def cleanup(self):
        with self.lock:
            expired = [key for key, r in self.records.items()
                       if r['state'] not in {'uploading', 'queued', 'running'} and time.time() - r['created'] >= TTL_SECONDS]
            records = [self.records.pop(key) for key in expired]
        for record in records:
            folder = record['folder'].resolve()
            if folder.parent == self.directory and folder.name == record['id']:
                shutil.rmtree(folder)
            preview = record.get('preview')
            if preview and preview.resolve().parent == (ROOT / '.local/analyzed_frames64').resolve():
                shutil.rmtree(preview)


def make_server(port=8766, origins=(), jobs=None):
    manager = jobs or Jobs()
    allowed = set(origins)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # URLs contain private job identifiers; don't log them.
            pass

        def permitted(self):
            origin = self.headers.get('Origin')
            local = {'http://127.0.0.1:' + str(self.server.server_port), 'http://localhost:' + str(self.server.server_port)}
            return not origin or origin in allowed | local

        def headers_for(self, status, content_type, length):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            origin = self.headers.get('Origin')
            if origin and self.permitted():
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.end_headers()

        def send_json(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.headers_for(status, 'application/json; charset=utf-8', len(data))
            self.wfile.write(data)

        def do_OPTIONS(self):
            if not self.permitted():
                return self.send_json(403, {'error': '허용되지 않은 출처입니다.'})
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin', 'null'))
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def do_GET(self):
            if not self.permitted():
                return self.send_json(403, {'error': '허용되지 않은 출처입니다.'})
            manager.cleanup()
            path = urlsplit(self.path).path
            if path == '/health':
                return self.send_json(200, {'status': 'online', 'mode': 'segmented64_v1', 'frame_policy': FRAME_POLICY, 'max_bytes': MAX_BYTES})
            if path == '/api/config':
                return self.send_json(200, {'apiBase': '', 'configured': True})
            if path in {'/', '/index.html', '/app.js', '/style.css'}:
                target = SITE / ('index.html' if path == '/' else path[1:])
                data = target.read_bytes()
                self.headers_for(200, mimetypes.guess_type(target)[0] or 'application/octet-stream', len(data))
                return self.wfile.write(data)
            parts = path.strip('/').split('/')
            record = manager.get(parts[1]) if len(parts) >= 2 and parts[0] == 'jobs' else None
            if not record:
                return self.send_json(404, {'error': '결과가 없거나 보관 시간이 만료되었습니다.'})
            if len(parts) == 2:
                result = {k: record[k] for k in ['id', 'state', 'message']}
                if record['state'] == 'completed':
                    result['result'] = record['result']
                return self.send_json(200, result)
            if len(parts) == 4 and parts[2] == 'frames' and record['state'] == 'completed':
                asset = record['assets'].get(parts[3])
                if asset and asset.is_file():
                    data = asset.read_bytes()
                    self.headers_for(200, 'image/png', len(data))
                    return self.wfile.write(data)
            self.send_json(404, {'error': '프레임을 찾지 못했습니다.'})

        def do_POST(self):
            if not self.permitted():
                return self.send_json(403, {'error': '허용되지 않은 출처입니다.'})
            record = None
            try:
                self.connection.settimeout(120)
                path = urlsplit(self.path).path
                if path not in {'/jobs/file', '/jobs/youtube'}:
                    return self.send_json(404, {'error': '존재하지 않는 요청입니다.'})
                length = int(self.headers.get('Content-Length', '0'))
                maximum = 4096 if path.endswith('youtube') else MAX_BYTES
                if self.headers.get('Transfer-Encoding') or not 0 < length <= maximum:
                    return self.send_json(413, {'error': '영상은 90 MB 이하로 입력하세요.'})
                record = manager.reserve()
                if path.endswith('youtube'):
                    body = json.loads(self.rfile.read(length))
                    source = youtube_url(str(body.get('url', '')))
                else:
                    target = record['folder'] / 'input.mp4'
                    remaining = length
                    with target.open('xb') as stream:
                        while remaining:
                            block = self.rfile.read(min(512 * 1024, remaining))
                            if not block:
                                raise ValueError('영상 업로드가 중단되었습니다.')
                            stream.write(block)
                            remaining -= len(block)
                    validate_video_header(target)
                    source = str(target)
                manager.submit(record, source)
                self.send_json(202, {'id': record['id'], 'state': 'queued'})
            except OverflowError as error:
                self.send_json(429, {'error': str(error)})
            except (ValueError, OSError, TypeError) as error:
                if record:
                    with manager.lock:
                        record.update(state='failed', message='입력 실패')
                    (record['folder'] / 'input.mp4').unlink(missing_ok=True)
                self.send_json(400, {'error': str(error)[:300]})

    class APIServer(ThreadingHTTPServer):
        def service_actions(self):
            manager.cleanup()

    server = APIServer(('127.0.0.1', port), Handler)
    server.jobs = manager
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--origin', action='append', default=[], help='Exact HTTPS origin of the published frontend')
    parser.add_argument('--analyze-duration', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.analyze_duration:
        if not args.input or not args.output:
            parser.error('--input and --output are required for analysis')
        try:
            analyze_duration_video(args.input, args.output)
        except Exception as error:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), encoding='utf-8')
            raise
        raise SystemExit(0)
    for origin in args.origin:
        parsed = urlsplit(origin)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.path or parsed.username:
            parser.error('Use an exact HTTPS origin without a path')
    server = make_server(args.port, args.origin)
    print(f'AI video server: http://127.0.0.1:{server.server_port} · keep this process and laptop running', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
