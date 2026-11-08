"""Bounded local web API for the frozen 64-frame model. No model training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(__file__).parent / 'site'
MAX_BYTES = 90 * 1024 * 1024
TTL_SECONDS = 3600


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
            record.update(state='running', message='64프레임 분석 준비 중')
        output = record['folder'] / 'result.json'
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        command = [sys.executable, '-u', str(ROOT / 'src/analyze_64frames.py'), '--input', source, '--output', str(output)]
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
                    'analyzed_frames', 'segments', 'skipped_segments', 'main_frame_indices', 'detail_frame_indices',
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
                return self.send_json(200, {'status': 'online', 'mode': 'segmented64_v1', 'max_bytes': MAX_BYTES})
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
    args = parser.parse_args()
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
