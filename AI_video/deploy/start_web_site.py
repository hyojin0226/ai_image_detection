"""Start the optional notebook URL service; browser file analysis stays independent.

Uses the operator's existing local Vercel login. Never reads or publishes tokens.
No model training, Git operations, or changes to weights and inference code.
"""
import argparse
from datetime import datetime
import json
import msvcrt
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'deploy/site'
URL = 'https://duhyeon-ai-video.vercel.app'
# Fakit's video page sends YouTube Shorts links to this notebook server as well.
FAKIT_URL = 'https://fakit-lens.vercel.app'
PROJECT = 'prj_SiXAdlXOnCa0OkzKODmdTv4iXqyk'
TOOLS = ROOT.parent / '.local/deploy-tools'
VERCEL = TOOLS / 'vercel/node_modules/@vercel/vc-native-win32-x64/bin/vercel.exe'
AUTH = ROOT.parent / '.local/vercel-auth'
CLOUDFLARED = TOOLS / 'cloudflared-windows-amd64.exe'
STATE = ROOT / '.local/web_launcher'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def json_get(url, timeout=8):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}), timeout=timeout) as response:
        return json.load(response)


def healthy(url):
    try:
        result = json_get(url + '/health', 4)
        return result.get('status') == 'online' and result.get('mode') == 'segmented64_v1'
    except (OSError, ValueError, urllib.error.URLError):
        return False


def valid_tunnel(value):
    try:
        parts = urlsplit(value)
        return parts.scheme == 'https' and parts.hostname.endswith('.trycloudflare.com') and not parts.username and not parts.password and parts.path in ('', '/') and not parts.query and not parts.fragment and parts.port in (None, 443)
    except (ValueError, AttributeError):
        return False


class Controller:
    def __init__(self, emit):
        STATE.mkdir(parents=True, exist_ok=True)
        self.emit = emit
        self.stop = threading.Event()
        self.retry = threading.Event()
        self.children = []
        self.tunnel = None
        self.active_url = ''
        self.log_lock = threading.Lock()
        self.env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        node = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin'
        if (node / 'node.exe').is_file():
            self.env['PATH'] = str(node) + os.pathsep + self.env.get('PATH', '')

    def log(self, text):
        with self.log_lock, (STATE / 'launcher.log').open('a', encoding='utf-8') as stream:
            stream.write(datetime.now().isoformat(timespec='seconds') + ' ' + text.rstrip() + '\n')

    def status(self, kind, message):
        self.log(kind + ': ' + message)
        temporary = STATE / 'status.tmp'
        temporary.write_text(json.dumps({'status': kind, 'message': message, 'site': URL,
             'api_base': self.active_url, 'pid': os.getpid(), 'time': datetime.now().isoformat()},
             ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(STATE / 'status.json')
        self.emit(kind, message)

    def spawn(self, command, **kwargs):
        if self.stop.is_set():
            raise RuntimeError('종료 중입니다.')
        process = subprocess.Popen(command, cwd=kwargs.pop('cwd', ROOT), env=self.env,
                                   creationflags=NO_WINDOW, **kwargs)
        self.children.append(process)
        return process

    def end_child(self, process):
        if process and process.poll() is None:
            # Only processes started by this launcher; never kill by name or port.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=NO_WINDOW, timeout=15)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def request_stop(self):
        self.stop.set()
        for process in reversed(list(self.children)):
            try:
                self.end_child(process)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def ensure_api(self):
        if healthy('http://127.0.0.1:8766'):
            return
        self.status('working', '1/3 · 영상 분석 서버를 켜고 있습니다.')
        python = Path(sys.executable).with_name('python.exe')
        with (STATE / 'server.log').open('a', encoding='utf-8') as log:
            process = self.spawn([str(python), '-u', str(ROOT / 'deploy/notebook_server.py'),
                                  '--port', '8766', '--origin', URL, '--origin', FAKIT_URL], stdout=log, stderr=log)
        for _ in range(30):
            if healthy('http://127.0.0.1:8766'):
                return
            if process.poll() is not None:
                raise RuntimeError('분석 서버를 켜지 못했습니다. server.log를 확인하세요.')
            if self.stop.wait(1):
                raise RuntimeError('종료 중입니다.')
        raise RuntimeError('분석 서버가 응답하지 않습니다.')

    def configured_url(self):
        try:
            value = json_get(URL + '/api/config').get('notebookApiBase', '')
            return value if valid_tunnel(value) else ''
        except (OSError, ValueError, urllib.error.URLError):
            return ''

    def new_tunnel(self):
        if not CLOUDFLARED.is_file():
            raise RuntimeError('연결 프로그램을 찾을 수 없습니다: ' + str(CLOUDFLARED))
        self.end_child(self.tunnel)
        self.status('working', '2/3 · 사이트와 노트북을 자동으로 연결하고 있습니다.')
        found = queue.Queue()
        self.tunnel = self.spawn([str(CLOUDFLARED), 'tunnel', '--url', 'http://127.0.0.1:8766',
                                  '--no-autoupdate', '--protocol', 'http2'], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        process = self.tunnel

        def collect():
            with (STATE / 'tunnel.log').open('a', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                    if match and valid_tunnel(match.group()):
                        found.put(match.group())
        threading.Thread(target=collect, daemon=True).start()
        deadline = time.monotonic() + 100
        value = ''
        while time.monotonic() < deadline and not self.stop.is_set():
            if process.poll() is not None:
                raise RuntimeError('외부 연결을 시작하지 못했습니다. 인터넷 연결을 확인하세요.')
            if not value:
                try:
                    value = found.get(timeout=1)
                except queue.Empty:
                    continue
            if healthy(value):
                return value
            self.stop.wait(2)
        raise RuntimeError('외부 연결을 기다리는 중입니다. 인터넷이 복구되면 자동으로 다시 시도합니다.')

    def cli(self, args, data=None, timeout=240):
        if not VERCEL.is_file() or not AUTH.is_dir():
            raise RuntimeError('이 노트북의 Vercel 실행 도구 또는 로그인 설정을 찾을 수 없습니다.')
        command = [str(VERCEL), *args, '--scope', 'duhyeon', '--global-config', str(AUTH)]
        process = self.spawn(command, cwd=SITE, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding='utf-8', errors='replace')
        try:
            output, _ = process.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.end_child(process)
            raise RuntimeError('사이트 연결 설정 반영이 지연되고 있습니다. 잠시 후 자동으로 다시 시도합니다.')
        self.log(output)
        if process.returncode:
            if any(word in output.lower() for word in ['log in', 'login', 'unauthorized', 'token is invalid', 'expired']):
                raise RuntimeError('Vercel 로그인이 만료되었습니다. 운영자 계정 재연결이 필요합니다.')
            raise RuntimeError('사이트 설정 반영을 완료하지 못했습니다. 잠시 후 자동으로 다시 시도합니다.')

    def publish_connection(self, value):
        self.status('working', '3/3 · 새 연결 주소를 사이트에 자동 반영하고 있습니다. 잠시 기다려 주세요.')
        self.cli(['env', 'add', 'NOTEBOOK_API_BASE', '--environment', 'production', '--project', PROJECT, '--force', '--yes'], value + '\n')
        self.cli(['--prod', '--yes'], timeout=300)
        for _ in range(20):
            if self.configured_url() == value and healthy(value):
                return
            if self.stop.wait(2):
                raise RuntimeError('종료 중입니다.')
        raise RuntimeError('사이트 반영을 확인하고 있습니다. 잠시 후 자동으로 다시 확인합니다.')

    def run(self):
        failures = 0
        try:
            while not self.stop.is_set():
                try:
                    self.ensure_api()
                    value = self.configured_url()
                    if not value or not healthy(value):
                        if self.active_url and healthy(self.active_url):
                            value = self.active_url
                        else:
                            value = self.new_tunnel()
                        self.active_url = value
                        self.publish_connection(value)
                    self.active_url = value
                    failures = 0
                    self.status('ready', '주소 분석 연결 완료 · 파일 분석도 가능합니다. 공유 주소는 그대로입니다.')
                    while not self.stop.wait(20):
                        if self.retry.is_set():
                            self.retry.clear()
                            break
                        if healthy('http://127.0.0.1:8766') and healthy(value):
                            failures = 0
                        else:
                            failures += 1
                            if failures >= 3:
                                self.status('working', '연결이 끊어져 자동으로 복구하고 있습니다.')
                                break
                except Exception as error:
                    if self.stop.is_set():
                        break
                    self.status('error', str(error))
                    # Keep a still-healthy new tunnel after a transient deployment error.
                    for _ in range(30):
                        if self.stop.wait(1) or self.retry.is_set():
                            self.retry.clear()
                            break
        finally:
            for process in reversed(self.children):
                try:
                    self.end_child(process)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self.status('stopped', '주소 분석 서버를 종료했습니다. 웹사이트의 파일 분석은 계속 가능합니다.')


def take_lock():
    STATE.mkdir(parents=True, exist_ok=True)
    stream = (STATE / 'instance.lock').open('a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        stream.close()
        return None
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Keep the same supervisor running without its window')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    lock = take_lock()
    if lock is None:
        if not args.no_browser:
            webbrowser.open(URL)
        return
    events = queue.Queue()
    controller = Controller(lambda kind, value: events.put((kind, value)))
    worker = threading.Thread(target=controller.run, daemon=True)
    worker.start()
    if args.headless:
        try:
            while worker.is_alive():
                try:
                    kind, value = events.get(timeout=1)
                    print(kind + ': ' + value, flush=True)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            controller.request_stop()
            worker.join(timeout=35)
        finally:
            lock.close()
        return
    window = tk.Tk()
    window.title('웹사이트 켜기 · AI 영상 분석')
    window.geometry('630x310')
    window.resizable(False, False)
    window.configure(bg='#f6f5fa')
    tk.Label(window, text='AI 영상 분석 사이트', font=('', 19, 'bold'), bg='#f6f5fa').pack(pady=(25, 10))
    tk.Label(window, text=URL, fg='#6551b7', bg='#f6f5fa', font=('', 12)).pack()
    status = tk.StringVar(value='자동으로 준비하고 있습니다…')
    label = tk.Label(window, textvariable=status, bg='#f6f5fa', wraplength=570, justify='center', font=('', 11))
    label.pack(padx=25, pady=22)
    buttons = tk.Frame(window, bg='#f6f5fa')
    buttons.pack()
    open_button = tk.Button(buttons, text='사이트 열기', width=16, state='disabled', command=lambda: webbrowser.open(URL))
    open_button.pack(side='left', padx=5)
    tk.Button(buttons, text='연결 다시 확인', width=16, command=controller.retry.set).pack(side='left', padx=5)
    tk.Label(window, text='주소 분석을 사용하려면 이 창을 켜 두세요. 종료·절전 후에도 사이트의 파일 분석은 가능합니다.',
             bg='#f6f5fa', fg='#777080', wraplength=570, font=('', 10)).pack(pady=18)
    opened = False
    closing = False

    def poll():
        nonlocal opened
        try:
            while True:
                kind, value = events.get_nowait()
                status.set(value)
                label.configure(fg='#217b5b' if kind == 'ready' else '#a24631' if kind == 'error' else '#343047')
                open_button.configure(state='normal' if kind == 'ready' else 'disabled')
                if kind == 'ready' and not opened and not args.no_browser:
                    opened = True
                    webbrowser.open(URL)
        except queue.Empty:
            pass
        if closing and not worker.is_alive():
            window.destroy()
        else:
            window.after(200, poll)

    def close():
        nonlocal closing
        if closing:
            return
        if messagebox.askyesno('웹 서버 종료', '종료하면 주소 분석 연결이 중단됩니다. 파일 분석은 계속 가능합니다.\n주소 분석 서버를 종료하시겠습니까?', parent=window):
            closing = True
            controller.stop.set()
            threading.Thread(target=controller.request_stop, daemon=True).start()
            status.set('웹 서버를 종료하고 있습니다…')
    window.protocol('WM_DELETE_WINDOW', close)
    window.after(100, poll)
    try:
        window.mainloop()
    finally:
        controller.request_stop()
        worker.join(timeout=35)
        lock.close()


if __name__ == '__main__':
    main()
