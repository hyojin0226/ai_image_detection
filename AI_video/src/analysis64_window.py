"""64-frame result window. TensorFlow stays in a separate worker process."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]


class AnalysisWindow:
    def __init__(self, window):
        self.window, self.result = window, None
        self.events = queue.Queue()
        self.images = []
        window.title('AI 영상 판별 · 64프레임 분석')
        window.geometry('1080x850')
        window.minsize(1000, 650)
        tk.Label(window, text='AI 영상 판별 · 64프레임 분석', font=('', 18, 'bold')).pack(pady=(14, 5))
        tk.Label(window, text='3~40초 · 90 MB 이하 · 8개 구간 × 8프레임 · 기존 확정 가중치 사용', fg='#555555').pack()
        row = tk.Frame(window)
        row.pack(fill='x', padx=20, pady=10)
        self.source = tk.StringVar()
        tk.Entry(row, textvariable=self.source, font=('', 12)).pack(side='left', fill='x', expand=True)
        tk.Button(row, text='영상 파일 선택', command=self.choose).pack(side='left', padx=6)
        self.analyze_button = tk.Button(row, text='분석 시작', command=self.start, bg='#dceaff', width=12)
        self.analyze_button.pack(side='left')
        self.status = tk.StringVar(value='YouTube Shorts 주소를 입력하거나 영상 파일을 선택하세요.')
        tk.Label(window, textvariable=self.status, fg='#245985', wraplength=1000).pack(padx=20, pady=3)
        self.summary = tk.StringVar(value='분석이 끝나면 주요 프레임 8개를 먼저 표시합니다. 나머지는 세부사항에서 확인할 수 있습니다.')
        tk.Label(window, textvariable=self.summary, justify='left', wraplength=1000, font=('', 11)).pack(fill='x', padx=20, pady=6)
        controls = tk.Frame(window)
        controls.pack(fill='x', padx=20, pady=4)
        tk.Label(controls, text='주요 프레임 8개 · 왼쪽 원본 / 오른쪽 Grad-CAM', font=('', 12, 'bold')).pack(side='left')
        self.details_button = tk.Button(controls, text='세부사항', command=self.open_details, state='disabled')
        self.details_button.pack(side='right')
        self.gallery = tk.Frame(window)
        self.gallery.pack(fill='both', expand=True, padx=12, pady=5)
        tk.Label(window, text='주황·빨강은 선택한 분류를 지지하는 영역입니다. 합성 위치의 확정 표시가 아닙니다.\n'
                 '각 프레임의 색 진하기는 서로 비교할 수 없습니다. 64프레임 방식의 정확도는 아직 재검증 전입니다.',
                 fg='#9a421f', justify='left', wraplength=1000).pack(fill='x', padx=20, pady=(4, 12))
        window.after(100, self.poll)

    def choose(self):
        path = filedialog.askopenfilename(title='3~40초 영상 선택', filetypes=[('영상 파일', '*.mp4 *.mov *.mkv *.webm *.avi'), ('모든 파일', '*.*')])
        if path:
            self.source.set(path)

    def start(self):
        source = self.source.get().strip().strip('"')
        if not source:
            self.status.set('주소 또는 영상 파일을 먼저 선택하세요.')
            return
        self.analyze_button.configure(state='disabled')
        self.details_button.configure(state='disabled')
        self.result = None
        self.images.clear()
        for child in self.gallery.winfo_children():
            child.destroy()
        self.summary.set('64프레임을 분석하고 각 프레임의 Grad-CAM을 계산합니다. 창을 닫지 말고 기다려 주세요.')
        self.status.set('분석 준비 중…')

        def worker():
            try:
                local = ROOT / '.local'
                local.mkdir(exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix='window64_', dir=local))
                output = folder / 'result.json'
                command = [sys.executable, '-u', str(ROOT / 'src/analyze_64frames.py'), '--input', source, '--output', str(output)]
                env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
                with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True,
                                          encoding='utf-8', errors='replace', env=env,
                                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            self.events.put(('progress', line.strip()))
                        code = process.wait()
                if not output.exists():
                    raise RuntimeError('분석을 완료하지 못했습니다. 실행 기록: ' + str(folder / 'worker.log'))
                result = json.loads(output.read_text(encoding='utf-8'))
                if code or result['status'] != 'completed':
                    raise RuntimeError(result.get('error', '분석 실패'))
                self.events.put(('complete', result))
            except Exception as error:
                self.events.put(('error', str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    self.status.set(value)
                elif kind == 'complete':
                    self.show_result(value)
                    self.analyze_button.configure(state='normal')
                elif kind == 'error':
                    self.status.set('분석 실패: ' + value)
                    self.summary.set('결과를 계산하지 못했습니다. 실패한 영상을 실제 촬영으로 기본 분류하지 않습니다.')
                    self.analyze_button.configure(state='normal')
        except queue.Empty:
            pass
        self.window.after(100, self.poll)

    def show_result(self, result):
        self.result = result
        target = 'AI 생성' if result['prediction'] == 'fake' else '실제 촬영'
        self.status.set(f"분석 완료 · {result['elapsed_sec']}초 · 유효 프레임 {result['analyzed_frames']}/64")
        omitted = f" · 검은 화면 때문에 미분석 구간 {len(result['skipped_segments'])}개" if result['skipped_segments'] else ''
        self.summary.set(f"결과: {result['prediction_ko']}  |  AI 점수 {result['fake_score']:.4f} / 임계값 0.5\n"
                         f"Grad-CAM 표시 대상: {target} 판단을 지지하는 영역{omitted}\n"
                         '주요 프레임은 해당 정보를 구간 평균으로 바꿨을 때 최종 점수 변화가 큰 순서입니다. 클릭하면 확대합니다.')
        by_id = {frame['index']: frame for frame in result['frames']}
        frames = [by_id[i] for i in result['main_frame_indices']]
        self.make_gallery(self.gallery, frames, self.images, columns=4, thumb_width=225)
        count = len(result['detail_frame_indices'])
        self.details_button.configure(text=f'세부사항 · 나머지 {count}개', state='normal')

    def make_gallery(self, parent, frames, images, columns=4, thumb_width=225):
        for child in parent.winfo_children():
            child.destroy()
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        body = tk.Frame(canvas)
        canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        def wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), 'units')
            return 'break'
        canvas.bind('<MouseWheel>', wheel)
        body.bind('<MouseWheel>', wheel)
        for position, frame in enumerate(frames):
            card = tk.Frame(body, bd=1, relief='solid')
            card.grid(row=position // columns, column=position % columns, padx=5, pady=6, sticky='n')
            with Image.open(frame['comparison_path']) as source:
                image = source.convert('RGB')
                image.thumbnail((thumb_width, 160))
            picture = ImageTk.PhotoImage(image, master=parent)
            images.append(picture)
            tk.Button(card, image=picture, command=lambda f=frame: self.zoom(f)).pack(padx=3, pady=3)
            delta = frame['video_score_delta']
            caption = f"{frame['index']}번 · {frame['timestamp_sec']:.3f}초 · 구간 {frame['segment']}\nAI 점수 영향 {delta:+.5f}"
            caption += '\n강조 영역: ' + (frame['strongest_zone'] or '양의 신호 없음')
            tk.Label(card, text=caption, wraplength=thumb_width, font=('', 10), fg='#23517a').pack(padx=3, pady=4)
            if frame.get('note'):
                tk.Label(card, text=frame['note'], wraplength=thumb_width, fg='#8b4a20').pack(padx=3, pady=3)
            # Bind only this gallery's widgets; scrolling never affects other windows.
            for widget in [card, *card.winfo_children()]:
                widget.bind('<MouseWheel>', wheel)
        if not frames:
            tk.Label(body, text='추가 프레임이 없습니다.').pack(padx=15, pady=15)

    def zoom(self, frame):
        popup = tk.Toplevel(self.window)
        popup.title(f"{frame['index']}번 · {frame['timestamp_sec']:.3f}초 · 원본 / Grad-CAM")
        with Image.open(frame['comparison_path']) as source:
            image = source.convert('RGB')
            image = image.resize((1024, 560), Image.Resampling.NEAREST)
        picture = ImageTk.PhotoImage(image, master=popup)
        popup.picture = picture
        tk.Label(popup, image=picture).pack(padx=8, pady=8)
        target = 'AI 생성' if frame['target_class'] == 'fake' else '실제 촬영'
        tk.Label(popup, text=f'{target} 판단을 지지하는 양의 신호입니다. 합성 위치를 인증하는 표시가 아닙니다.').pack(pady=8)

    def open_details(self):
        if not self.result:
            return
        result = self.result
        popup = tk.Toplevel(self.window)
        popup.title('세부사항 · 나머지 프레임과 구간별 결과')
        popup.geometry('1080x780')
        popup.images = []
        scores = '  |  '.join(f"{item['start_sec']:.1f}~{item['end_sec']:.1f}초: {item['fake_score']:.3f}" for item in result['segments'])
        tk.Label(popup, text='구간별 AI 점수\n' + scores, wraplength=1000, justify='left').pack(fill='x', padx=16, pady=10)
        if result['skipped_segments']:
            message = '\n'.join(f"구간 {item['segment']}: {item['reason']}" for item in result['skipped_segments'])
            tk.Label(popup, text=message, fg='#a04020', wraplength=1000).pack(padx=16, pady=4)
        tk.Label(popup, text='메인 8개를 제외한 나머지 프레임을 시간순으로 표시합니다.\n'
                 '점수 영향 +: AI 점수를 높인 정보 / −: 낮춘 정보 (평균 특징으로 대체한 비교 기준)').pack(pady=4)
        gallery = tk.Frame(popup)
        gallery.pack(fill='both', expand=True, padx=12)
        by_id = {frame['index']: frame for frame in result['frames']}
        self.make_gallery(gallery, [by_id[i] for i in result['detail_frame_indices']], popup.images)
        tk.Button(popup, text='프레임 저장 폴더 열기', command=lambda: os.startfile(result['preview_directory'])).pack(pady=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, help='Display a saved result without model inference')
    args = parser.parse_args()
    window = tk.Tk()
    app = AnalysisWindow(window)
    if args.result:
        app.show_result(json.loads(args.result.read_text(encoding='utf-8')))
    window.mainloop()


if __name__ == '__main__':
    main()
