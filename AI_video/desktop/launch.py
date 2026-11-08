"""Entry point for the self-contained Windows distribution; model files stay unchanged."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['PATH'] = str(ROOT / 'runtime/node') + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, str(ROOT / 'src'))


def self_test(video=None):
    """Check the bundled interpreter, GUI, decoder, weights and actual gradient path."""
    sys.path.insert(0, str(ROOT / 'tools'))
    from release import check
    check()
    import tkinter as tk
    window = tk.Tk()
    window.withdraw()
    window.update()
    window.destroy()
    import baseline_common
    import numpy as np
    import tensorflow as tf
    import imageio_ffmpeg
    import yt_dlp
    import yt_dlp_ejs
    import image_model_backbone as backbone
    from predict_final_research import RELEASE, verify
    from video_explanation import positive_gradcam
    verify()
    backbone.SOURCE = RELEASE / 'image_model'
    encoder, _ = backbone.load_image_backbone()
    image = tf.zeros((1, 256, 256, 3), tf.float32)
    embedding, real = encoder(image)
    bundle = backbone.load_image_backbone(include_gradcam=True)
    from analyze_64frames import load_members, make_differentiable_head
    head = make_differentiable_head(load_members())
    sequence = np.tile(np.concatenate([embedding.numpy()[0], [1 - float(real.numpy()[0, 0])]]), (8, 1))
    # Exercise the exact RBF-video-score -> EfficientNet spatial gradient path.
    activation = bundle[2](image)
    with tf.GradientTape() as tape:
        tape.watch(activation)
        features, real_score = bundle[3](activation)
        row = tf.concat([features[0], 1 - real_score[0]], axis=0)
        updated = tf.tensor_scatter_nd_update(tf.constant(sequence, tf.float64), [[0]], tf.cast(row[None], tf.float64))
        score = head(updated)
    gradient = tape.gradient(score, activation)
    if gradient is None or not np.isfinite(gradient.numpy()).all():
        raise ValueError('Bundled Grad-CAM gradient path failed')
    positive_gradcam(activation.numpy()[0], gradient.numpy()[0])
    decoder = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'],
                             capture_output=True, check=True,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    node = subprocess.run([str(ROOT / 'runtime/node/node.exe'), '--version'],
                          capture_output=True, text=True, check=True,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = {'status': 'passed', 'python': sys.version.split()[0],
            'python_executable': sys.executable, 'tensorflow': tf.__version__,
            'embedding_shape': list(embedding.shape), 'image_score': float(real.numpy()[0, 0]),
            'gradcam_gradient_verified': True, 'ffmpeg': decoder.stdout.decode().splitlines()[0],
            'node': node.stdout.strip(), 'model_integrity': 'passed',
            'training_or_test_evaluation': False}
    if video:
        local = ROOT / '.local'
        local.mkdir(exist_ok=True)
        output = local / 'desktop_video_check.json'
        with (local / 'desktop_video_check.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(ROOT / 'runtime/python/pythonw.exe'), '-u',
                            str(ROOT / 'src/analyze_64frames.py'), '--input', str(video), '--output', str(output)],
                           env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
                           stdout=log, stderr=log, check=True, timeout=600,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        analysis = json.loads(output.read_text(encoding='utf-8'))
        assert analysis['status'] == 'completed' and analysis['analyzed_frames'] == 64
        assert len(analysis['main_frame_indices']) == 8 and len(analysis['detail_frame_indices']) == 56
        assert all(Path(frame['comparison_path']).is_file() for frame in analysis['frames'])
        result['video_pipeline'] = {'frames': 64, 'main_frames': 8, 'detail_frames': 56,
                                    'gradcam_comparison_images': 64,
                                    'result': str(output), 'elapsed_sec': analysis['elapsed_sec'],
                                    'synthetic_functional_check_only': True}
    return result


def main():
    if '--self-test' in sys.argv:
        output = Path(sys.argv[sys.argv.index('--self-test') + 1])
        try:
            video = Path(sys.argv[sys.argv.index('--video') + 1]) if '--video' in sys.argv else None
            result = self_test(video)
        except Exception:
            result = {'status': 'failed', 'error': traceback.format_exc()}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0 if result['status'] == 'passed' else 1
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    with (local / 'desktop.log').open('a', encoding='utf-8', buffering=1) as log:
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
        try:
            sys.argv = [str(ROOT / 'src/analysis64_window.py'), *sys.argv[1:]]
            runpy.run_path(sys.argv[0], run_name='__main__')
        except Exception:
            traceback.print_exc(file=log)
            from tkinter import messagebox
            messagebox.showerror('AI 영상 판별', '실행하지 못했습니다. 실행 기록: ' + str(local / 'desktop.log'))
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
