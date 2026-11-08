"""Frozen project release. Checks by default; inference requires explicit input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'models/final_research_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    lock = json.loads((RELEASE / 'release_lock_final.json').read_text(encoding='utf-8'))
    if sha(RELEASE / 'finalization.json') != lock['manifest_sha256']:
        raise ValueError('확정 모델의 설정 파일이 변경되었습니다.')
    manifest = json.loads((RELEASE / 'finalization.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['release_files_sha256'].items():
        if sha(RELEASE / relative) != expected:
            raise ValueError('확정 모델 파일이 변경되었습니다: ' + relative)
    for relative, expected in manifest['project_runtime_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('실행 코드가 확정 버전과 다릅니다: ' + relative)
    if manifest['threshold'] != 0.5:
        raise ValueError('확정 임계값은 0.5입니다.')
    return manifest


def run(source):
    verify()
    import baseline_common  # CPU settings before TensorFlow import.
    import tensorflow as tf
    import image_model_backbone as backbone
    import predict_improved_candidate as core
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    core.RELEASE = RELEASE
    backbone.SOURCE = RELEASE / 'image_model'
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='final_research_', dir=local) as folder:
        if source.startswith(('https://', 'http://')):
            from shorts_worker import download
            path, _ = download(source, Path(folder), lambda stage, message: print(message, flush=True),
                               max_duration_sec=40)
        else:
            path = Path(source).expanduser().resolve()
            if path.is_relative_to((ROOT / 'data/sample/test').resolve()):
                raise ValueError('보존된 Test는 개별 판별 입력으로 사용하지 않습니다.')
            if not path.is_file() or not 0 < path.stat().st_size <= 90 * 1024 * 1024:
                raise ValueError('90 MB 이하의 영상 파일을 선택하세요.')
            destination = Path(folder) / ('input' + path.suffix)
            shutil.copyfile(path, destination)
            path = destination
        result = core.predict_video(path)
    result.update(model='final_research_v1', candidate='development_improved_v2',
                  research_version_confirmed=True, production_gate_passed=False,
                  score_is_calibrated_probability=False)
    output = local / 'final_research_result.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n판별 결과: ' + ('AI 생성 영상 추정' if result['prediction'] == 'fake' else '실제 촬영 영상 추정'))
    print(f"AI 점수: {result['score_fake']:.4f} / 임계값: 0.5")
    print('사용한 프레임 시점(초): ' + ', '.join(f'{t:.3f}' for t in result['timestamps_sec']))
    print('연구용 확정 버전입니다. 점수는 정답 확률이 아니며 30초 초과 성능은 미검증입니다.')
    print('결과 저장: ' + str(output))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify()
    if args.verify_only or not (args.input or args.interactive):
        print('확정 모델 파일·실행 코드 확인 완료. 학습·영상 예측은 실행하지 않았습니다.')
        return
    source = args.input
    if args.interactive and not source:
        print('AI 영상 분류 · 연구용 확정 v1 (3~40초 / 90 MB 이하)')
        print('YouTube Shorts 주소 또는 영상 경로를 입력하세요. 빈칸으로 Enter를 누르면 파일 선택창이 열립니다.')
        source = input('주소 또는 경로: ').strip().strip('"')
        if not source:
            import tkinter as tk
            from tkinter import filedialog
            window = tk.Tk()
            window.withdraw()
            try:
                source = filedialog.askopenfilename(title='3~40초 영상 선택')
            finally:
                window.destroy()
        if not source:
            print('선택을 취소했습니다.')
            return
    run(source)


if __name__ == '__main__':
    main()
