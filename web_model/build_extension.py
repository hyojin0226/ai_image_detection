"""모델을 넣은 크롬 확장을 조립하고 ZIP 으로 묶는다 (서버 없이 동작).

extension/ 에 다음을 채운 뒤 dist/ai-face-detector-extension-<버전>.zip 을 만든다.
- fakit-face-model.js  <- web_model/fakit-face-model.js (브라우저 추론 모듈)
- model/efficientnetv2s_fake_face.onnx, model/model_config.json  <- web_model/out/ (export_onnx.py 결과)
- model/blaze_face_short_range.tflite  <- server/models/
- lib/ort/, lib/mediapipe/  <- 없으면 jsdelivr 에서 고정 버전으로 받는다 (MV3 는 원격 코드 실행 금지)

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\export_onnx.py      (out/ 이 없을 때)
    server\\.venv\\Scripts\\python.exe web_model\\build_extension.py
"""
from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
DIST = ROOT / "dist"
ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/"
LIBS = {
    "lib/ort/ort.min.js": ORT + "ort.min.js",
    "lib/ort/ort-wasm-simd-threaded.mjs": ORT + "ort-wasm-simd-threaded.mjs",
    "lib/ort/ort-wasm-simd-threaded.wasm": ORT + "ort-wasm-simd-threaded.wasm",
    "lib/mediapipe/vision_bundle.mjs": MEDIAPIPE + "vision_bundle.mjs",
    "lib/mediapipe/vision_wasm_internal.js": MEDIAPIPE + "wasm/vision_wasm_internal.js",
    "lib/mediapipe/vision_wasm_internal.wasm": MEDIAPIPE + "wasm/vision_wasm_internal.wasm",
}
COPIES = {
    "fakit-face-model.js": ROOT / "web_model" / "fakit-face-model.js",
    "model/efficientnetv2s_fake_face.onnx": ROOT / "web_model" / "out" / "efficientnetv2s_fake_face.onnx",
    "model/model_config.json": ROOT / "web_model" / "out" / "model_config.json",
    "model/blaze_face_short_range.tflite": ROOT / "server" / "models" / "blaze_face_short_range.tflite",
}
NOTICES = """이 확장에 포함된 외부 구성요소
- ONNX Runtime Web 1.20.1 (lib/ort/) — MIT License, Copyright (c) Microsoft Corporation. https://github.com/microsoft/onnxruntime
- MediaPipe Tasks Vision 0.10.14 (lib/mediapipe/) — Apache License 2.0, Copyright Google LLC. https://github.com/google-ai-edge/mediapipe
- BlazeFace short-range 얼굴 검출 모델 (model/blaze_face_short_range.tflite) — Apache License 2.0, Google LLC.
- EfficientNetV2S 얼굴 판정 모델 (model/efficientnetv2s_fake_face.onnx) — 이 프로젝트에서 학습. timm tf_efficientnetv2_s.in21k_ft_in1k 사전학습 가중치 기반 (Apache License 2.0).
"""
SKIP = {".gitignore", "README.md"}


def main() -> None:
    for rel, url in LIBS.items():
        target = EXT / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"[lib] {url}")
            urllib.request.urlretrieve(url, target)
    for rel, source in COPIES.items():
        if not source.exists():
            raise SystemExit(f"없음: {source} (export_onnx.py 를 먼저 실행하세요)")
        target = EXT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (EXT / "THIRD_PARTY_NOTICES.txt").write_text(NOTICES, encoding="utf-8")

    version = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))["version"]
    folder = f"ai-face-detector-extension-{version}"
    DIST.mkdir(exist_ok=True)
    out = DIST / f"{folder}.zip"
    files = sorted(p for p in EXT.rglob("*") if p.is_file() and p.name not in SKIP)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{folder}/{path.relative_to(EXT).as_posix()}")
    print(f"[zip] {out} ({out.stat().st_size / 1e6:.1f} MB, 파일 {len(files)}개)")


if __name__ == "__main__":
    main()
