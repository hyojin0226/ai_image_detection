"""ConvNeXt-T fake-face 분류 모델 로더 (PyTorch, CPU).

WSL 실험 저장소에서 학습한 최종 모델(Phase 6: 세 계열 전체 + SD VAE 재구성·SBI 의사 fake)을 쓴다.
학습·평가 코드(src/models.py cnn_transform, src/perturb.py EVAL_JPEG)와 같은 전처리를 지킨다.
- 입력: 256x256 RGB crop -> JPEG 품질 90 재인코딩(평가 때와 동일) -> /255 -> ImageNet mean/std 정규화
- 출력: logit 1개, sigmoid = P(real). 클래스 순서 fake=0, real=1
- threshold 는 보기(mode)별로 다름: face(얼굴 crop x2.0) / full(이미지 전체)
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVER_DIR / "models"

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["fake", "real"]
DEFAULT_THRESHOLD = 0.5
CONFIG_FILE_NAME = "convnext_tiny_fake_face.json"


def find_config() -> Path:
    """FAKE_FACE_MODEL_CONFIG 환경변수 또는 server/models/convnext_tiny_fake_face.json."""
    override = os.environ.get("FAKE_FACE_MODEL_CONFIG")
    path = Path(override) if override else MODELS_DIR / CONFIG_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"모델 설정 파일이 없습니다: {path}. README 의 '준비물' 대로 "
            f"convnext_tiny_fake_face.pt 와 {CONFIG_FILE_NAME} 를 server/models/ 에 복사하세요."
        )
    return path


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class FakeFaceModel:
    def __init__(self):
        import timm
        import torch

        self.config_path = find_config()
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)
        self.model_path = self.config_path.parent / self.config["weights"]
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 가중치가 없습니다: {self.model_path}")

        self.model = timm.create_model(self.config["timm_name"], pretrained=False, num_classes=1)
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.source = self.config.get("model_name", "torch")

        thresholds = self.config.get("decision_threshold_real", {})
        self.thresholds = {mode: float(thresholds.get(mode, DEFAULT_THRESHOLD)) for mode in ("face", "full")}
        self.jpeg_quality = int(self.config.get("eval_jpeg_quality", 0))
        self._mean = torch.tensor(self.config["mean"]).view(1, 3, 1, 1)
        self._std = torch.tensor(self.config["std"]).view(1, 3, 1, 1)
        self._torch = torch
        # 첫 요청이 느리지 않도록 미리 한 번 실행
        self.predict_prob_real(np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32))

    # ---------- 전처리 ----------
    def model_inputs(self, batch: np.ndarray) -> list[Image.Image]:
        """(N,256,256,3) 0~255 -> 모델이 실제로 보는 이미지(JPEG 90 재인코딩 후)."""
        images = [Image.fromarray(np.clip(item, 0, 255).astype(np.uint8)) for item in np.asarray(batch)]
        if self.jpeg_quality:
            images = [_jpeg(image, self.jpeg_quality) for image in images]
        return images

    def _tensor(self, images: list[Image.Image]):
        array = np.stack([np.asarray(image, dtype=np.float32) for image in images]) / 255.0
        tensor = self._torch.from_numpy(array).permute(0, 3, 1, 2)
        return (tensor - self._mean) / self._std

    # ---------- 추론 ----------
    def predict_prob_real(self, batch: np.ndarray) -> np.ndarray:
        """batch: (N, 256, 256, 3) float32, 0~255. 반환: (N,) P(real)."""
        with self._torch.inference_mode():
            logits = self.model(self._tensor(self.model_inputs(batch)))
        return self._torch.sigmoid(logits.float()).reshape(-1).numpy().astype(float)

    def threshold_for(self, mode: str) -> float:
        return self.thresholds.get(mode, DEFAULT_THRESHOLD)

    def label_for(self, prob_real: float, mode: str = "face") -> str:
        return CLASS_NAMES[1] if prob_real >= self.threshold_for(mode) else CLASS_NAMES[0]

    # ---------- Grad-CAM ----------
    def gradcam(self, image: Image.Image, target: str) -> np.ndarray:
        """마지막 stage 특징맵 기준 Grad-CAM. target='fake' 면 FAKE 쪽 근거(-logit), 'real' 이면 REAL 쪽 근거.
        image 는 model_inputs() 로 만든 256x256 이미지. 반환: (256, 256) 0~1."""
        torch = self._torch
        captured = {}

        def hook(_module, _inputs, output):
            output.retain_grad()
            captured["features"] = output

        handle = self.model.stages[-1].register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logit = self.model(self._tensor([image])).reshape(())
                score = -logit if target == "fake" else logit
                self.model.zero_grad(set_to_none=True)
                score.backward()
            features = captured["features"].detach()[0]
            gradients = captured["features"].grad.detach()[0]
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * features).sum(dim=0)).numpy()
        if cam.max() <= 0:
            return np.zeros(IMAGE_SIZE[::-1], dtype=np.float32)
        cam = cam / cam.max()
        cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize(IMAGE_SIZE, Image.BICUBIC)
        return np.asarray(cam_image, dtype=np.float32) / 255.0
