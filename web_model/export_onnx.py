"""최종 EfficientNetV2S(PyTorch) 모델을 브라우저(onnxruntime-web)용 ONNX 로 내보낸다.

재학습 없이 학습된 가중치를 그대로 담는다. 입력은 정규화된 (N,3,256,256) 얼굴 crop(JPEG 90 재인코딩 후). 출력은 두 개:
- logit:   (N, 1)     sigmoid = P(real). 서버 FakeFaceModel 과 같은 값
- cam_map: (N, h, w)  REAL 방향 Grad-CAM 의 ReLU 이전 맵. FAKE 근거는 -cam_map 에 ReLU.

EfficientNet 머리는 (bn2 특징맵) -> 전역 평균 풀링 -> Linear 라서 logit 의 특징맵 기울기가
모든 위치에서 fc_w / (h*w) 로 같다. 그래서 Grad-CAM 채널 가중치 = fc_w / (h*w) 이고,
브라우저에서 역전파 없이 닫힌 식으로 계산해 그래프에 같이 넣는다 (서버 model.gradcam(bn2) 과 같은지 verify_onnx.py 로 확인).

서버 설정의 input_size 가 256 이 아니면(예: 320) 256 crop 을 키우는 단계를 그래프 안에 넣는다. 서버·학습은 PIL bicubic
(가로 -> uint8 -> 세로 -> uint8, 22비트 정수 계수)이라 ONNX Resize 로는 픽셀이 조금씩 달라지고, 이 탐지 모델은 그 차이에도
점수가 꽤 움직인다. 그래서 PIL 계수 행렬 곱 + 정수 반올림으로 PIL 과 비트 단위로 같게 만든다. tta_hflip 이면 좌우 반전 logit 평균도 그래프에 넣는다
(Grad-CAM 은 서버와 같이 원본 방향만). 그래서 브라우저 쪽 전처리(256 crop -> JPEG -> 정규화)는 바뀌지 않는다.

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\export_onnx.py
출력:
    web_model/out/efficientnetv2s_fake_face.fp32.onnx   (검증 기준)
    web_model/out/efficientnetv2s_fake_face.onnx        (가중치 fp16 저장, 연산은 fp32 -> 배포용)
    web_model/out/model_config.json                     (브라우저 전처리·판정 설정)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import timm
import torch
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "server" / "models" / "efficientnetv2s_fake_face.json"
OUT_DIR = Path(__file__).resolve().parent / "out"
MODEL_STEM = "efficientnetv2s_fake_face"
CROP_SIZE = 256


class ExportWrapper(torch.nn.Module):
    """logit 과 Grad-CAM(REAL 방향) 맵을 함께 내는 래퍼."""

    def __init__(self, model: torch.nn.Module, input_size: int, tta_hflip: bool, mean=(0.5,) * 3, std=(0.5,) * 3):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.input_size = input_size
        self.tta_hflip = tta_hflip
        if input_size != CROP_SIZE:
            hi, lo = pil_bicubic_matrices(CROP_SIZE, input_size)
            self.register_buffer("pil_hi", hi)
            self.register_buffer("pil_lo", lo)
        for param in self.parameters():
            param.requires_grad_(False)

    def _pil_pass(self, pixels: torch.Tensor, transpose: bool) -> torch.Tensor:
        """PIL ImagingResample 한 방향: 22비트 정수 계수 합 -> (합 + 2^21) >> 22 -> 0~255 자르기.
        계수를 상위·하위 11비트로 나눠 곱·합이 모두 2^24 미만 정수가 되게 해서 float32 에서도 PIL 과 비트 단위로 같다."""
        if transpose:
            s_hi, s_lo = self.pil_hi @ pixels, self.pil_lo @ pixels                  # (out, in) @ (.., in, w)
        else:
            s_hi, s_lo = pixels @ self.pil_hi.T, pixels @ self.pil_lo.T              # (.., h, in) @ (in, out)
        q = torch.floor(s_hi / 2048.0)
        r = s_hi - q * 2048.0
        out = q + torch.floor((r * 2048.0 + s_lo + 2097152.0) / 4194304.0)
        return torch.clamp(out, 0.0, 255.0)

    def _features_logit(self, x: torch.Tensor):
        features = self.model.forward_features(x)  # (N, C, h, w): conv_head -> bn2(+SiLU) 출력 = 서버 Grad-CAM 층
        fc = self.model.get_classifier()
        logit = features.mean(dim=(2, 3)) @ fc.weight[0].unsqueeze(1) + fc.bias  # (N, 1)
        return features, logit, fc.weight[0]

    def forward(self, x: torch.Tensor):
        if self.input_size != CROP_SIZE:
            pixels = torch.round((x * self.std + self.mean) * 255.0)  # 정규화 전 0~255 정수로 되돌림
            pixels = self._pil_pass(pixels, transpose=False)  # 가로 (PIL 과 같은 순서)
            pixels = self._pil_pass(pixels, transpose=True)   # 세로
            x = (pixels / 255.0 - self.mean) / self.std
        features, logit, fc_w = self._features_logit(x)
        if self.tta_hflip:
            _, flipped, _ = self._features_logit(torch.flip(x, dims=[3]))
            logit = (logit + flipped) / 2
        hw = features.shape[2] * features.shape[3]
        cam_map = (fc_w[None, :, None, None] / hw * features).sum(dim=1)  # (N, h, w)
        return logit, cam_map


def load_model(config_path: Path) -> tuple[torch.nn.Module, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = timm.create_model(config["timm_name"], pretrained=False, num_classes=1)
    state = torch.load(config_path.parent / config["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def _cubic(x: float, a: float = -0.5) -> float:
    x = abs(x)
    if x < 1:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def pil_bicubic_matrices(in_size: int, out_size: int, bits: int = 22) -> tuple[torch.Tensor, torch.Tensor]:
    """PIL Image.resize(BICUBIC) 의 8비트 정수 계수 행렬 (out, in) 을 상위(x2^11)·하위 11비트로 나눠 돌려준다.
    PIL libImaging/Resample.c precompute_coeffs + normalize_coeffs_8bpc 와 같은 계산."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    coeffs = np.zeros((out_size, in_size), dtype=np.int64)
    for i in range(out_size):
        center = (i + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        k = np.array([_cubic((x + xmin - center + 0.5) / filterscale) for x in range(xmax - xmin)])
        k = k / k.sum()
        coeffs[i, xmin:xmax] = np.where(k < 0, np.trunc(k * (1 << bits) - 0.5), np.trunc(k * (1 << bits) + 0.5))
    hi = np.floor_divide(coeffs, 2048)
    lo = coeffs - hi * 2048
    return torch.tensor(hi, dtype=torch.float32), torch.tensor(lo, dtype=torch.float32)


def check_pil_resize(wrapper: "ExportWrapper", sample: Path | None) -> None:
    """그래프 안 확대가 PIL 과 픽셀 단위로 같은지 확인."""
    from PIL import Image
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)]
    if sample is not None and sample.exists():
        images.append(np.asarray(Image.open(sample).convert("RGB").resize((CROP_SIZE, CROP_SIZE))))
    for image in images:
        ref = np.asarray(Image.fromarray(image).resize((wrapper.input_size,) * 2, Image.BICUBIC), dtype=np.float32)
        pixels = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None]
        got = wrapper._pil_pass(wrapper._pil_pass(pixels, False), True)[0].permute(1, 2, 0).numpy()
        print(f"[resize] PIL bicubic 과 다른 픽셀 {int((ref != got).sum())}개 / {ref.size}")


def fp16_weights(src: Path, dst: Path, min_elements: int = 1024, keep_fp32: tuple[str, ...] = ("pil_",)) -> None:
    """큰 fp32 가중치만 fp16 으로 저장하고 Cast 로 fp32 복원 -> 연산은 그대로 fp32.
    onnxruntime-web 의 wasm(CPU) 백엔드에서도 fp16 연산 지원 여부와 무관하게 동작한다."""
    model = onnx.load(str(src))
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(init)
        if array.size < min_elements or any(key in init.name for key in keep_fp32):  # 확대 계수는 정확해야 함
            continue
        name = init.name
        half_name = f"{name}__fp16"
        init.CopyFrom(numpy_helper.from_array(array.astype(np.float16), half_name))
        casts.append(helper.make_node("Cast", [half_name], [name], to=onnx.TensorProto.FLOAT, name=f"{name}__cast"))
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    print(f"[fp16] Cast 로 감싼 가중치 {len(casts)}개")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.config)
    input_size = int(config.get("input_size", CROP_SIZE))
    tta = bool(config.get("tta_hflip", False))
    wrapper = ExportWrapper(model, input_size, tta, config["mean"], config["std"]).eval()
    dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE)

    # 래퍼 logit 이 원래 모델과 같은지 먼저 확인 (리사이즈·TTA 없는 경우만 정확히 같아야 함)
    with torch.inference_mode():
        plain = ExportWrapper(model, CROP_SIZE, False)
        ref = model(dummy)
        logit, cam = plain(dummy)
    print(f"[check] 래퍼 vs 원본 logit 차이: {(ref - logit).abs().max().item():.2e}, cam {tuple(cam.shape)}")
    if input_size != CROP_SIZE:
        check_pil_resize(wrapper, ROOT / "data" / "sample" / "test" / "fake" / "SDXL_image_0000902.jpg")

    fp32_path = OUT_DIR / f"{MODEL_STEM}.fp32.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(fp32_path),
        input_names=["input"],
        output_names=["logit", "cam_map"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}, "cam_map": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    fp16_path = OUT_DIR / f"{MODEL_STEM}.onnx"
    fp16_weights(fp32_path, fp16_path)

    side = input_size // 32
    browser_config = {
        "model_name": config["model_name"],
        "model_file": fp16_path.name,
        "input_name": "input",
        "outputs": {"logit": "logit of P(real)", "cam_map": f"REAL 방향 Grad-CAM, ReLU 이전 (N,{side},{side})"},
        "input_size": CROP_SIZE,
        "model_input_size": input_size,
        "tta_hflip": tta,
        "mean": config["mean"],
        "std": config["std"],
        "eval_jpeg_quality": config.get("eval_jpeg_quality", 0),
        "decision_threshold_real": config["decision_threshold_real"],
        "class_names": ["fake", "real"],
        "face_crop": {"scale": 2.0, "shift_up": 0.05, "min_face_ratio": 0.04, "max_faces": 8},
        "source_config": str(args.config.resolve().relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT_DIR / "model_config.json").write_text(json.dumps(browser_config, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in (fp32_path, fp16_path):
        print(f"[out] {path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
