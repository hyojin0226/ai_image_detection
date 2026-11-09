"""강건성 평가용 이미지 변환 (PIL 기반이라 TensorFlow/PyTorch 양쪽에서 사용 가능)."""

import io
import random

from PIL import Image, ImageFilter


def jpeg(quality: int):
    def apply(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return apply


def resize(scale: float):
    """축소 후 원래 크기로 복원 (SNS 썸네일처럼 해상도가 줄어든 상황)."""
    def apply(image: Image.Image) -> Image.Image:
        width, height = image.size
        small = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
        return small.resize((width, height), Image.BILINEAR)
    return apply


def blur(sigma: float):
    def apply(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return apply


def random_jpeg(low: int = 60, high: int = 95):
    """Phase 3 학습용: real·fake 모두 무작위 품질로 재인코딩해서 파일 형식·압축 차이 단서를 없앱니다."""
    def apply(image: Image.Image) -> Image.Image:
        return jpeg(random.randint(low, high))(image)
    return apply


# Phase 3 JPEG 정규화 규칙 (EXPERIMENT_PLAN.md Phase 2 대조 실험 참고)
TRAIN_JPEG = random_jpeg(60, 95)
EVAL_JPEG = jpeg(90)


def random_degrade(resize_p: float = 0.5, resize_range=(0.4, 1.0), blur_p: float = 0.3, blur_range=(0.3, 1.2)):
    """실험 A 후속: 축소·복원과 블러를 무작위로 넣은 뒤 TRAIN_JPEG. real·fake(의사 fake 포함) 모두 같은 분포로 적용.
    최종 모델이 축소 0.5x·블러 σ1.0 에서 GAN fake 를 크게 놓친 문제를 겨냥합니다 (crop 배율 무작위화는 넣지 않음)."""
    def apply(image: Image.Image) -> Image.Image:
        if random.random() < resize_p:
            width, height = image.size
            scale = random.uniform(*resize_range)
            method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), method)
            image = image.resize((width, height), random.choice([Image.BILINEAR, Image.BICUBIC]))
        if random.random() < blur_p:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur_range)))
        return TRAIN_JPEG(image)
    return apply


# EXPERIMENT_PLAN.md Phase 1의 변환 목록
PERTURBATIONS = {
    "none": None,
    "jpeg_q90": jpeg(90),
    "jpeg_q70": jpeg(70),
    "jpeg_q50": jpeg(50),
    "resize_0.75": resize(0.75),
    "resize_0.5": resize(0.5),
    "blur_0.5": blur(0.5),
    "blur_1.0": blur(1.0),
}
