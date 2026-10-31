"""강건성 평가용 이미지 변환 (PIL 기반이라 TensorFlow/PyTorch 양쪽에서 사용 가능)."""

import io

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
