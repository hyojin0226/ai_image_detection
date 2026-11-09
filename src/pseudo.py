"""Phase 6 의사 fake (pseudo-fake): 새 생성기 데이터 없이 real 이미지만으로 fake 예시를 만듭니다.

1. vae: SD VAE 재구성 (scripts/51_vae_reconstruct.py 로 미리 생성, Rajan et al. ICLR 2025).
   원본 real 과 내용·구도·파일 형식이 같고 잠재 확산 디코더 흔적만 달라서, 확산 계열 공통 흔적을 배우게 합니다.
2. sbi: Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) 단순화 버전. 학습 중 즉석 생성.
   같은 real 이미지를 색·선명도·크기만 살짝 바꾼 사본을 얼굴 영역 타원 마스크로 다시 합성해,
   얼굴 경계의 합성 흔적(블렌딩 경계, 색 불일치, 해상도 차이)을 배우게 합니다.
   원 논문은 얼굴 랜드마크 볼록 껍질 마스크를 쓰지만, 여기서는 얼굴 검출 박스 기반 타원 마스크로 근사합니다.

의사 fake 행은 원본 real 행을 복사해 y=0(fake), generator=pseudo_vae / pseudo_sbi 로 둡니다.
valid / test 는 건드리지 않습니다 (모델 선택·평가는 진짜 데이터로만).
"""

import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.config import DATA_DIR
from src.data import load_image

PSEUDO_VAE_MANIFEST = DATA_DIR / "manifests" / "pseudo_vae.csv"
PSEUDO_FRACTION = 0.5  # 의사 fake 종류마다 real train 장수의 50% 를 추가


def add_pseudo_rows(train: pd.DataFrame, kinds: list, boxes: dict, seed: int) -> pd.DataFrame:
    """train real 행에서 의사 fake 행을 만들어 붙입니다. vae 경로의 얼굴 박스는 원본 박스를 씁니다."""
    real = train[train["y"] == 1]
    parts = [train.assign(pseudo="")]
    for offset, kind in enumerate(kinds):
        sample = real.sample(frac=PSEUDO_FRACTION, random_state=seed + offset).copy()
        if kind == "vae":
            recon = dict(pd.read_csv(PSEUDO_VAE_MANIFEST).values)
            sample = sample[sample["path"].isin(recon)]
            sample["path"] = sample["path"].map(recon)
            for orig, new in recon.items():
                if orig in boxes:
                    boxes[new] = boxes[orig]
        sample = sample.assign(y=0, label="fake", family="pseudo", generator=f"pseudo_{kind}", pseudo=kind)
        parts.append(sample)
    return pd.concat(parts).reset_index(drop=True)


def _ellipse_mask(size, box) -> Image.Image:
    width, height = size
    if box is None:
        box = (0.25, 0.2, 0.5, 0.6)
    x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    cx, cy = x + w / 2, y + h / 2 + random.uniform(-0.05, 0.1) * h
    rx, ry = w / 2 * random.uniform(0.8, 1.25), h / 2 * random.uniform(0.9, 1.4)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 0.15 * min(rx, ry) + 2)))


def self_blend(image: Image.Image, box) -> Image.Image:
    """SBI: 살짝 바꾼 자기 사본을 얼굴 타원 마스크로 다시 합성합니다."""
    width, height = image.size
    source = image
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        if random.random() < 0.7:
            source = enhancer(source).enhance(random.uniform(0.88, 1.12))
    roll = random.random()
    if roll < 0.3:
        scale = random.uniform(0.5, 0.85)
        source = source.resize((round(width * scale), round(height * scale)), Image.BILINEAR).resize((width, height), Image.BILINEAR)
    elif roll < 0.5:
        source = source.filter(ImageFilter.UnsharpMask(radius=2, percent=random.randint(60, 150), threshold=2))
    elif roll < 0.65:
        source = source.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    # 약간의 크기·위치 어긋남 (원 논문의 source-target 정렬 오차)
    scale = random.uniform(0.96, 1.04)
    dx, dy = (random.uniform(-0.02, 0.02) * width for _ in range(2))
    affine = (1 / scale, 0, -dx + width * (1 - 1 / scale) / 2, 0, 1 / scale, -dy + height * (1 - 1 / scale) / 2)
    source = source.transform((width, height), Image.AFFINE, affine, resample=Image.BILINEAR)
    # 어긋나서 이미지 밖(검은 영역)이 된 곳은 합성하지 않습니다
    valid = Image.new("L", (width, height), 255).transform((width, height), Image.AFFINE, affine, resample=Image.NEAREST)
    ratio = random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    mask = np.asarray(_ellipse_mask((width, height), box), dtype=np.float32) / 255.0 * ratio
    mask = (mask * (np.asarray(valid, dtype=np.float32) / 255.0))[..., None]
    blended = np.asarray(image, dtype=np.float32) * (1 - mask) + np.asarray(source, dtype=np.float32) * mask
    return Image.fromarray(blended.round().clip(0, 255).astype(np.uint8))


class PseudoDataset:
    """Phase 3 전체 이미지 보기(ManifestDataset)와 같은 처리 + pseudo=='sbi' 행은 즉석 SBI 합성."""

    def __init__(self, rows: pd.DataFrame, boxes: dict, transform, perturb=None):
        self.paths = rows["path"].tolist()
        self.labels = rows["y"].tolist()
        self.kinds = rows["pseudo"].fillna("").tolist()
        self.boxes = boxes
        self.transform = transform
        self.perturb = perturb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = load_image(path)
        if self.kinds[index] == "sbi":
            image = self_blend(image, self.boxes.get(path))
        if self.perturb is not None:
            image = self.perturb(image)
        return self.transform(image), self.labels[index]
