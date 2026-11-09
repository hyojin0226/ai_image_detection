"""Phase 5 학습 증강: 기존 데이터만으로 다양성을 늘리고 지름길 단서를 지웁니다.

1. 얼굴 crop 무작위화: real·fake 모두 같은 얼굴 검출 박스 기준으로 배율 1.4~3.0 crop.
   flux_sdxl 처럼 계열마다 얼굴 크기(구도)가 다른 단서를 지우고, 데모 서버의 얼굴 crop 입력과 조건을 맞춥니다.
2. SNS 열화 체인: 축소·복원, 블러/샤픈, 밝기·대비·채도, 피부 보정 같은 스무딩, JPEG 40~95.
   Phase 1 에서 압축이 세질수록 real 을 FAKE 로 오판하던 문제를 겨냥합니다. 두 클래스에 똑같이 적용합니다.
3. 생성기 균형 샘플링: 클래스 안에서 생성기별 장수^-0.5 가중 (SDXL 이 대부분인 diffusion 계열 등).
"""

import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from src.config import DATA_DIR, IMAGE_SIZE
from src.perturb import jpeg

FACE_BOXES_PATH = DATA_DIR / "manifests" / "face_boxes.csv"

# 데모 서버(server/face.py)와 같은 평가용 crop
EVAL_CROP_SCALE = 2.0
CROP_SHIFT_UP = 0.05
TRAIN_CROP_SCALE = (1.4, 3.0)
TRAIN_CROP_JITTER = 0.05


def load_face_boxes(path=FACE_BOXES_PATH) -> dict:
    """path -> (x, y, w, h) 비율 좌표. 얼굴을 못 찾은 이미지는 들어 있지 않습니다."""
    boxes = pd.read_csv(path).dropna(subset=["w"])
    return {p: (x, y, w, h) for p, x, y, w, h in zip(boxes["path"], boxes["x"], boxes["y"], boxes["w"], boxes["h"])}


def face_crop(image: Image.Image, box, scale: float, jitter: float = 0.0) -> Image.Image:
    """얼굴 박스 한 변 x scale 정사각형 crop 후 256x256. 이미지 밖으로 나가면 안쪽으로 밀어 맞춥니다(패딩 없음)."""
    if box is None:
        return image.resize(IMAGE_SIZE, Image.BILINEAR)
    width, height = image.size
    x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    side = min(max(w, h) * scale, width, height)
    cx = x + w / 2 + random.uniform(-jitter, jitter) * side
    cy = y + h / 2 - CROP_SHIFT_UP * side + random.uniform(-jitter, jitter) * side
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return image.crop((left, top, left + side, top + side)).resize(IMAGE_SIZE, Image.BILINEAR)


def _beauty_smooth(image: Image.Image) -> Image.Image:
    """피부 보정 앱처럼 경계는 살리고 질감만 뭉개는 스무딩."""
    try:
        import cv2

        array = cv2.bilateralFilter(np.asarray(image), d=5, sigmaColor=random.uniform(20, 50), sigmaSpace=5)
        return Image.fromarray(array)
    except ImportError:
        return image.filter(ImageFilter.SMOOTH)


def sns_degrade(image: Image.Image) -> Image.Image:
    """SNS 업로드·보정 앱을 거친 것 같은 무작위 열화. 마지막에 항상 JPEG 40~95 로 재인코딩합니다."""
    if random.random() < 0.5:
        width, height = image.size
        scale = random.uniform(0.4, 1.0)
        method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), method)
        image = image.resize((width, height), random.choice([Image.BILINEAR, Image.BICUBIC]))

    roll = random.random()
    if roll < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    elif roll < 0.45:
        image = image.filter(ImageFilter.UnsharpMask(radius=random.uniform(1, 2), percent=random.randint(50, 150), threshold=2))

    if random.random() < 0.5:
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhancer(image).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.2:
        image = _beauty_smooth(image)

    return jpeg(random.randint(40, 95))(image)


def balanced_weights(rows: pd.DataFrame, power: float = 0.5) -> np.ndarray:
    """클래스 안에서 생성기별 장수^-power 가중. 클래스별 가중치 합은 원래 장수와 같게 유지합니다."""
    weights = pd.Series(1.0, index=rows.index)
    for _, class_rows in rows.groupby("y"):
        counts = class_rows.groupby("generator").size()
        generator_weight = class_rows["generator"].map(counts.astype(float) ** -power)
        weights[class_rows.index] = generator_weight / generator_weight.sum() * len(class_rows)
    return weights.to_numpy()


class FaceCropDataset:
    """얼굴 crop 보기를 쓰는 Dataset.

    train=True: 무작위 배율 crop + SNS 열화(JPEG 포함)
    train=False: 서버와 같은 x2.0 crop + perturb(평가용 JPEG 90 등)
    """

    def __init__(self, rows: pd.DataFrame, boxes: dict, transform, train: bool, perturb=None):
        self.paths = rows["path"].tolist()
        self.labels = rows["y"].tolist()
        self.boxes = boxes
        self.transform = transform
        self.train = train
        self.perturb = perturb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        box = self.boxes.get(path)
        if self.train:
            image = sns_degrade(face_crop(image, box, random.uniform(*TRAIN_CROP_SCALE), TRAIN_CROP_JITTER))
        else:
            image = face_crop(image, box, EVAL_CROP_SCALE)
            if self.perturb is not None:
                image = self.perturb(image)
        return self.transform(image), self.labels[index]
