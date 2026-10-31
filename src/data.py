"""manifest 기반 데이터 로딩.

`data/manifests/split_manifest.csv`를 기준으로 split 목록을 다루고,
파일 복사 없이 목록(CSV)만으로 LOGO split을 구성할 수 있게 합니다.
"""

import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import ROOT, MANIFEST_PATH, LABEL_TO_INDEX, IMAGE_SIZE

MAIN_SPLITS = ["train", "valid", "test"]


def _generator_name(row) -> str:
    """세부 생성기 이름. flux_sdxl은 원본 경로로 FLUX_DEV / FLUX_PRO / SDXL을 구분합니다."""
    if row["label"] == "real":
        return "real"
    if row["source"] == "flux_sdxl":
        for name in ["FLUX_DEV", "FLUX_PRO", "SDXL"]:
            if f"/{name}/" in row["original_path"]:
                return name.lower()
    return row["source"]


def load_manifest(path: Path = MANIFEST_PATH, include_source_copies: bool = False) -> pd.DataFrame:
    """manifest를 읽고 `path`(절대 경로), `y`(0/1), `generator` 열을 추가합니다.

    `test_by_source/*` 행은 test split의 복사본이라 기본으로 제외합니다.
    """
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    if not include_source_copies:
        manifest = manifest[manifest["split"].isin(MAIN_SPLITS)]

    manifest = manifest.copy()
    manifest["path"] = manifest["saved_path"].map(lambda p: str(ROOT / p))
    manifest["y"] = manifest["label"].map(LABEL_TO_INDEX).astype(int)
    manifest["generator"] = manifest.apply(_generator_name, axis=1)
    return manifest.reset_index(drop=True)


def _source_from_name(name: str) -> str:
    """파일명 접두어로 소스를 판별. 접두어가 없는 파일은 add_dataset.sh로 추가한 1M Fake Faces / FFHQ입니다."""
    for source in ["gravex", "stylegan", "flux_sdxl"]:
        if name.startswith(f"{source}_"):
            return source
    return "ffhq_1mff"


def scan_data_dir(data_dir: Path = ROOT / "data") -> pd.DataFrame:
    """실제 data/{train,valid,test}/{fake,real} 폴더를 읽어 manifest와 같은 형태로 만듭니다.

    split_manifest.csv는 add_dataset.sh 추가분과 이후 valid/test 조정이 반영되어 있지 않으므로,
    실제 학습·평가에 쓰인 구성은 이 함수 기준으로 봐야 합니다.
    """
    rows = []
    for split in MAIN_SPLITS:
        for label in LABEL_TO_INDEX:
            folder = data_dir / split / label
            for name in sorted(os.listdir(folder)):
                rows.append({
                    "source": _source_from_name(name),
                    "split": split,
                    "label": label,
                    "path": str(folder / name),
                    "y": LABEL_TO_INDEX[label],
                })
    return pd.DataFrame(rows)


FFHQ_NAME = re.compile(r"\d{5}\.(png|jpg)")
ONE_MFF_NAME = re.compile(r"[A-Z0-9]{10}\.jpg")  # 1M Fake Faces 파일명 형식


def _classify(source: str, label: str, original_name: str, original_path: str):
    """(family, generator, domain, image_key, group_key)를 반환합니다.

    - family: fake의 생성기 계열 (gan / diffusion / manip). 출처가 불분명하면 unknown
    - domain: 사진 종류 (photo: FFHQ 같은 정지 사진 / video: 영상 프레임). real·fake 모두 기록
    - image_key: 같은 사진을 하나로 묶는 키 (FFHQ ID, 1MFF 파일명)
    - group_key: split을 나눌 때 함께 움직여야 하는 단위 (같은 영상의 프레임 등)
    """
    name = original_name
    if FFHQ_NAME.fullmatch(name):
        key = f"ffhq:{int(name[:5])}"
        return "real", "ffhq", "photo", key, key
    if ONE_MFF_NAME.fullmatch(name):
        key = f"1mff:{name}"
        return "gan", "stylegan_1mff", "photo", key, key

    path_key = f"path:{original_path or name}"
    if source == "flux_sdxl":
        generator = next((g.lower() for g in ["FLUX_DEV", "FLUX_PRO", "SDXL"] if f"/{g}/" in original_path), "unknown")
        return "diffusion", generator, "photo", path_key, path_key
    if "_SD_" in name:
        return "diffusion", "sd", "photo", path_key, path_key

    # 영상 프레임: 같은 영상(또는 같은 대상 영상)의 프레임은 같은 split에 들어가야 합니다.
    if match := re.match(r"fake_(DeepFakeDetection)_(.+)_\d+\.jpg$", name):
        return "manip", "ffpp_dfd", "video", path_key, f"ffpp_dfd:{match.group(2)}"
    if match := re.match(r"fake_([A-Za-z0-9]+)_(\d+)_\d+_\d+\.jpg$", name):
        return "manip", f"ffpp_{match.group(1).lower()}", "video", path_key, f"ffpp:{match.group(2)}"
    if match := re.match(r"real_(\d+)_\d+\.jpg$", name):
        return "real", "ffpp_real", "video", path_key, f"ffpp:{match.group(1)}"
    if match := re.match(r"(celebdf)_[a-z]+_(.+?)_frame", name):
        family = "manip" if label == "fake" else "real"
        return family, "celebdf", "video", path_key, f"celebdf:{match.group(2)}"
    if match := re.match(r"(dfdc)_[a-z]+_(?:fake_)?([a-z]+)_\d+_\d+\.jpg$", name):
        family = "manip" if label == "fake" else "real"
        return family, "dfdc", "video", path_key, f"dfdc:{match.group(2)}"

    # aug_ai_*, 1024_man_men_*, ai_detect_*, Aumented real 등: 생성기·출처를 특정할 수 없음
    return "unknown", "unknown", "unknown", path_key, path_key


def annotate(rows: pd.DataFrame) -> pd.DataFrame:
    """scan_data_dir() 결과에 원본 파일명, 계열, 중복 제거 키, 파일 형식을 추가합니다."""
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    original_by_name = dict(zip(manifest["saved_path"].str.split("/").str[-1], manifest["original_path"]))

    rows = rows.copy()
    name = rows["path"].str.split("/").str[-1]
    original_path = name.map(original_by_name).fillna("")
    original_name = original_path.str.split("/").str[-1].where(original_path != "", name)
    info = [
        _classify(s, l, n, p)
        for s, l, n, p in zip(rows["source"], rows["label"], original_name, original_path)
    ]
    rows[["family", "generator", "domain", "image_key", "group_key"]] = pd.DataFrame(info, index=rows.index)
    rows["original_name"] = original_name
    rows["ext"] = name.str.extract(r"\.(\w+)$")[0].str.lower()
    return rows


def select(manifest: pd.DataFrame, split=None, sources=None) -> pd.DataFrame:
    rows = manifest
    if split is not None:
        rows = rows[rows["split"] == split]
    if sources is not None:
        rows = rows[rows["source"].isin(sources)]
    return rows.reset_index(drop=True)


def sample_per_class(rows: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """클래스당 최대 `max_per_class`장. 소스 비율은 원래 비율을 유지합니다."""
    parts = []
    for _, class_rows in rows.groupby("label"):
        fraction = min(max_per_class, len(class_rows)) / len(class_rows)
        for _, source_rows in class_rows.groupby("source"):
            parts.append(source_rows.sample(frac=fraction, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def load_image(path: str, size=IMAGE_SIZE) -> Image.Image:
    """RGB로 열고 `size`로 bilinear 리사이즈 (1차 학습의 image_dataset_from_directory와 동일)."""
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.BILINEAR)


class ManifestDataset:
    """PyTorch용 Dataset. `transform`은 PIL 이미지를 받아 텐서를 반환해야 합니다."""

    def __init__(self, rows: pd.DataFrame, transform=None, perturb=None, size=IMAGE_SIZE):
        self.paths = rows["path"].tolist()
        self.labels = rows["y"].tolist()
        self.transform = transform
        self.perturb = perturb
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_image(self.paths[index], self.size)
        if self.perturb is not None:
            image = self.perturb(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
