from pathlib import Path
import random
import shutil
import csv
from collections import defaultdict

# =========================================================
# 0. 기본 설정
# =========================================================

SEED = 42
random.seed(SEED)

# 최종 데이터셋이 만들어질 위치
OUT_ROOT = Path("data")

# 처음에는 True로 실행해서 개수만 확인
# 실제 복사하려면 False로 바꾸기
DRY_RUN = False

# 기존 data/train, valid, test 안에 파일이 있어도 유지할지 여부
# 처음부터 다시 만들고 싶으면 True
RESET_OUTPUT = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =========================================================
# 1. 원본 데이터셋 경로 설정
#    여기를 네 컴퓨터 실제 경로로 수정해야 함
# =========================================================

# 예시 1: WSL 내부에 있는 경우
# Path("/home/hyojin/datasets/GRAVEX-200K/real")

# 예시 2: Windows C드라이브 다운로드 폴더에 있는 경우
# Path("/mnt/c/Users/user/Downloads/GRAVEX-200K/real")


# -------------------------
# [1번] GRAVEX
# -------------------------
from pathlib import Path

GRAVEX_REAL_DIR = Path("/mnt/c/Users/user/Downloads/archive/my_real_vs_ai_dataset/my_real_vs_ai_dataset/real")
GRAVEX_FAKE_DIR = Path("/mnt/c/Users/user/Downloads/archive/my_real_vs_ai_dataset/my_real_vs_ai_dataset/ai_images")


# -------------------------
# [2번] 140k Real and Fake Faces
# -------------------------
K140_TRAIN_REAL_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/train/real")
K140_TRAIN_FAKE_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/train/fake")

K140_VALID_REAL_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/valid/real")
K140_VALID_FAKE_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/valid/fake")

K140_TEST_REAL_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/test/real")
K140_TEST_FAKE_DIR = Path("/mnt/c/Users/user/Downloads/140k_Real_and_Fake_Faces/real_vs_fake/real-vs-fake/test/fake")


# -------------------------
# [3번] 130k Real vs Fake Face
# -------------------------
K130_REAL_DIR = Path("/mnt/c/ai_image/real")

FLUX_DEV_DIR = Path("/mnt/c/Users/user/Downloads/FLUX_DEV")
FLUX_PRO_DIR = Path("/mnt/c/Users/user/Downloads/FLUX_PRO")
SDXL_DIR = Path("/mnt/c/ai_image/SDXL")


# =========================================================
# 2. 기존 분할 수량
# =========================================================

COUNTS = {
    "gravex": {
        "real": {"train": 36800, "valid": 4600, "test": 4600},
        "fake": {"train": 36800, "valid": 4600, "test": 4600},
    },
    "stylegan": {
        "real": {"train": 50000, "valid": 10000, "test": 10000},
        "fake": {"train": 50000, "valid": 10000, "test": 10000},
    },
    "flux_sdxl": {
        "real": {"train": 36800, "valid": 4600, "test": 4600},
        "fake": {"train": 36800, "valid": 4600, "test": 4600},
    },
}


# =========================================================
# 3. 유틸 함수
# =========================================================

def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder}")

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"이미지 파일이 없습니다: {folder}")

    return files


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    if DRY_RUN:
        return

    ensure_dir(dst.parent)

    if dst.exists():
        return

    shutil.copy2(src, dst)


def make_name(source, split, label, idx, src):
    return f"{source}_{split}_{label}_{idx:06d}{src.suffix.lower()}"


def split_files(files, train_n, valid_n, test_n):
    files = list(files)
    random.shuffle(files)

    need = train_n + valid_n + test_n

    if len(files) < need:
        raise RuntimeError(f"파일 수 부족: 필요 {need}장 / 실제 {len(files)}장")

    return {
        "train": files[:train_n],
        "valid": files[train_n:train_n + valid_n],
        "test": files[train_n + valid_n:train_n + valid_n + test_n],
    }


def copy_to_main(files, source, split, label, manifest):
    for idx, src in enumerate(files):
        dst_name = make_name(source, split, label, idx, src)
        dst = OUT_ROOT / split / label / dst_name

        copy_file(src, dst)

        manifest.append({
            "source": source,
            "split": split,
            "label": label,
            "original_path": str(src),
            "saved_path": str(dst),
        })


def copy_to_source_test(files, source, label, manifest):
    for idx, src in enumerate(files):
        dst_name = make_name(source, "test_by_source", label, idx, src)
        dst = OUT_ROOT / "test_by_source" / source / label / dst_name

        copy_file(src, dst)

        manifest.append({
            "source": source,
            "split": f"test_by_source/{source}",
            "label": label,
            "original_path": str(src),
            "saved_path": str(dst),
        })


def allocate_by_ratio(total, source_file_dict):
    counts = {k: len(v) for k, v in source_file_dict.items()}
    source_total = sum(counts.values())

    raw = {
        k: counts[k] / source_total * total
        for k in counts
    }

    result = {
        k: int(raw[k])
        for k in raw
    }

    remain = total - sum(result.values())

    order = sorted(
        raw.keys(),
        key=lambda k: raw[k] - result[k],
        reverse=True
    )

    for k in order[:remain]:
        result[k] += 1

    return result


def write_manifest(manifest):
    manifest_dir = OUT_ROOT / "manifests"
    ensure_dir(manifest_dir)

    manifest_path = manifest_dir / "split_manifest.csv"

    if DRY_RUN:
        print("[DRY_RUN] manifest 저장 생략")
        return

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source", "split", "label", "original_path", "saved_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"[저장 완료] {manifest_path}")


def print_count(title, files):
    print(f"{title}: {len(files):,}장")


def count_output():
    print("\n========== 최종 폴더 파일 수 ==========")

    target_dirs = [
        OUT_ROOT / "train" / "real",
        OUT_ROOT / "train" / "fake",
        OUT_ROOT / "valid" / "real",
        OUT_ROOT / "valid" / "fake",
        OUT_ROOT / "test" / "real",
        OUT_ROOT / "test" / "fake",
        OUT_ROOT / "test_by_source" / "gravex" / "real",
        OUT_ROOT / "test_by_source" / "gravex" / "fake",
        OUT_ROOT / "test_by_source" / "stylegan" / "real",
        OUT_ROOT / "test_by_source" / "stylegan" / "fake",
        OUT_ROOT / "test_by_source" / "flux_sdxl" / "real",
        OUT_ROOT / "test_by_source" / "flux_sdxl" / "fake",
    ]

    for d in target_dirs:
        if d.exists():
            n = len(list_images(d))
        else:
            n = 0
        print(f"{d}: {n:,}장")


# =========================================================
# 4. 메인 실행
# =========================================================

def main():
    print("========== 데이터셋 정리 시작 ==========")

    if RESET_OUTPUT and OUT_ROOT.exists():
        if DRY_RUN:
            print(f"[DRY_RUN] 삭제 예정: {OUT_ROOT}")
        else:
            print(f"[삭제] 기존 출력 폴더 삭제: {OUT_ROOT}")
            shutil.rmtree(OUT_ROOT)

    # 폴더 생성
    for split in ["train", "valid", "test"]:
        ensure_dir(OUT_ROOT / split / "real")
        ensure_dir(OUT_ROOT / split / "fake")

    for source in ["gravex", "stylegan", "flux_sdxl"]:
        ensure_dir(OUT_ROOT / "test_by_source" / source / "real")
        ensure_dir(OUT_ROOT / "test_by_source" / source / "fake")

    manifest = []

    # =====================================================
    # 1번 GRAVEX
    # =====================================================
    print("\n[1번 GRAVEX]")

    gravex_real = list_images(GRAVEX_REAL_DIR)
    gravex_fake = list_images(GRAVEX_FAKE_DIR)

    print_count("GRAVEX real 원본", gravex_real)
    print_count("GRAVEX fake 원본", gravex_fake)

    gravex_real_split = split_files(
        gravex_real,
        train_n=COUNTS["gravex"]["real"]["train"],
        valid_n=COUNTS["gravex"]["real"]["valid"],
        test_n=COUNTS["gravex"]["real"]["test"],
    )

    gravex_fake_split = split_files(
        gravex_fake,
        train_n=COUNTS["gravex"]["fake"]["train"],
        valid_n=COUNTS["gravex"]["fake"]["valid"],
        test_n=COUNTS["gravex"]["fake"]["test"],
    )

    for split in ["train", "valid", "test"]:
        copy_to_main(gravex_real_split[split], "gravex", split, "real", manifest)
        copy_to_main(gravex_fake_split[split], "gravex", split, "fake", manifest)

    copy_to_source_test(gravex_real_split["test"], "gravex", "real", manifest)
    copy_to_source_test(gravex_fake_split["test"], "gravex", "fake", manifest)

    print("GRAVEX 완료")

    # =====================================================
    # 2번 140k
    # =====================================================
    print("\n[2번 140k / stylegan]")

    k140_dirs = {
        "train": {
            "real": K140_TRAIN_REAL_DIR,
            "fake": K140_TRAIN_FAKE_DIR,
        },
        "valid": {
            "real": K140_VALID_REAL_DIR,
            "fake": K140_VALID_FAKE_DIR,
        },
        "test": {
            "real": K140_TEST_REAL_DIR,
            "fake": K140_TEST_FAKE_DIR,
        },
    }

    k140_selected = defaultdict(dict)

    for split in ["train", "valid", "test"]:
        for label in ["real", "fake"]:
            files = list_images(k140_dirs[split][label])

            need = COUNTS["stylegan"][label][split]

            if len(files) < need:
                raise RuntimeError(
                    f"140k {split}/{label} 수 부족: 필요 {need}장 / 실제 {len(files)}장"
                )

            random.shuffle(files)
            selected = files[:need]
            k140_selected[split][label] = selected

            print(f"140k {split}/{label}: {len(files):,}장 중 {need:,}장 사용")

            copy_to_main(selected, "stylegan", split, label, manifest)

    copy_to_source_test(k140_selected["test"]["real"], "stylegan", "real", manifest)
    copy_to_source_test(k140_selected["test"]["fake"], "stylegan", "fake", manifest)

    print("140k 완료")

    # =====================================================
    # 3번 130k / flux_sdxl
    # =====================================================
    print("\n[3번 130k / flux_sdxl]")

    k130_real = list_images(K130_REAL_DIR)

    flux_dev = list_images(FLUX_DEV_DIR)
    flux_pro = list_images(FLUX_PRO_DIR)
    sdxl = list_images(SDXL_DIR)

    print_count("130k real 원본", k130_real)
    print_count("FLUX_DEV 원본", flux_dev)
    print_count("FLUX_PRO 원본", flux_pro)
    print_count("SDXL 원본", sdxl)

    k130_real_split = split_files(
        k130_real,
        train_n=COUNTS["flux_sdxl"]["real"]["train"],
        valid_n=COUNTS["flux_sdxl"]["real"]["valid"],
        test_n=COUNTS["flux_sdxl"]["real"]["test"],
    )

    for split in ["train", "valid", "test"]:
        copy_to_main(k130_real_split[split], "flux_sdxl", split, "real", manifest)

    fake_sources = {
        "flux_dev": flux_dev,
        "flux_pro": flux_pro,
        "sdxl": sdxl,
    }

    train_quota = allocate_by_ratio(
        COUNTS["flux_sdxl"]["fake"]["train"],
        fake_sources
    )
    valid_quota = allocate_by_ratio(
        COUNTS["flux_sdxl"]["fake"]["valid"],
        fake_sources
    )
    test_quota = allocate_by_ratio(
        COUNTS["flux_sdxl"]["fake"]["test"],
        fake_sources
    )

    print("\n130k fake 분배")
    print("train:", train_quota)
    print("valid:", valid_quota)
    print("test :", test_quota)

    selected_fake = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for source_name, files in fake_sources.items():
        files = list(files)
        random.shuffle(files)

        tr_n = train_quota[source_name]
        va_n = valid_quota[source_name]
        te_n = test_quota[source_name]

        need = tr_n + va_n + te_n

        if len(files) < need:
            raise RuntimeError(
                f"{source_name} 수 부족: 필요 {need}장 / 실제 {len(files)}장"
            )

        selected_fake["train"].extend(files[:tr_n])
        selected_fake["valid"].extend(files[tr_n:tr_n + va_n])
        selected_fake["test"].extend(files[tr_n + va_n:tr_n + va_n + te_n])

    for split in ["train", "valid", "test"]:
        random.shuffle(selected_fake[split])
        copy_to_main(selected_fake[split], "flux_sdxl", split, "fake", manifest)

    copy_to_source_test(k130_real_split["test"], "flux_sdxl", "real", manifest)
    copy_to_source_test(selected_fake["test"], "flux_sdxl", "fake", manifest)

    print("130k 완료")

    # =====================================================
    # manifest 저장 및 결과 확인
    # =====================================================
    write_manifest(manifest)

    if not DRY_RUN:
        count_output()

    print("\n========== 완료 ==========")

    if DRY_RUN:
        print("현재는 DRY_RUN=True라 실제 복사는 하지 않았습니다.")
        print("출력 개수와 경로가 맞으면 DRY_RUN=False로 바꾸고 다시 실행하세요.")
    else:
        print("실제 파일 복사가 완료되었습니다.")


if __name__ == "__main__":
    main()