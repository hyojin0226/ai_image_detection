"""Phase 6: real 학습 이미지를 SD VAE 로 재구성해 의사 fake 를 만듭니다 (Rajan et al., ICLR 2025 "Aligned Datasets").

- 원본 real 과 재구성 이미지는 내용·구도가 같고 "잠재 확산 모델 디코더의 흔적"만 다릅니다.
  그래서 모델이 얼굴 내용·구도 같은 지름길 대신 디코더 흔적을 배우도록 유도합니다.
- 파일 형식 지름길(Phase 2)을 막기 위해 원본과 같은 형식으로 저장합니다.
  PNG -> PNG, JPEG -> 원본의 양자화 테이블·서브샘플링을 그대로 써서 JPEG.
- 대상: 세 LOGO split 의 train real 합집합 (valid/test 이미지는 재구성하지 않습니다).

실행 (.venv-exp2): python scripts/51_vae_reconstruct.py [--limit 100]
출력: data/pseudo/vae/<원본 파일명>, data/manifests/pseudo_vae.csv (orig_path, path)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, JpegImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, ROOT

VAE_DIR = ROOT / "models_aux" / "sd-vae-ft-mse"
OUT_DIR = DATA_DIR / "pseudo" / "vae"
MANIFEST = DATA_DIR / "manifests" / "pseudo_vae.csv"


def real_train_paths() -> list:
    paths = set()
    for holdout in ["diffusion", "gan", "manip"]:
        split = pd.read_csv(DATA_DIR / "manifests" / f"logo_{holdout}.csv")
        paths |= set(split[(split["role"] == "train") & (split["label"] == "real")]["path"])
    return sorted(paths)


def save_like(source: Image.Image, image: Image.Image, out_path: Path):
    if source.format == "JPEG":
        image.save(out_path, "JPEG", qtables=source.quantization, subsampling=JpegImagePlugin.get_sampling(source))
    else:
        image.save(out_path, source.format or "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float16).cuda().eval()
    paths = real_train_paths()[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} real train images -> {OUT_DIR}", flush=True)

    records = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        todo, sources, arrays = [], [], []
        for path in batch_paths:
            out_path = OUT_DIR / Path(path).name
            records.append({"orig_path": path, "path": str(out_path)})
            if out_path.exists():
                continue
            source = Image.open(path)
            todo.append(out_path)
            sources.append(source)
            arrays.append(np.asarray(source.convert("RGB"), dtype=np.float32))
        if not todo:
            continue
        # 모두 256x256 이지만 혹시 다르면 하나씩 처리
        groups = [[i] for i in range(len(arrays))] if len({a.shape for a in arrays}) > 1 else [list(range(len(arrays)))]
        for group in groups:
            x = torch.from_numpy(np.stack([arrays[i] for i in group])).permute(0, 3, 1, 2).cuda().half() / 127.5 - 1
            with torch.no_grad():
                latents = vae.encode(x).latent_dist.mode()
                y = vae.decode(latents).sample
            y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for i, out in zip(group, y):
                save_like(sources[i], Image.fromarray(out), todo[i])
        for source in sources:
            source.close()
        if (start // args.batch_size) % 100 == 0:
            print(f"{start + len(batch_paths)}/{len(paths)}", flush=True)

    pd.DataFrame(records).to_csv(MANIFEST, index=False)
    print(f"saved {MANIFEST} ({len(records)} rows)")


if __name__ == "__main__":
    main()
