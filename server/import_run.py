"""WSL 학습 run(scripts/32_train_cnn.py 결과)을 데모 서버용 모델로 가져온다.

outputs/exp2/<run>/<holdout>/best.pt 를 server/models/<name>.pt 로 복사하고, 같은 이름의 설정 JSON 을 만든다.
threshold 는 그 run 이 valid 에서 정한 값: 얼굴 crop 보기 = crop_view/config.yaml, 전체 이미지 = config.yaml.
입력 해상도(img_size)·좌우 반전 평균(tta)도 run 설정을 그대로 옮긴다.

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe server\\import_run.py efficientnetv2s_pseudo_vae_deg_t4 --name efficientnetv2s_fake_face
    server\\.venv\\Scripts\\python.exe server\\import_run.py efficientnetv2s_pseudo_sbi --holdout diffusion --name logo_diff_sbi
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

WSL_ROOT = Path(r"\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection")
MODELS_DIR = Path(__file__).resolve().parent / "models"
TIMM = {
    "efficientnetv2s": ("tf_efficientnetv2_s.in21k_ft_in1k", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], "bn2"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--holdout", default="all")
    parser.add_argument("--name", required=True, help="server/models/<name>.pt / .json")
    args = parser.parse_args()

    run_dir = WSL_ROOT / "outputs" / "exp2" / args.run / args.holdout
    full = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    face = yaml.safe_load((run_dir / "crop_view" / "config.yaml").read_text(encoding="utf-8"))
    timm_name, mean, std, layer = TIMM[full["model"]]

    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(run_dir / "best.pt", MODELS_DIR / f"{args.name}.pt")
    config = {
        "model_name": f"{args.run}/{args.holdout}",
        "timm_name": timm_name,
        "weights": f"{args.name}.pt",
        "output": "logit of P(real); fake=0, real=1",
        "input_size": int(full.get("img_size", 256)),
        "tta_hflip": bool(full.get("tta", False)),
        "gradcam_layer": layer,
        "mean": mean,
        "std": std,
        "eval_jpeg_quality": 90,
        "decision_threshold_real": {"face": round(float(face["threshold"]), 4), "full": round(float(full["threshold"]), 4)},
        "threshold_note": "valid Macro F1 기준. face = 얼굴 crop x2.0 보기, full = 이미지 전체 보기",
        "training_run": f"WSL outputs/exp2/{args.run}/{args.holdout}",
        "training_options": {k: full.get(k) for k in ("pseudo", "degrade", "epochs", "lr", "zero_head", "img_size",
                                                      "ema", "drop_path", "crop_mix", "tta", "seed")},
    }
    path = MODELS_DIR / f"{args.name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {path} (threshold face {config['decision_threshold_real']['face']}, full {config['decision_threshold_real']['full']})")


if __name__ == "__main__":
    main()
