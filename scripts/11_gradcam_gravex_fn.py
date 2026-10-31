"""Phase 1 추가 분석: GRAVEX에서 놓친 Fake(Real로 오판) 상위 사례의 Grad-CAM.

10_eval_robustness.py의 변환 없음(none) 예측을 사용합니다.
비교를 위해 제대로 잡아낸 GRAVEX Fake 사례도 함께 그립니다.

실행 (WSL, 저장소 루트):
    source scripts/tf_gpu_env.sh
    python scripts/11_gradcam_gravex_fn.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASELINE_MODEL_PATH, BASELINE_CONFIG_PATH, EXP2_DIR, MANIFEST_PATH
from src.data import load_image

PREDICTIONS_PATH = EXP2_DIR / "robustness" / "predictions" / "none.csv"
OUT_DIR = EXP2_DIR / "robustness" / "gradcam_gravex"
LAST_CONV_LAYER_NAME = "top_conv"
N_CASES = 16


def head_forward(model, features):
    import tensorflow as tf

    """배포된 .keras 모델의 분류층(GAP+GMP 결합 → Dense 256 → BN → ReLU → Dropout → sigmoid)을 그대로 적용합니다.

    1차 노트북의 classifier_head(GAP → Dropout → Dense)와 구조가 달라서 레이어 이름으로 재구성합니다.
    """
    layer = model.get_layer
    pooled = layer("pooled")([layer("gap")(features), layer("gmp")(features)])
    x = layer("relu")(layer("bn")(layer("dense_256")(pooled), training=False))
    x = layer("dropout_1")(x, training=False)
    # sigmoid 이전 logit을 반환합니다. P(real)이 0이나 1에 붙으면 확률 기준 기울기가 사라지기 때문입니다.
    dense = layer("probability_real")
    return tf.matmul(x, dense.kernel) + dense.bias


def build_gradcam(model):
    """저장된 1차 모델에서 backbone을 꺼내 Grad-CAM 계산 함수를 만듭니다 (1차 노트북 16절과 같은 방식)."""
    import tensorflow as tf

    base_model = model.get_layer("efficientnetv2-s")
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(LAST_CONV_LAYER_NAME).output, base_model.output],
    )

    def gradcam(image_array):
        """fake 쪽 logit(-logit_real)에 기여한 영역의 히트맵."""
        batch = tf.convert_to_tensor(image_array[None], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, backbone_outputs = feature_model(batch, training=False)
            logit_real = head_forward(model, backbone_outputs)[:, 0]
            target = -logit_real
        prob_real = tf.sigmoid(logit_real)
        gradients = tape.gradient(target, conv_outputs)
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.maximum(tf.reduce_sum(conv_outputs[0] * weights, axis=-1), 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), float(prob_real[0])

    return gradcam


def miss_rate_by_origin(gravex_fake, threshold):
    """GRAVEX fake를 원본 파일명으로 출처(FaceForensics++ 조작 방식, CelebDF, DFDC, 생성 이미지 등)별로 나눠 놓친 비율을 계산합니다."""
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    manifest = manifest[(manifest["source"] == "gravex") & (manifest["split"] == "test")]
    original_name = manifest["original_path"].str.split("/").str[-1]
    origin = (
        original_name.str.extract(r"^fake_([A-Za-z0-9]+)_")[0].radd("FF++/")
        .fillna(original_name.str.extract(r"^(dfdc|celebdf|aug)_")[0])
        .fillna("generated_face")
    )
    origin_by_name = dict(zip(manifest["saved_path"].str.split("/").str[-1], origin))

    rows = gravex_fake.assign(
        origin=gravex_fake["path"].map(lambda p: origin_by_name.get(Path(p).name, "unknown")),
        missed=gravex_fake["prob_real"] >= threshold,
    )
    return (
        rows.groupby("origin")
        .agg(images=("missed", "size"), missed=("missed", "sum"), miss_rate=("missed", "mean"))
        .sort_values("missed", ascending=False)
        .reset_index()
    )


def draw_grid(cases, gradcam, title, path):
    columns = 4
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows * 2, columns, figsize=(columns * 3, rows * 6))
    axes = np.atleast_2d(axes)
    for i, (_, case) in enumerate(cases.iterrows()):
        image = np.asarray(load_image(case["path"]), dtype=np.float32)
        heatmap, prob_real = gradcam(image)
        heatmap = np.asarray(
            Image.fromarray(np.uint8(heatmap * 255)).resize(image.shape[1::-1], Image.BILINEAR),
            dtype=np.float32,
        ) / 255
        r, c = (i // columns) * 2, i % columns
        axes[r, c].imshow(image.astype(np.uint8))
        axes[r, c].set_title(f"P(real)={prob_real:.3f}\n{Path(case['path']).name}", fontsize=8)
        axes[r + 1, c].imshow(image.astype(np.uint8))
        axes[r + 1, c].imshow(heatmap, cmap="jet", alpha=0.4)
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    import tensorflow as tf

    threshold = json.loads(BASELINE_CONFIG_PATH.read_text())["decision_threshold_real"]
    predictions = pd.read_csv(PREDICTIONS_PATH)
    gravex_fake = predictions[(predictions["source"] == "gravex") & (predictions["label"] == "fake")]

    missed = gravex_fake[gravex_fake["prob_real"] >= threshold].sort_values("prob_real", ascending=False)
    caught = gravex_fake[gravex_fake["prob_real"] < threshold].sort_values("prob_real")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missed.to_csv(OUT_DIR / "gravex_false_negatives.csv", index=False)
    print(f"GRAVEX fake {len(gravex_fake)}장 중 놓친 것 {len(missed)}장 ({len(missed) / len(gravex_fake):.1%})")

    by_origin = miss_rate_by_origin(gravex_fake, threshold)
    by_origin.to_csv(OUT_DIR / "gravex_miss_rate_by_origin.csv", index=False)
    print(by_origin.to_string(index=False))

    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    gradcam = build_gradcam(model)

    # 재구성한 분류층이 원래 모델과 같은 확률을 내는지 확인
    sample = np.asarray(load_image(missed.iloc[0]["path"]), dtype=np.float32)
    expected = float(model.predict(sample[None], verbose=0)[0, 0])
    _, actual = gradcam(sample)
    assert abs(expected - actual) < 1e-4, (expected, actual)

    draw_grid(missed.head(N_CASES), gradcam, "GRAVEX fake predicted REAL (most confident misses)", OUT_DIR / "missed_top.png")
    draw_grid(caught.head(N_CASES // 2), gradcam, "GRAVEX fake correctly predicted FAKE", OUT_DIR / "caught_examples.png")
    print("저장:", OUT_DIR)


if __name__ == "__main__":
    main()
