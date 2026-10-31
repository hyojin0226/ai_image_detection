"""notebooks/01_robustness_report.ipynb 생성.

노트북은 저장된 결과(CSV/PNG)만 읽습니다. 모델을 불러오거나 학습하지 않습니다.
실행: python scripts/12_build_robustness_report.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_robustness_report.ipynb
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = [
    new_markdown_cell(
        "# Phase 1. 강건성 테스트 리포트\n\n"
        "1차 EfficientNetV2S 모델(threshold 0.24, 재학습 없음)에 JPEG 재압축·리사이즈·블러를 적용한 결과입니다.\n\n"
        "- 평가셋: 현재 `data/test` (1차 ID Test와 같은 26,876장)\n"
        "- 변환은 256x256으로 맞춘 뒤 적용\n"
        "- 소스별 fake/real 수가 달라서 소스 비교는 **balanced accuracy**와 **AUC** 기준"
    ),
    new_code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n\n"
        "RESULT_DIR = Path('../outputs/exp2/robustness')\n"
        "summary = pd.read_csv(RESULT_DIR / 'robustness_summary.csv')\n"
        "ORDER = ['none', 'jpeg_q90', 'jpeg_q70', 'jpeg_q50', 'resize_0.75', 'resize_0.5', 'blur_0.5', 'blur_1.0']"
    ),
    new_markdown_cell("## 1. 전체 결과"),
    new_code_cell(
        "columns = ['perturbation', 'accuracy', 'balanced_accuracy', 'roc_auc_real_positive', 'fake_recall', 'real_recall', 'delta_balanced_accuracy']\n"
        "summary[summary['subset'] == 'all'][columns].set_index('perturbation').loc[ORDER].round(4)"
    ),
    new_markdown_cell("## 2. 소스별 balanced accuracy / Fake recall"),
    new_code_cell(
        "for metric in ['balanced_accuracy', 'fake_recall', 'roc_auc_real_positive']:\n"
        "    table = summary.pivot(index='perturbation', columns='subset', values=metric).loc[ORDER]\n"
        "    display(table.round(4))\n"
        "    ax = table.drop(columns='all').plot(marker='o', figsize=(9, 4), title=metric)\n"
        "    ax.set_xticks(range(len(ORDER)), ORDER, rotation=30)\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(RESULT_DIR / f'robustness_{metric}.png', dpi=150)\n"
        "    plt.show()"
    ),
    new_markdown_cell(
        "## 3. 원본 해상도 분포\n\n"
        "fake와 real의 원본 해상도가 소스 안에서 다르면, 모델이 '진짜/가짜' 대신 해상도 차이(리사이즈 흔적)를 단서로 쓸 수 있습니다."
    ),
    new_code_cell(
        "sizes = pd.read_csv(RESULT_DIR / 'image_sizes.csv')\n"
        "sizes.pivot_table(index=['source', 'size'], columns='label', values='images', fill_value=0)"
    ),
    new_markdown_cell(
        "## 4. GRAVEX에서 놓친 Fake\n\n"
        "GRAVEX fake에는 전체 생성 얼굴뿐 아니라 실제 영상의 얼굴만 바꾼 **부분 조작**(FaceForensics++, CelebDF, DFDC)이 섞여 있습니다. 원본 파일명 기준 출처별 놓친 비율:"
    ),
    new_code_cell("pd.read_csv(RESULT_DIR / 'gradcam_gravex' / 'gravex_miss_rate_by_origin.csv')"),
    new_markdown_cell("### Grad-CAM\n\n위: 원본, 아래: fake 쪽 logit에 기여한 영역"),
    new_code_cell(
        "display(Image(RESULT_DIR / 'gradcam_gravex' / 'missed_top.png'))\n"
        "display(Image(RESULT_DIR / 'gradcam_gravex' / 'caught_examples.png'))"
    ),
]

notebook = new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
output = ROOT / "notebooks" / "01_robustness_report.ipynb"
nbformat.write(notebook, output)
print("저장:", output)
