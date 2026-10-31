#!/bin/bash
# TensorFlow(.venv)가 pip로 설치된 CUDA 라이브러리를 찾도록 LD_LIBRARY_PATH를 설정합니다.
# 사용: source scripts/tf_gpu_env.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_DIR="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia"
for lib_dir in "$NVIDIA_DIR"/*/lib; do
    LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
done
export LD_LIBRARY_PATH
source "$REPO_ROOT/.venv/bin/activate"
