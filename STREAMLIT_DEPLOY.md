# Streamlit Community Cloud 배포

이 프로젝트는 GitHub 저장소를 Streamlit Community Cloud에 연결해 무료 공개 URL로 배포할 수 있습니다.

## 배포에 필요한 파일

아래 파일만 배포에 필수입니다.

- `streamlit_app.py`
- `requirements.txt`
- `.gitattributes`
- `outputs/efficientnetv2s_experiment_01/models/inference_config.json`
- `outputs/efficientnetv2s_experiment_01/models/efficientnetv2s_fake_face_final.keras`

최종 모델은 약 122MB이므로 일반 Git 파일로 push할 수 없습니다. `.gitattributes`는 이 모델을 Git LFS로 관리하도록 설정되어 있습니다.

## 1. Git LFS 설치

Ubuntu/WSL:

```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

Windows에서는 [Git LFS](https://git-lfs.com/)를 설치한 뒤 터미널에서 실행합니다.

```bash
git lfs install
```

설정 확인:

```bash
git lfs track
git check-attr filter -- outputs/efficientnetv2s_experiment_01/models/efficientnetv2s_fake_face_final.keras
```

`filter: lfs`가 출력되어야 합니다.

## 2. 배포 파일을 GitHub에 push

현재 저장소에는 실험 결과 등 다른 변경 사항도 있으므로, 배포 파일만 명시적으로 추가하는 편이 안전합니다.

```bash
git add streamlit_app.py requirements.txt .gitattributes .gitignore STREAMLIT_DEPLOY.md
git add outputs/efficientnetv2s_experiment_01/models/inference_config.json
git add outputs/efficientnetv2s_experiment_01/models/efficientnetv2s_fake_face_final.keras
git commit -m "Prepare Streamlit deployment"
git push origin main
```

GitHub에서 모델 파일을 열었을 때 Git LFS 포인터 정보가 보이면 정상입니다.

## 3. Streamlit Community Cloud에서 배포

1. <https://share.streamlit.io>에 GitHub 계정으로 로그인합니다.
2. `Create app`을 선택합니다.
3. Repository에 `hyojin0226/ai_image_detection`을 선택합니다.
4. Branch는 push한 브랜치를 선택합니다.
5. Main file path에 `streamlit_app.py`를 입력합니다.
6. Advanced settings에서 Python `3.12`를 선택합니다.
7. `Deploy`를 누릅니다.

배포가 끝나면 `https://...streamlit.app` 형식의 주소가 생성되며 다른 컴퓨터와 휴대폰에서도 접속할 수 있습니다.

## 로컬 네트워크에서만 잠깐 공유하기

같은 Wi-Fi 안에서 테스트할 때는 다음 명령을 실행합니다.

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

다른 컴퓨터에서 `http://실행한컴퓨터의-IP주소:8501`로 접속합니다. 이 방법은 인터넷 공개 배포가 아니며 실행 컴퓨터가 켜져 있어야 합니다.

## 문제 해결

- `Model file not found`: 모델이 Git LFS로 push되었는지 확인합니다.
- `This app has gone over its resource limits`: Streamlit Cloud 앱을 재부팅하고, 계속 발생하면 메모리가 더 큰 유료 호스팅을 사용합니다.
- 설치 중 TensorFlow 오류: Streamlit 배포 설정에서 Python 3.12를 선택했는지 확인합니다.
- 배포 로그: Streamlit 앱 화면의 `Manage app`에서 확인합니다.
