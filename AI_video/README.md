# AI 영상 분류 모델

**v1.0.0 · 64프레임 + Grad-CAM · 연구용 최종 프로젝트 버전**

3~40초 영상의 실제 촬영 / AI 생성을 추정하는 프로그램이다. 기존 EfficientNetV2S를 고정하고, 영상 특징을 집계하는 RBF 분류기 2개의 평균 점수를 사용한다. 최종 영상 분류 부분은 GRU가 아니다. 임계값은 **0.5**이다.

## 실행

현재 노트북: **`start_64frame_model.bat`을 파일 탐색기에서 더블클릭 → 주소 또는 영상 파일 입력 → 분석 시작**. 기존 `test_latest_model.bat`도 같은 화면을 연다.

다른 컴퓨터에서도 이 브랜치의 코드를 내려받으면 포함된 가중치로 실행할 수 있다. Python 3.12를 설치한 뒤 `AI_video` 폴더에서 아래 명령을 실행한다.

GitHub의 Download ZIP으로 받거나, Git 사용 시 아래처럼 줄바꿈 자동 변환을 끄고 받는다. 가중치·코드의 무결성 해시를 보존하기 위한 설정이다.

```powershell
git clone --config core.autocrlf=false --branch study/KongDuHyun https://github.com/hyojin0226/ai_image_detection.git
cd ai_image_detection/AI_video
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# 학습된 가중치는 models/final_research_v1/에 이미 포함되어 있다.
.venv\Scripts\python tools/release.py check
.venv\Scripts\python src/analysis64_window.py
```

최종 이미지 모델과 영상 분류 가중치 18개 파일은 [`models/final_research_v1/`](models/final_research_v1/)에 포함되어 있다. 별도 다운로드나 재학습은 필요하지 않다. `tools/release.py check`로 코드와 가중치 파일의 무결성을 확인할 수 있다. 로컬 파일 분석은 인터넷 없이 동작한다. YouTube 입력은 공개 세로 영상의 다운로드가 가능한 경우 지원하며 인터넷과 Node.js가 필요할 수 있다.

## 동작과 결과

`영상 → 8개 구간 × 8장 → 비율 유지·256×256 RGB → EfficientNetV2S → 구간 특징·RBF 점수 → 구간 평균 → 판별`

- 입력은 **3~40초, 최대 90 MB**이며 세로 영상을 우선 대상으로 한다. 로컬 가로 영상도 비율을 유지한다.
- 메인에는 최종 점수에 대한 민감도가 큰 **주요 프레임 8개**, **세부사항**에는 나머지 **56개**와 구간별 점수를 표시한다.
- 모든 분석 프레임의 원본과 **Grad-CAM**을 나란히 표시한다. 클릭하면 확대한다. 실제 촬영으로 판별하면 실제 촬영 판단을 지지하는 영역을 표시한다.
- 검은 화면만 제외한다. 유효 시점이 부족한 구간은 미분석으로 명시하며, 이 경우 분석 프레임 수가 64개보다 적을 수 있다.
- 약 16초 개발용 영상 한 개에서 64프레임·Grad-CAM 생성에 약 **54초**가 걸렸다. 40초 영상은 아직 직접 시간 측정하지 않았다.

| 분석 방식 | 평가 자료 | 정확도 | 해석 |
| --- | --- | ---: | --- |
| 종전 전체 8프레임 | 검증 200개 | 90.0% | 모델 선택에 사용한 개발 검증 |
| 종전 전체 8프레임 | 기존 Test 1,000개 재비교 | 81.5% | 실제 오탐률 11.6%. 새 독립 시험이 아닌 참고 비교 |
| **현재 64프레임** | 기능 확인 완료 | **미측정** | 64개 Grad-CAM, 메인 8개·세부 56개 생성 확인 |

**한계:** 프로젝트 버전 확정은 목표 정확도 95% 달성을 뜻하지 않는다. 종전 정확도를 64프레임 방식에 적용할 수 없다. AI 시험 자료는 Sora에 한정되고, 30초 초과 AI 자료가 없어 해당 구간의 양쪽 분류 성능은 미검증이다. Grad-CAM은 합성 위치나 생성 사실의 증명이 아니며 프레임끼리 색의 진하기를 비교할 수 없다. AI 점수는 보정된 정답 확률이 아니다.

## 구조

```text
AI_video/
├── start_64frame_model.bat       # 실행
├── AI_video.code-workspace      # VS Code 실행·파일 검사
├── src/                        # 화면·64프레임 분석·필수 연결 모듈
├── configs/                    # 최종 설정
├── releases/v1.0.0.json         # 고정한 코드·가중치 해시와 모델 규격
├── tools/release.py            # 파일 검사·가중치 ZIP 생성
├── tests/                      # 분할·점수·Grad-CAM 검사
├── requirements.txt
├── models/final_research_v1/    # 실행에 필요한 학습된 가중치 포함
└── data/sample/                # train / valid / test → real / fake
```

학습: 실제 500개·AI 500개 / 검증: 실제 100개·AI 100개 / 기존 시험: 실제 500개·AI 500개이다. `data/sample/*.csv`에 배정 목록을 기록했다. 시험 자료는 학습에 사용하지 않았다.

원본 영상·캐시·과거 가중치·분석 결과·개인 환경은 GitHub 업로드에서 제외한다. 실행에 필요한 최종 가중치만 포함한다. 분석 결과는 `.local/analyzed_frames64/`에 저장된다. 과거 코드는 `archive/`에 로컬 보관하며, 오래된 안내 문서는 삭제했다. `start_final_model.bat`은 종전 8프레임 호환 실행 파일로 보존한다.

코드는 상위 프로젝트의 MIT 라이선스를 따른다. 데이터·가중치에는 별도의 재배포 조건이 적용될 수 있으므로 MIT를 일괄 적용하지 않는다. 이 브랜치에는 실행 코드와 최종 가중치를 함께 제공한다. 원본 학습·시험 영상은 재배포하지 않는다.
