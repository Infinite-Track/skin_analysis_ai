# 얼굴 피부 분석 AI

얼굴 사진에서 **모공·주름·색소침착·여드름·피부나이**를 예측한다.
AI Hub의 [한국인 피부상태 측정 데이터](https://www.aihub.or.kr/)로 학습하고, 상용 API(ailabtools)와 성능을 비교한다.

사진 3장(정면·좌·우)을 받아 얼굴을 6부위로 나누고, 부위마다 피부 특징을 예측한다.

```
사진 3장 → mediapipe 랜드마크 → 6부위 다각형 크롭 → 배율 통일
        → ConvNeXt 백본(공유) → 항목별 헤드 7개 → 부위별 점수 → 최종 점수
```

## 설계에서 중요한 결정들

| 결정 | 이유 |
|---|---|
| 사각형이 아닌 **다각형**으로 부위 추출 | 사각형은 부위 밖 피부가 섞인다. 랜드마크를 이어 실제 부위 모양대로 자른다 |
| **얼굴 세로 길이로 배율 통일** | 기기·거리에 따라 얼굴 크기가 4배까지 차이 난다. 보정하지 않으면 "모공이 크다"와 "크게 찍혔다"를 구분할 수 없다 |
| 부위마다 **다른 입력 크기** | 이마는 가로로 길고(608×288) 볼은 세로로 길다(512×576). 하나로 통일하면 왜곡되거나 여백이 낭비된다 |
| **디지털카메라 사진 제외** | 검은 배경 + 스튜디오 조명이라 같은 사람도 색이 크게 달라진다. 기기 차이(10.8)가 사람 간 차이(7.4)보다 커서 색소 학습에 방해가 된다 |
| **크롭 1장 = 샘플 1개**로 학습 | 사람 단위로 학습하면 샘플이 800개뿐이다. 크롭 단위면 9,000개가 되고, 추론할 때 평균내면 같은 효과가 난다 |
| 정답은 **측정값 우선** | 전문가 등급은 61%가 한 등급에 몰려 있고 경계가 모호하다. 기기 측정값이 더 객관적이다 |
| 나눌 때 **사람 단위** | 같은 사람의 사진은 정답이 같다. 섞이면 점수가 부풀려진다 |

## 폴더 구조

```
face_analysis_AI/
├── raw_data/                    # AI Hub 원본 (git 제외, 43GB)
├── data/                        # 만들어지는 데이터 (git 제외, 11GB)
│   ├── 01.원천데이터/{사람}/{기기}/{각도}_{부위}.png
│   ├── 02.정답지데이터/labels_region.csv, labels_face.csv
│   └── index.csv                # 크롭 목록 + 가시성 + 얼굴 크기
├── data_preprocessing/
│   ├── face_regions.py          # 랜드마크·부위 다각형·크롭·정규화 (공용 모듈)
│   ├── explore_data.ipynb       # ① 분포·이상치·결측 점검
│   ├── face_regions.ipynb       # ② 한 사람으로 부위 분할 확인
│   ├── extract_regions.ipynb    # ③ 전체 사진 → 크롭 저장
│   └── build_labels.ipynb       # ④ 정답 CSV 생성
├── modeling/
│   ├── config.py                # 설정 (실험은 여기만 고친다)
│   ├── data.py                  # 크롭 → 텐서, 부위별 배치
│   ├── model.py                 # 백본 1개 + 헤드 7개
│   ├── engine.py                # 학습·평가 루프
│   ├── tune.ipynb               # ⑤ 하이퍼파라미터 실험
│   └── train.ipynb              # ⑥ 최종 학습 + ailabtools 비교
└── ailabtools/                  # 비교용 상용 API 결과 (테스트 165명)
```

## 처음 설정하기

### 1. 환경

```bash
python3 -m venv fa-ai
source fa-ai/bin/activate          # 윈도우: fa-ai\Scripts\activate

# NVIDIA GPU가 있으면 torch를 먼저 CUDA 버전으로
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. mediapipe 모델 받기

```bash
mkdir -p data_preprocessing/models
curl -L -o data_preprocessing/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o data_preprocessing/models/selfie_multiclass_256x256.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

- `face_landmarker.task` — 얼굴 랜드마크 478개
- `selfie_multiclass_256x256.tflite` — 배경·머리카락·얼굴 피부 분리

### 3. 데이터 놓기

AI Hub에서 받은 `028.한국인_피부상태_측정_데이터`를 압축 해제해 이 위치에 둔다.

```
raw_data/028.한국인_피부상태_측정_데이터/3.개방데이터/1.데이터/
├── Training/{01.원천데이터, 02.라벨링데이터}/{1. 디지털카메라, 2. 스마트패드, 3. 스마트폰}/
├── Validation/...
└── Other/메타데이터/
```

## 실행 순서

| 순서 | 노트북 | 하는 일 | 시간 |
|---|---|---|---|
| 1 | `data_preprocessing/explore_data.ipynb` | 분포·이상치·결측 확인 | 2분 |
| 2 | `data_preprocessing/extract_regions.ipynb` | 5,790장 → 크롭 26,754개 | 30분 |
| 3 | `data_preprocessing/build_labels.ipynb` | 정답 CSV 2개 생성 | 2분 |
| 4 | `modeling/tune.ipynb` | 하이퍼파라미터 실험 | 실험당 1시간(맥) / 20분(RTX) |
| 5 | `modeling/train.ipynb` | 최종 학습 + 테스트 비교 | 13시간(맥) / 3시간(RTX) |

부위 분할이 잘 되는지 눈으로 보려면 `face_regions.ipynb`를 아무 때나 실행하면 된다.

## 현재 결과

볼 모공 개수로 먼저 검증한 결과다 (테스트 165명, 학습에 쓰지 않은 사람들).

| 방법 | 스피어맨 | 피어슨 | MAE |
|---|---|---|---|
| **우리 모델** | **0.889** | 0.879 | 154개 |
| ailabtools | 0.519 | 0.482 | 단위 다름 |

- 정답: AI Hub 기기가 잰 모공 개수 (중앙값 834개)
- 나이만으로 예측하면 0.201이므로, 나이를 보고 찍는 게 아니라 실제로 피부 질감을 읽고 있다
- 참고로 피부나이는 ailabtools가 0.939로 매우 잘한다 (쉽게 이기기 어려운 항목)

## 예측 항목

부위마다 정답이 있는 칸만 학습한다 (없는 칸은 loss에서 제외).

| | 이마 | 미간 | 눈가 좌/우 | 볼 좌/우 |
|---|:-:|:-:|:-:|:-:|
| 모공 | · | · | · | 등급 + **개수** |
| 주름 | 등급 | 등급 | 등급 + **Ra** | · |
| 색소 | 등급 | · | · | 등급 |
| 여드름 | **개수** | · | · | **개수** |
| 피부나이 | **나이** | **나이** | **나이** | **나이** |

굵은 글씨는 기기 측정값·실제값이고, 나머지는 사람이 매긴 등급이다.
여드름은 디지털카메라 사진에만 좌표가 있어서, 그 사진에 랜드마크를 찍어 부위별 개수로 변환한다.

## 알아둘 점

- **mediapipe는 0.10.35 고정.** 1.0.x는 macOS arm64에서 Metal 초기화 오류로 크래시한다.
- **좌/우는 사진 기준이다.** 데이터셋 라벨(`l_`/`r_`)이 그렇게 되어 있다. 정면 사진에서 화면 왼쪽이 좌.
- **저해상도 사진 57장은 자동으로 제외된다.** 100만 픽셀 미만이면 배율을 맞추느라 3~4배 확대해야 해서 질감이 뭉개진다.
- **테스트 165명은 어디에도 쓰지 않는다.** `data.load_rows()`가 처음부터 분리한다.
- `.env`에 ailabtools API 키가 들어간다 (git 제외).
