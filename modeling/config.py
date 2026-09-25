"""학습 설정. 실험할 때 이 파일만 바꾸면 된다.

하이퍼파라미터 실험은 tune.ipynb에서 이 값들을 덮어쓰며 돌린다.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CROP_DIR = ROOT / "data" / "01.원천데이터"
INDEX_PATH = ROOT / "data" / "index.csv"
LABEL_PATH = ROOT / "data" / "02.정답지데이터" / "labels_region.csv"
TEST_SUBJECTS_PATH = ROOT / "ailabtools" / "test_set" / "test_subjects_165.json"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

REGIONS = ["forehead", "glabella", "l_eye", "r_eye", "l_cheek", "r_cheek"]
VIEWS = ["F"]  # 실험용은 정면만. 최종 학습은 ["F", "L", "R"]

# 항목 → (정답 열, 정답이 있는 부위, log 변환 여부)
#   log=True: 개수처럼 한쪽으로 치우친 값. log(1+x)로 바꿔 학습한다
TASKS = {
    "pore_count": dict(column="pore_count", regions=["l_cheek", "r_cheek"], log=True),
    "pore_grade": dict(column="pore_grade", regions=["l_cheek", "r_cheek"], log=False),
    "wrinkle_grade": dict(column="wrinkle_grade", regions=["forehead", "glabella", "l_eye", "r_eye"], log=False),
    "wrinkle_ra": dict(column="wrinkle_ra", regions=["l_eye", "r_eye"], log=False),
    "pigment_grade": dict(column="pigment_grade", regions=["forehead", "l_cheek", "r_cheek"], log=False),
    "acne_count": dict(column="acne_count", regions=["forehead", "l_cheek", "r_cheek"], log=True),
    "age": dict(column="age", regions=REGIONS, log=False),
}
TASK_NAMES = list(TASKS)
PRIMARY_TASK = "pore_count"  # 설정을 비교할 때 기준으로 삼는 항목
DETACHED_TASKS = ()  # 백본에 영향을 주지 않을 항목 (예: ("age",))

MIN_SKIN_PX = 20_000  # 피부가 이보다 적은 크롭은 제외 (가려졌거나 추출이 이상한 경우)
MIRROR_RIGHT = True  # 오른쪽 부위를 좌우 반전해 왼쪽과 방향을 맞춘다

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
FREEZE_STAGES = 2  # 백본 앞 몇 단계를 얼릴지 (0~4). 데이터가 적어서 앞쪽은 ImageNet 가중치를 그대로 쓴다
HEAD_HIDDEN = 256  # 헤드 중간층 노드 수 (0이면 768→1 한 층)
HEAD_ACT = "gelu"  # gelu | relu | silu
DROPOUT = 0.2

FOLDS = 5  # 데이터를 몇 등분할지 (돌릴 fold 수는 노트북에서 따로 정한다)
EPOCHS = 15
PATIENCE = 4  # 검증 점수가 이만큼 좋아지지 않으면 중단
BATCH_SIZE = 8
LR_BACKBONE = 2e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.05

AUG_ROTATE = 10  # 회전 ±도
AUG_SCALE = 0.05  # 크기 ±비율
AUG_BRIGHT = 0.1  # 밝기 ±비율

NUM_WORKERS = 4
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet 사전학습 가중치에 맞춘 픽셀값 정규화
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
