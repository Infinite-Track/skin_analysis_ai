"""mediapipe 랜드마크를 이어 얼굴 6부위(이마, 미간, 눈가 좌/우, 볼 좌/우)를 다각형으로 추출한다.

- 좌/우는 데이터셋 라벨(l_/r_)과 같은 '사진 기준'이다. 정면 사진에서 화면 왼쪽이 좌.
- 우측 부위는 좌측 윤곽의 좌우 대칭 랜드마크를 쓴다.
- mediapipe 랜드마크는 이마 중간(10번)까지만 있어서 이마 윗변은 위로 늘린다 (FOREHEAD_EXTEND).
- 다각형은 피부 분할 모델의 '얼굴 피부' 영역과 겹치는 부분만 남긴다.
  → 측면 사진의 배경, 앞머리, 눈 가림 막대가 빠진다.
- 정규화(배율 통일 + 부위별 고정 크기)도 이 파일에 있다. 학습할 때 모델 쪽에서 가져다 쓴다.
- mediapipe는 0.10.35 고정 (1.0.x는 이 Mac에서 Metal 초기화 오류로 크래시).
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
SEGMENTER_PATH = ROOT / "models" / "selfie_multiclass_256x256.tflite"
FACE_SKIN = 3  # 분할 클래스: 0 배경, 1 머리카락, 2 몸 피부, 3 얼굴 피부, 4 옷, 5 기타
BLACK_LEVEL = 15  # 이보다 어두우면 눈 가림 막대로 보고 뺀다 (막대는 값이 0, 피부는 100 이상)
DATA_DIR = ROOT.parent / "raw_data/028.한국인_피부상태_측정_데이터/3.개방데이터/1.데이터"

REGION_NAMES = ["forehead", "glabella", "l_eye", "r_eye", "l_cheek", "r_cheek"]

# 부위별 윤곽 랜드마크 (순서대로 이으면 다각형)
POLYGONS = {
    # 윗변 7점(103~332)은 위로 늘린다. 아랫변은 눈썹 윗선과 미간 윗선(9번).
    # 눈썹 꼬리 쪽 점(333, 293, 63, 104)은 옆으로 튀어나와 윤곽선이 꼬여서 뺐다.
    "forehead": [103, 67, 109, 10, 338, 297, 332, 334, 296, 336, 9, 107, 66, 105],
    # 눈썹 안쪽 끝 사이 ~ 콧대 시작
    "glabella": [107, 9, 336, 285, 417, 168, 193, 55],
    # 눈꼬리 바깥: 눈썹 꼬리 ~ 관자놀이 ~ 광대 위. 눈 쪽 변은 검은 막대 경계로 대체한다(EYE_BAR_EDGE)
    "l_eye": [70, 139, 143, 116, 111],
    "r_eye": [300, 368, 372, 345, 340],
    # 눈 밑 ~ 코 옆 ~ 입꼬리 옆 ~ 턱선 ~ 얼굴 옆선
    # 턱 쪽 점(204, 424)은 턱 안쪽으로 튀어나와 윤곽선이 톱니처럼 파여서 뺐다.
    "l_cheek": [116, 111, 228, 229, 230, 231, 232, 121, 47, 126, 209, 129, 203, 206, 216, 212, 202, 211, 170,
                149, 150, 136, 172, 58, 132, 137],
    "r_cheek": [345, 340, 448, 449, 450, 451, 452, 350, 277, 355, 429, 358, 423, 426, 436, 432, 422, 431, 395,
                378, 379, 365, 397, 288, 361, 366],
}
FOREHEAD_TOP = 7  # POLYGONS["forehead"] 앞 7점이 윗변
FOREHEAD_EXTEND = 0.5  # 이마 윗변을 (9번→10번 거리)의 몇 배만큼 위로 올릴지

# 눈가의 눈 쪽 변을 만들 때 쓰는 점: (위 끝, 아래 끝). x는 검은 막대 경계로 바꾼다.
EYE_BAR_EDGE = {"l_eye": (46, 111), "r_eye": (276, 340)}
EYE_BAR_MARGIN = 2  # 막대 경계에서 이만큼 떨어뜨린다 (막대 가장자리 픽셀 제외)

# 측면 사진에서 가려지는 부위. 사진 각도(F, Ft, Fb, L, L15, L30, R, R15, R30)의 첫 글자로 찾는다.
HIDDEN_IN_VIEW = {"L": ("l_eye", "l_cheek"), "R": ("r_eye", "r_cheek")}


def is_visible(view, region):
    """사진 각도 기준 가시성. L 사진은 좌 눈가·볼, R 사진은 우 눈가·볼이 가려진다 (덜 돌린 L15/R15도 동일)."""
    return region not in HIDDEN_IN_VIEW.get(view[0], ())


def create_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(landmarker, bgr):
    """랜드마크를 원본 픽셀 좌표 [(x, y), ...]로 반환한다. 얼굴을 못 찾으면 None."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    return [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]


def create_segmenter():
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(SEGMENTER_PATH), delegate=BaseOptions.Delegate.CPU),
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def face_skin_mask(segmenter, rgb, points):
    """사진 전체 크기의 얼굴 피부 마스크 (얼굴 피부=1).

    분할 모델 입력이 256x256이라 사진 전체를 넣으면 얼굴이 작아진다.
    그래서 랜드마크 주변(이마 확장분 포함)만 잘라서 분할한다.
    """
    p = np.asarray(points)
    h, w = rgb.shape[:2]
    (x1, y1), (x2, y2) = p.min(0), p.max(0)
    margin = 0.5 * (y2 - y1)
    x1, y1 = int(max(0, x1 - margin / 2)), int(max(0, y1 - margin))
    x2, y2 = int(min(w, x2 + margin / 2)), int(min(h, y2 + margin / 2))
    face = np.ascontiguousarray(rgb[y1:y2, x1:x2])
    category = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=face)).category_mask.numpy_view()
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = category.reshape(face.shape[:2]) == FACE_SKIN
    return mask


def face_height(points):
    """얼굴 세로 길이 (이마 10번 ~ 턱 152번, 픽셀). 고개를 좌우로 돌려도 거의 변하지 않아 배율 기준으로 쓴다."""
    p = np.asarray(points)
    return float(np.linalg.norm(p[10] - p[152]))


def eye_bar_box(bgr_or_rgb, points):
    """익명화용 검은 막대의 위치 (x1, y1, x2, y2). 못 찾으면 None.

    눈 높이에 있는 가로로 긴 새까만 덩어리를 찾는다.
    """
    p = np.asarray(points)
    face_h = face_height(points)
    eye_y = p[[33, 263], 1].mean()
    dark = (bgr_or_rgb.max(axis=2) <= BLACK_LEVEL).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        wide = w > 0.3 * face_h and h > 0.02 * face_h
        at_eye_level = abs(centers[i][1] - eye_y) < 0.2 * face_h
        if wide and at_eye_level and (best is None or area > best[0]):
            best = (area, (x, y, x + w, y + h))
    return best[1] if best else None


def region_polygons(points, eye_bar=None):
    """부위별 다각형 {부위: (N, 2) 픽셀 좌표 배열}.

    eye_bar를 주면 눈가의 눈 쪽 변을 막대 경계까지 늘린다 (막대 옆 피부를 끝까지 쓴다).
    없으면 랜드마크만으로 만든 좁은 눈가가 된다.
    """
    p = np.asarray(points, dtype=np.float64)
    polygons = {name: p[ids] for name, ids in POLYGONS.items()}
    # 9번(미간)→10번(이마 위) 방향으로 올리면 고개가 기울어져도 얼굴 기준 '위'로 늘어난다.
    polygons["forehead"][:FOREHEAD_TOP] += FOREHEAD_EXTEND * (p[10] - p[9])

    for name, (top_id, bottom_id) in EYE_BAR_EDGE.items():
        top, bottom = p[top_id], p[bottom_id]
        if eye_bar is None:
            edge = np.array([top, bottom])  # 막대가 없으면 랜드마크 그대로
        else:
            # 왼쪽 눈가는 막대의 왼쪽 끝까지, 오른쪽 눈가는 오른쪽 끝까지
            x = eye_bar[0] - EYE_BAR_MARGIN if name == "l_eye" else eye_bar[2] + EYE_BAR_MARGIN
            edge = np.array([[x, top[1]], [x, bottom[1]]])
        polygons[name] = np.vstack([polygons[name], edge[::-1]])
    return polygons


def crop_polygon(rgb, polygon, skin_mask):
    """다각형을 감싸는 사각형으로 자르고, 다각형 안이면서 얼굴 피부인 곳만 남긴다 (나머지는 0=검정).

    (crop, mask)를 반환한다. mask는 남긴 픽셀이 1. 남는 픽셀이 없으면 None.
    눈 가림 막대의 가장자리는 분할 모델이 피부로 잘못 보는 일이 있어서, 새까만 픽셀은 따로 뺀다.
    """
    h, w = rgb.shape[:2]
    x1, y1 = np.floor(polygon.min(0)).astype(int).clip(0, [w, h])
    x2, y2 = np.ceil(polygon.max(0)).astype(int).clip(0, [w, h])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - [x1, y1]).astype(np.int32)], 1)
    mask &= skin_mask[y1:y2, x1:x2]
    crop = rgb[y1:y2, x1:x2]
    mask &= crop.max(axis=2) > BLACK_LEVEL
    if not mask.any():
        return None
    return crop * mask[..., None], mask


def save_region_png(path, crop, mask):
    """크롭을 RGBA PNG로 저장한다. RGB = 크롭, 알파 = 픽셀 마스크 (피부 255, 나머지 0)."""
    rgba = np.dstack([crop, mask * 255])
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


# ---------------------------------------------------------------------------
# 정규화: 배율 통일 → 부위별 고정 크기 캔버스
#   기기·사람마다 사진 속 얼굴 크기가 달라서, 그냥 고정 크기로 늘이면
#   같은 모공이 사진마다 다른 픽셀 크기로 보인다. 그래서 얼굴 세로 길이를
#   TARGET_FACE_HEIGHT에 맞춘 뒤 캔버스에 담는다.
# ---------------------------------------------------------------------------
TARGET_FACE_HEIGHT = 900  # 이 길이를 기준으로 모든 사진의 배율을 맞춘다

# 부위별 캔버스 (가로, 세로)
CANVAS = {
    "forehead": (608, 288),
    "glabella": (192, 160),
    "l_eye": (192, 224),
    "r_eye": (192, 224),
    "l_cheek": (512, 576),
    "r_cheek": (512, 576),
}


def normalize_crop(crop, mask, face_h, region):
    """크롭을 배율 통일 후 부위 캔버스 가운데에 담는다. (이미지, 마스크) 반환.

    crop: RGB 배열, mask: 피부 1 / 나머지 0, face_h: 그 사진의 얼굴 세로 길이(픽셀).
    캔버스보다 크면 가운데를 기준으로 잘라내고, 작으면 남는 곳은 0(검정)으로 둔다.
    """
    scale = TARGET_FACE_HEIGHT / face_h
    # 축소에는 INTER_AREA가 모아레·계단현상이 적다. 마스크는 0/1이 유지되도록 최근접.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    crop = cv2.resize(crop, size, interpolation=interpolation)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    canvas_w, canvas_h = CANVAS[region]
    out = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    out_mask = np.zeros((canvas_h, canvas_w), np.uint8)

    # 캔버스보다 큰 쪽은 가운데만 남기고, 작은 쪽은 가운데에 놓는다.
    src_x, dst_x, w = _center_align(crop.shape[1], canvas_w)
    src_y, dst_y, h = _center_align(crop.shape[0], canvas_h)
    out[dst_y:dst_y + h, dst_x:dst_x + w] = crop[src_y:src_y + h, src_x:src_x + w]
    out_mask[dst_y:dst_y + h, dst_x:dst_x + w] = mask[src_y:src_y + h, src_x:src_x + w]
    return out, out_mask


def _center_align(src_len, dst_len):
    """한 축에 대해 (원본 시작점, 캔버스 시작점, 옮길 길이)를 구한다."""
    length = min(src_len, dst_len)
    return (src_len - length) // 2, (dst_len - length) // 2, length
