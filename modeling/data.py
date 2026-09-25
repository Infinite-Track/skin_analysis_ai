"""데이터셋: 저장된 부위 크롭(RGBA PNG) → 모델 입력 텐서 + 항목별 정답.

한 샘플 = 크롭 1장 (사람 × 기기 × 각도 × 부위).
크롭마다 그 부위에 정답이 있는 항목만 채우고, 나머지는 마스크로 표시해 loss에서 뺀다.
정답은 사람 단위라 같은 사람의 여러 사진이 같은 값을 갖는다. 그래서 나눌 때는 사람 단위여야 한다.
"""
import csv
import json
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

import config

sys.path.append(str(config.ROOT / "data_preprocessing"))
from face_regions import CANVAS, normalize_crop  # noqa: E402


def load_rows():
    """index.csv와 정답을 합쳐 학습에 쓸 크롭 목록을 만든다. 테스트 165명은 따로 뺀다."""
    labels = {}
    with open(config.LABEL_PATH, newline="") as f:
        for row in csv.DictReader(f):
            values = {}
            for task, spec in config.TASKS.items():
                value = row[spec["column"]]
                if value and row["region"] in spec["regions"]:
                    values[task] = float(value)
            labels[(row["subject"], row["region"])] = values

    test_subjects = {s["subject_id"] for s in json.loads(config.TEST_SUBJECTS_PATH.read_text())["subjects"]}

    rows, test_rows = [], []
    with open(config.INDEX_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row["region"] not in config.REGIONS or row["view"] not in config.VIEWS:
                continue
            if int(row["skin_px"]) < config.MIN_SKIN_PX:
                continue
            targets = labels.get((row["subject"], row["region"]))
            if not targets:
                continue
            sample = dict(subject=row["subject"], device=row["device"], view=row["view"], region=row["region"],
                          file=row["file"], face_h=float(row["face_h"]), targets=targets)
            (test_rows if row["subject"] in test_subjects else rows).append(sample)
    return rows, test_rows


def target_stats(rows):
    """항목마다 평균·표준편차를 구한다. 검증 fold 정보는 쓰지 않도록 학습 rows만 넣을 것.

    항목별로만 구하고 부위별로는 나누지 않는다. 한 항목 안에서는 부위가 달라도 단위가 같고
    (예: 주름 등급은 어느 부위든 0~6), 부위별 차이(볼이 이마보다 색소가 짙다)는 실제 차이라
    모델이 배워야 할 정보다. 부위별로 표준화하면 그 정보를 지우게 된다.
    """
    collected = {}
    for row in rows:
        for task, value in row["targets"].items():
            collected.setdefault(task, []).append(transform(task, value))
    return {task: (float(np.mean(v)), float(np.std(v)) or 1.0) for task, v in collected.items()}


def transform(task, value):
    """치우친 값은 log를 씌워 분포를 고르게 만든다."""
    return float(np.log1p(value)) if config.TASKS[task]["log"] else float(value)


def inverse(task, value):
    return float(np.expm1(value)) if config.TASKS[task]["log"] else float(value)


class RegionDataset(Dataset):
    """크롭을 읽어 배율을 맞추고 캔버스에 담아 텐서로 돌려준다.

    반환: 이미지, 표준화된 정답 (항목 수,), 마스크 (항목 수,), 행 번호
    """

    def __init__(self, rows, stats, train=False):
        self.rows = rows
        self.stats = stats
        self.train = train
        self._rng = None

    def __len__(self):
        return len(self.rows)

    @property
    def rng(self):
        # 워커마다 다른 난수를 쓰도록 첫 사용 시점에 만든다 (전부 같은 seed면 증강이 겹친다)
        if self._rng is None:
            worker = get_worker_info()
            self._rng = np.random.default_rng(config.SEED + (worker.id if worker else 0))
        return self._rng

    def __getitem__(self, i):
        row = self.rows[i]
        path = config.CROP_DIR / row["file"]
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"크롭을 읽지 못했다: {path}")
        crop = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
        mask = (rgba[..., 3] > 0).astype(np.uint8)

        if config.MIRROR_RIGHT and row["region"].startswith("r_"):
            crop, mask = crop[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.train:
            crop, mask = self.augment(crop, mask)

        crop, mask = normalize_crop(crop, mask, row["face_h"], row["region"])

        image = torch.from_numpy(crop).permute(2, 0, 1).float() / 255
        image = (image - torch.tensor(config.PIXEL_MEAN)[:, None, None]) / torch.tensor(config.PIXEL_STD)[:, None, None]
        # 정규화 후 0은 '평균 색'이다. 다각형 밖을 0으로 두면 새까만 테두리 대신 무난한 회색이 된다.
        image *= torch.from_numpy(mask).float()

        targets = torch.zeros(len(config.TASK_NAMES))
        valid = torch.zeros(len(config.TASK_NAMES))
        for t, task in enumerate(config.TASK_NAMES):
            if task in row["targets"]:
                mean, std = self.stats[task]
                targets[t] = (transform(task, row["targets"][task]) - mean) / std
                valid[t] = 1
        return image, targets, valid, i

    def augment(self, crop, mask):
        """회전·확대축소·밝기. 흐림이나 강한 색 변화는 모공·색소 신호를 망가뜨려서 쓰지 않는다."""
        h, w = crop.shape[:2]
        angle = self.rng.uniform(-config.AUG_ROTATE, config.AUG_ROTATE)
        scale = self.rng.uniform(1 - config.AUG_SCALE, 1 + config.AUG_SCALE)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        crop = cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
        if config.AUG_BRIGHT and self.rng.random() < 0.5:
            crop = np.clip(crop * self.rng.uniform(1 - config.AUG_BRIGHT, 1 + config.AUG_BRIGHT), 0, 255).astype(np.uint8)
        return crop, mask


class RegionBatchSampler(torch.utils.data.Sampler):
    """한 배치를 같은 부위끼리만 묶는다.

    부위마다 캔버스 크기가 달라서(이마 288x608, 볼 576x512) 섞으면 텐서 하나로 못 담는다.
    배치 안은 같은 부위, 배치 순서는 섞어서 학습이 한쪽 부위에 쏠리지 않게 한다.
    """

    def __init__(self, rows, batch_size, shuffle, drop_last):
        self.by_region = {}
        for i, row in enumerate(rows):
            self.by_region.setdefault(row["region"], []).append(i)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(config.SEED + self.epoch)
        self.epoch += 1
        batches = []
        for indices in self.by_region.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        total = 0
        for indices in self.by_region.values():
            count = len(indices) / self.batch_size
            total += int(count) if self.drop_last else int(np.ceil(count))
        return total


def input_sizes():
    return {region: CANVAS[region] for region in config.REGIONS}
