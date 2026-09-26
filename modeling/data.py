"""크롭 → 텐서, 부위별 배치. 노트북에서 가져다 쓴다.

DataLoader 워커는 윈도우에서 새 프로세스로 뜨는데(spawn), 노트북 안에서 정의한 클래스는
새 프로세스가 찾지 못한다. 그래서 워커로 넘어가는 Dataset은 이 파일에 둔다.
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info

# 전처리와 학습이 같은 정규화 규칙을 쓰도록 전처리 모듈에서 가져온다
sys.path.append(str(Path(__file__).resolve().parent.parent / "data_preprocessing"))
from face_regions import normalize_crop  # noqa: E402

PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet 사전학습 기준


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread는 윈도우에서 한글 경로를 못 연다. 바이트로 읽어 디코딩한다."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


class RegionDataset(Dataset):
    """크롭 → 이미지 텐서, 표준화된 정답 (항목 수,), 마스크 (항목 수,), 행 번호

    tasks: 노트북의 TASKS. 출력 순서와 log 변환 여부를 여기서 읽는다
    """

    def __init__(self, rows, stats, cfg, tasks, crop_dir, train=False):
        self.rows, self.stats, self.cfg, self.train = rows, stats, cfg, train
        self.tasks, self.crop_dir = tasks, Path(crop_dir)
        self._rng = None

    def __len__(self):
        return len(self.rows)

    @property
    def rng(self):
        # 워커마다 다른 난수를 쓰도록 첫 사용 시점에 만든다 (전부 같은 seed면 증강이 겹친다)
        if self._rng is None:
            worker = get_worker_info()
            self._rng = np.random.default_rng(self.cfg["seed"] + (worker.id if worker else 0))
        return self._rng

    def __getitem__(self, i):
        row = self.rows[i]
        rgba = read_image(self.crop_dir / row["file"])
        if rgba is None:
            raise FileNotFoundError(f"크롭을 읽지 못했다: {self.crop_dir / row['file']}")
        crop = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
        mask = (rgba[..., 3] > 0).astype(np.uint8)

        if self.cfg["mirror_right"] and row["region"].startswith("r_"):
            crop, mask = crop[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.train:
            crop, mask = self.augment(crop, mask)
        crop, mask = normalize_crop(crop, mask, row["face_h"], row["region"])

        image = torch.from_numpy(crop).permute(2, 0, 1).float() / 255
        image = (image - torch.tensor(PIXEL_MEAN)[:, None, None]) / torch.tensor(PIXEL_STD)[:, None, None]
        # 정규화 후 0은 '평균 색'이다. 다각형 밖을 0으로 두면 새까만 테두리 대신 무난한 회색이 된다
        image *= torch.from_numpy(mask).float()

        targets, valid = torch.zeros(len(self.tasks)), torch.zeros(len(self.tasks))
        for t, (task, spec) in enumerate(self.tasks.items()):
            if task in row["targets"]:
                value = row["targets"][task]
                value = np.log1p(value) if spec["log"] else value  # 노트북의 transform과 같은 규칙
                mean, std = self.stats[task]
                targets[t] = (value - mean) / std
                valid[t] = 1
        return image, targets, valid, i

    def augment(self, crop, mask):
        cfg = self.cfg
        h, w = crop.shape[:2]
        angle = self.rng.uniform(-cfg["aug_rotate"], cfg["aug_rotate"])
        scale = self.rng.uniform(1 - cfg["aug_scale"], 1 + cfg["aug_scale"])
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        crop = cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
        if cfg["aug_bright"] and self.rng.random() < 0.5:
            factor = self.rng.uniform(1 - cfg["aug_bright"], 1 + cfg["aug_bright"])
            crop = np.clip(crop * factor, 0, 255).astype(np.uint8)
        return crop, mask


class RegionBatchSampler(Sampler):
    """한 배치를 같은 부위끼리만 묶는다.

    부위마다 캔버스 크기가 달라서(이마 288x608, 볼 576x512) 섞으면 텐서 하나로 못 담는다.
    배치 안은 같은 부위, 배치 순서는 섞어서 학습이 한쪽 부위에 쏠리지 않게 한다.
    """

    def __init__(self, rows, batch_size, shuffle, drop_last, seed):
        self.by_region = defaultdict(list)
        for i, row in enumerate(rows):
            self.by_region[row["region"]].append(i)
        self.batch_size, self.shuffle, self.drop_last, self.seed = batch_size, shuffle, drop_last, seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
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
        return sum(len(v) // self.batch_size if self.drop_last else int(np.ceil(len(v) / self.batch_size))
                   for v in self.by_region.values())
