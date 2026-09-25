"""학습·평가 루프. train.ipynb(최종 학습)와 tune.ipynb(설정 실험)가 같이 쓴다."""
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

import config
import data
from model import SkinModel


def make_loader(rows, stats, train):
    dataset = data.RegionDataset(rows, stats, train=train)
    sampler = data.RegionBatchSampler(rows, config.BATCH_SIZE, shuffle=train, drop_last=train)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.NUM_WORKERS,
                      persistent_workers=config.NUM_WORKERS > 0)


def masked_loss(pred, target, valid, loss_fn):
    """정답이 있는 칸만 손실에 넣는다. 항목별로 평균낸 뒤 항목끼리 평균 — 칸 수가 많은 항목이 독차지하지 않게."""
    losses = []
    for t in range(pred.shape[1]):
        mask = valid[:, t] > 0
        if mask.any():
            losses.append(loss_fn(pred[mask, t], target[mask, t]))
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def predict(net, loader):
    """표준화된 예측값을 그대로 반환한다 (원래 단위 변환은 evaluate에서)."""
    net.eval()
    preds, valids, indices = [], [], []
    with torch.no_grad():
        for images, _, valid, index in loader:
            preds.append(net(images.to(config.DEVICE)).cpu().numpy())
            valids.append(valid.numpy())
            indices.append(index.numpy())
    return np.concatenate(preds), np.concatenate(valids), np.concatenate(indices)


def evaluate(net, loader, rows, stats):
    """항목별 성능. 같은 (사람, 부위)의 여러 사진 예측을 평균한 뒤 순위 상관과 MAE를 잰다."""
    preds, valids, indices = predict(net, loader)
    results = {}
    for t, task in enumerate(config.TASK_NAMES):
        grouped = defaultdict(list)
        truth = {}
        mean, std = stats[task]
        for pred, valid, i in zip(preds[:, t], valids[:, t], indices):
            if not valid:
                continue
            row = rows[i]
            key = (row["subject"], row["region"])
            grouped[key].append(data.inverse(task, pred * std + mean))
            truth[key] = row["targets"][task]
        if len(grouped) < 10:
            continue
        keys = list(grouped)
        pred_mean = np.array([np.mean(grouped[k]) for k in keys])
        actual = np.array([truth[k] for k in keys])
        results[task] = dict(spearman=float(spearmanr(pred_mean, actual).statistic),
                             mae=float(np.abs(pred_mean - actual).mean()), n=len(keys))
    return results


def mean_spearman(results):
    return float(np.mean([r["spearman"] for r in results.values()])) if results else 0.0


def train_fold(train_rows, valid_rows, epochs=None, verbose=True, save_path=None):
    """한 fold 학습. 검증 점수가 가장 좋았던 시점의 결과를 돌려준다."""
    epochs = epochs or config.EPOCHS
    stats = data.target_stats(train_rows)  # 검증 fold 정보는 쓰지 않는다
    train_loader = make_loader(train_rows, stats, train=True)
    valid_loader = make_loader(valid_rows, stats, train=False)

    net = SkinModel().to(config.DEVICE)
    optimizer = torch.optim.AdamW(net.param_groups(), weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    loss_fn = nn.SmoothL1Loss()  # 이상치(모공 2000개 같은 경우)에 덜 휘둘린다

    best = dict(score=-1, epoch=0, results={}, history=[])
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        net.train()
        started, total = time.time(), 0.0
        for images, targets, valid, _ in train_loader:
            optimizer.zero_grad()
            loss = masked_loss(net(images.to(config.DEVICE)), targets.to(config.DEVICE), valid.to(config.DEVICE), loss_fn)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()

        results = evaluate(net, valid_loader, valid_rows, stats)
        score = results.get(config.PRIMARY_TASK, {}).get("spearman", 0)
        best["history"].append(dict(epoch=epoch, loss=total / len(train_loader),
                                    primary=score, mean=mean_spearman(results),
                                    **{f"{k}_spearman": v["spearman"] for k, v in results.items()}))
        if verbose:
            summary = "  ".join(f"{k[:9]} {v['spearman']:.3f}" for k, v in results.items())
            print(f"  epoch {epoch:2d} loss {total / len(train_loader):.4f} | {summary} | "
                  f"평균 {mean_spearman(results):.3f} ({time.time() - started:.0f}초)", flush=True)

        if score > best["score"]:
            best.update(score=score, epoch=epoch, results=results, stats=stats)
            if save_path:
                torch.save({"model": net.state_dict(), "stats": stats}, save_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                if verbose:
                    print(f"  {config.PATIENCE}번 좋아지지 않아 중단")
                break
    return best


def split_folds(rows, n_splits=None):
    """사람 단위로 나눈다. 같은 사람의 사진은 정답이 같아서 섞이면 점수가 부풀려진다."""
    groups = [r["subject"] for r in rows]
    splitter = GroupKFold(n_splits=n_splits or config.FOLDS)
    return [([rows[i] for i in train_idx], [rows[i] for i in valid_idx])
            for train_idx, valid_idx in splitter.split(rows, groups=groups)]
