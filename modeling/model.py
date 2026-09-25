"""모델: 공유 백본(ConvNeXt) + 항목별 헤드.

백본은 부위와 항목 전부가 공유한다. 전역 평균 풀링 덕분에 부위마다 입력 크기가 달라도
항상 768개 특징이 나오고, 헤드는 그 768개를 각자의 숫자로 바꾼다.
헤드는 백본의 0.7%밖에 안 돼서, 항목을 늘려도 계산 비용은 거의 그대로다.
"""
import timm
import torch
import torch.nn as nn

import config

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def make_head(in_features):
    """768 → (중간층) → 1. HEAD_HIDDEN이 0이면 한 층짜리."""
    if not config.HEAD_HIDDEN:
        return nn.Linear(in_features, 1)
    return nn.Sequential(
        nn.Linear(in_features, config.HEAD_HIDDEN),
        ACTIVATIONS[config.HEAD_ACT](),
        nn.Dropout(config.DROPOUT),
        nn.Linear(config.HEAD_HIDDEN, 1),
    )


class SkinModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(config.BACKBONE, pretrained=pretrained, num_classes=0)
        self.heads = nn.ModuleDict({task: make_head(self.backbone.num_features) for task in config.TASK_NAMES})
        if config.FREEZE_STAGES:
            self.freeze(config.FREEZE_STAGES)

    def freeze(self, stages):
        """앞쪽 단계는 경계·질감을 잡는 부분이라 ImageNet 가중치를 그대로 쓴다 (과적합 방지)."""
        for param in self.backbone.stem.parameters():
            param.requires_grad = False
        for stage in self.backbone.stages[:stages]:
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        for task in config.TASK_NAMES:
            # DETACHED_TASKS는 헤드만 학습한다. 잘 안 배워지는 항목이 백본을 망가뜨리지 않게 막는 장치
            source = features.detach() if task in config.DETACHED_TASKS else features
            outputs.append(self.heads[task](source).squeeze(1))
        return torch.stack(outputs, dim=1)  # (B, 항목 수)

    def param_groups(self):
        """백본은 작은 학습률(이미 잘 배운 상태), 헤드는 큰 학습률(랜덤에서 시작)."""
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {"params": backbone, "lr": config.LR_BACKBONE},
            {"params": self.heads.parameters(), "lr": config.LR_HEAD},
        ]
