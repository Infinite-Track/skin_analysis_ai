"""모델 정의. 노트북에서 가져다 쓴다.

공유 백본(ConvNeXt) 하나 + 항목별 헤드 여러 개.
전역 평균 풀링 덕분에 부위마다 입력 크기가 달라도 항상 768개 특징이 나오고,
헤드는 그 768개를 각자의 숫자(모공 개수, 주름 등급, 나이 ...)로 바꾼다.
헤드 하나는 백본의 0.7%라, 항목을 늘려도 계산 비용은 거의 그대로다.

하이퍼파라미터는 전부 생성자 인자다. 노트북에서 값만 바꿔 실험하면 된다.
"""
import timm
import torch
import torch.nn as nn

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class SkinModel(nn.Module):
    """부위 크롭 → 항목별 예측.

    tasks: 항목 이름 목록. 출력은 이 순서대로 (배치, 항목 수)
    freeze_stages: 백본 앞 몇 단계를 고정할지 (0~4). 데이터가 적을수록 크게
    head_hidden: 헤드 중간층 노드 수. 0이면 768→1 한 층
    detached_tasks: 백본에 기울기를 흘리지 않을 항목.
        잘 배워지지 않는 항목이 백본을 망가뜨리는 것(negative transfer)을 막는다
    """

    def __init__(self, tasks, backbone="convnext_tiny.fb_in22k_ft_in1k", pretrained=True,
                 freeze_stages=2, head_hidden=256, head_act="gelu", dropout=0.2, detached_tasks=()):
        super().__init__()
        self.tasks = list(tasks)
        self.detached_tasks = set(detached_tasks)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.heads = nn.ModuleDict({
            task: self._make_head(self.backbone.num_features, head_hidden, head_act, dropout)
            for task in self.tasks
        })
        if freeze_stages:
            self._freeze(freeze_stages)

    @staticmethod
    def _make_head(in_features, hidden, activation, dropout):
        if not hidden:
            return nn.Linear(in_features, 1)
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _freeze(self, stages):
        """앞쪽 단계는 경계·질감을 잡는 부분이라 ImageNet 가중치를 그대로 쓴다 (과적합 방지)."""
        for param in self.backbone.stem.parameters():
            param.requires_grad = False 
        for stage in self.backbone.stages[:stages]:
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        outputs = [self.heads[task](features.detach() if task in self.detached_tasks else features).squeeze(1)
                   for task in self.tasks]
        return torch.stack(outputs, dim=1)  # (배치, 항목 수)

    def param_groups(self, lr_backbone, lr_head):
        """백본은 작은 학습률(이미 잘 배운 상태), 헤드는 큰 학습률(랜덤에서 시작)."""
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        return [{"params": backbone, "lr": lr_backbone},
                {"params": self.heads.parameters(), "lr": lr_head}]

    def summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        heads = sum(p.numel() for p in self.heads.parameters())
        return (f"전체 {total:,} / 학습 {trainable:,} / 헤드 {heads:,} ({heads / total * 100:.1f}%)"
                f" | 항목 {len(self.tasks)}개")
