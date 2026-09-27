import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        smoothed = (1 - self.epsilon) * one_hot + self.epsilon / self.num_classes
        return (-smoothed * log_probs).sum(dim=1).mean()


def _pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
    """(N, N) матрица евклидовых расстояний между строками x."""
    sq = (x ** 2).sum(dim=1, keepdim=True)
    dist = sq + sq.t() - 2 * x @ x.t()
    return dist.clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al., 2017) с soft-margin или заданным margin.

    cross_camera=True: самый далёкий позитив выбирается только среди кадров того же
    ТС с ДРУГОЙ камеры (если в батче таких нет - откат на обычный batch-hard).
    Мотивация: в замерах по эталонному протоколу однокамерные пары имеют медиану
    косинуса 0.845, а кросс-камерные - 0.311, и именно кросс-камерные идут в зачёт
    (однокамерные выбрасываются junk-фильтрацией). Обычный batch-hard почти всегда
    подсовывает в качестве "трудного" позитива однокамерный кадр, т.е. учит лёгкому.
    """

    def __init__(self, margin: float = 0.3, cross_camera: bool = False):
        super().__init__()
        self.margin = margin
        self.cross_camera = cross_camera
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) if margin > 0 else None

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                cameras: torch.Tensor = None) -> torch.Tensor:
        dist_mat = _pairwise_euclidean(features)
        labels = labels.view(-1, 1)
        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos
        neg_inf = torch.full_like(dist_mat, float("-inf"))

        dist_ap = torch.where(is_pos, dist_mat, neg_inf).max(dim=1)[0]
        if self.cross_camera and cameras is not None:
            cameras = cameras.view(-1, 1)
            pos_cross = is_pos & cameras.ne(cameras.t())
            has_cross = pos_cross.any(dim=1)
            ap_cross = torch.where(pos_cross, dist_mat, neg_inf).max(dim=1)[0]
            dist_ap = torch.where(has_cross, ap_cross, dist_ap)

        dist_an = torch.where(is_neg, dist_mat, torch.full_like(dist_mat, float("inf"))).min(dim=1)[0]

        y = torch.ones_like(dist_an)
        if self.ranking_loss is not None:
            return self.ranking_loss(dist_an, dist_ap, y)
        return F.soft_margin_loss(dist_an - dist_ap, y)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class CameraAdversarialHead(nn.Module):
    """Классификатор камеры через слой инверсии градиента (Ganin & Lempitsky, 2015).

    Голова учится предсказывать camera_id, а backbone из-за инверсии градиента -
    наоборот, делать признак неинформативным о камере. Живёт только в train.py и
    в model.state_dict() не попадает, поэтому чекпоинты/инференс не меняются.
    """

    def __init__(self, feat_dim: int, num_cameras: int, lambd: float = 0.1):
        super().__init__()
        self.lambd = lambd
        self.fc = nn.Linear(feat_dim, num_cameras)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, feat: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
        logits = self.fc(_GradReverse.apply(feat, self.lambd))
        return F.cross_entropy(logits, cameras)
