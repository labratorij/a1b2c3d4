import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .backbone_ibn import resnet50_ibn_a, resnet101_ibn_a

logger = logging.getLogger("reid")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TimmBackbone(nn.Module):
    """Любая CNN из timm как backbone: forward -> карта признаков (B, C, H/s, W/s).

    Основной сценарий - CLIP-претрейн: `timm:resnet50_clip.openai`, `timm:resnet101_clip.openai`
    (image-encoder CLIP от OpenAI, 400M пар картинка-текст; берётся конв-ствол без attention-pool).
    Годятся и другие: `timm:resnest50d`, `timm:seresnext101_32x8d` и т.п.
    last_stride=1 реализован через output_stride=16 (dilation в последней стадии - стандарт timm).
    Нормализация входа берётся из pretrained_cfg модели (у CLIP она не ImageNet'овская).
    """

    def __init__(self, name: str, pretrained: bool, last_stride: int, img_size=None):
        super().__init__()
        import timm
        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="")
        probe = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
        # ViT-семейство отдаёт токены (B, N+prefix, C), свёрточные - карту (B, C, H, W).
        # У ViT нет output_stride, зато нужен img_size, если вход не совпадает с родным
        # (позиционные эмбеддинги интерполируются timm'ом).
        self.is_vit = hasattr(probe, "patch_embed")
        if self.is_vit:
            if img_size is not None:
                kwargs["img_size"] = tuple(img_size)
        elif last_stride == 1:
            kwargs["output_stride"] = 16
        del probe
        self.net = timm.create_model(name, **kwargs)
        self.num_prefix_tokens = int(getattr(self.net, "num_prefix_tokens", 0)) if self.is_vit else 0
        self.out_channels = self.net.num_features
        cfg = getattr(self.net, "pretrained_cfg", {}) or {}
        self.input_mean = tuple(cfg.get("mean", IMAGENET_MEAN))
        self.input_std = tuple(cfg.get("std", IMAGENET_STD))
        self._grad_checkpoint = False

    @property
    def grad_checkpoint(self):
        return self._grad_checkpoint

    @grad_checkpoint.setter
    def grad_checkpoint(self, value: bool):
        self._grad_checkpoint = bool(value)
        self.net.set_grad_checkpointing(self._grad_checkpoint)

    def forward(self, x):
        feat = self.net.forward_features(x)
        if feat.dim() == 4:                      # свёрточный backbone: уже (B, C, H, W)
            return feat
        # ViT: убираем служебные токены (CLS и т.п.) и раскладываем патчи в 2D-сетку,
        # чтобы дальше работали GeM/GAP и PCB-полосы по высоте
        feat = feat[:, self.num_prefix_tokens:]
        b, n, c = feat.shape
        gh, gw = self.net.patch_embed.grid_size
        if gh * gw != n:                         # dynamic_img_size: сетка не совпала с конфигом
            gh = gw = int(round(n ** 0.5))
            if gh * gw != n:
                raise RuntimeError(f"не удалось разложить {n} токенов в сетку для {type(self.net).__name__}")
        return feat.transpose(1, 2).reshape(b, c, gh, gw)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class GeM(nn.Module):
    """Generalized-mean pooling (Radenovic et al., TPAMI 2019): (mean(x^p))^(1/p),
    p обучаемый. p=1 - обычный GAP, p->inf - max-pool; стартуем с p=3.
    Стандарт в fast-reid "stronger baseline" (SBS) вместо GAP."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # clamp(min=eps): после ReLU карта >= 0, eps защищает степень от нуля;
        # считаем в fp32 - под autocast x^p при p~3 в fp16 переполняется
        x = x.float().clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)

    def extra_repr(self):
        return f"p={self.p.item():.3f}"


class MarginClassifier(nn.Module):
    """ID-классификатор без bias с опциональным margin-softmax.

    kind='linear'  - обычный nn.Linear(bias=False), как в Bag of Tricks (логиты = W·f).
    Остальные варианты работают на косинусе cos = <f/|f|, w/|w|>, умноженном на scale s,
    и штрафуют целевой класс margin'ом m - это заставляет модель напрямую сжимать
    внутриклассовый разброс косинусов (важно для порога "свой/чужой" в candidates.csv):
      cosface  (Wang et al. 2018, AM-Softmax): target = s·(cos - m)
      arcface  (Deng et al. 2019):             target = s·cos(θ + m)
      circle   (Sun et al. 2020, CircleSoftmax из fast-reid): адаптивные веса
               α_p = relu(1 + m - cos), α_n = relu(cos + m); target = s·α_p·(cos - (1 - m)),
               other = s·α_n·(cos - m)
    Параметр называется `weight` с той же формой, что у nn.Linear, поэтому старые
    чекпоинты (classifier.weight) грузятся без переименований.
    В eval / без labels возвращает s·cos (или W·f для linear) - в инференсе не используется.
    """

    KINDS = ("linear", "cosface", "arcface", "circle")

    def __init__(self, feat_dim: int, num_classes: int, kind: str = "linear",
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown head kind: {kind} (ожидается один из {self.KINDS})")
        self.kind = kind
        self.scale = float(scale)
        self.margin = float(margin)
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        if kind == "linear":
            nn.init.normal_(self.weight, std=0.001)  # weights_init_classifier из BoT
        else:
            nn.init.normal_(self.weight, std=0.01)   # веса нормируются, масштаб init не важен

    def forward(self, feat: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        if self.kind == "linear":
            return F.linear(feat, self.weight)

        # косинусы считаем в fp32 с выключенным autocast (иначе F.linear всё равно уйдёт в fp16,
        # а acos и умножение на s=30..64 в fp16 теряют точность)
        with torch.autocast(device_type=feat.device.type, enabled=False):
            cos = F.linear(F.normalize(feat.float(), dim=1), F.normalize(self.weight.float(), dim=1))
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None or not self.training:
            return self.scale * cos

        one_hot = F.one_hot(labels, self.num_classes).bool()
        m, s = self.margin, self.scale
        if self.kind == "cosface":
            logits = torch.where(one_hot, cos - m, cos) * s
        elif self.kind == "arcface":
            target = torch.cos(torch.acos(cos) + m)
            # easy-margin fallback: если θ + m > π, cos(θ+m) перестаёт быть монотонным - берём cos - m·sin(m)
            target = torch.where(cos > math.cos(math.pi - m), target, cos - m * math.sin(m))
            logits = torch.where(one_hot, target, cos) * s
        else:  # circle
            alpha_p = torch.clamp_min(1 + m - cos, 0.0)
            alpha_n = torch.clamp_min(cos + m, 0.0)
            s_p = s * alpha_p * (cos - (1 - m))
            s_n = s * alpha_n * (cos - m)
            logits = torch.where(one_hot, s_p, s_n)
        return logits

    def extra_repr(self):
        return f"kind={self.kind}, scale={self.scale}, margin={self.margin}"


def _build_pooling(pooling: str) -> nn.Module:
    if pooling == "gem":
        return GeM()
    if pooling == "avg":
        return nn.AdaptiveAvgPool2d(1)
    raise ValueError(f"unknown pooling: {pooling} (ожидается 'avg' или 'gem')")


def _build_backbone(backbone: str, pretrained: bool, last_stride: int = 2, img_size=None):
    if backbone.startswith("timm:"):
        net = TimmBackbone(backbone[len("timm:"):], pretrained, last_stride, img_size=img_size)
        return net, net.out_channels
    if backbone == "resnet50_ibn_a":
        net = resnet50_ibn_a(pretrained=pretrained, last_stride=last_stride)
        return net, net.out_channels
    if backbone == "resnet101_ibn_a":
        net = resnet101_ibn_a(pretrained=pretrained, last_stride=last_stride)
        return net, net.out_channels
    if backbone in ("resnet50", "resnet101"):
        if backbone == "resnet101":
            from torchvision.models import resnet101, ResNet101_Weights
            net = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if last_stride == 1:
            # torchvision: даунсемпл в первом блоке layer4 сидит в conv2 (3x3) и в downsample-conv
            net.layer4[0].conv2.stride = (1, 1)
            net.layer4[0].downsample[0].stride = (1, 1)
        return nn.Sequential(*list(net.children())[:-2]), 2048
    raise ValueError(f"unknown backbone: {backbone}")


class ReIDModel(nn.Module):
    """Backbone + GAP + BNNeck (Bag of Tricks, Luo et al. 2019).

    forward возвращает:
      - global_feat: признак ДО BN (используется для triplet loss)
      - bn_feat: признак ПОСЛЕ BN (используется как эмбеддинг на инференсе и для ID-classifier)
      - cls_score: логиты классификатора (только при training и известном num_classes)
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50_ibn_a", pretrained: bool = True,
                 last_stride: int = 2, pooling: str = "avg",
                 head: str = "linear", head_scale: float = 30.0, head_margin: float = 0.3,
                 img_size=None):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone, pretrained, last_stride, img_size)
        self.feat_dim = feat_dim
        self.out_dim = feat_dim
        self.gap = _build_pooling(pooling)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # no bias per Bag-of-Tricks
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = MarginClassifier(feat_dim, num_classes, head, head_scale, head_margin)

    def forward(self, x, labels=None):
        feat_map = self.backbone(x)
        global_feat = self.gap(feat_map).flatten(1)  # (B, feat_dim), до BN
        bn_feat = self.bottleneck(global_feat)        # (B, feat_dim), после BN

        if self.training:
            cls_score = self.classifier(bn_feat, labels)  # labels нужны margin-головам (cosface/arcface/circle)
            return global_feat, bn_feat, cls_score
        return bn_feat

    @torch.no_grad()
    def extract(self, x) -> torch.Tensor:
        was_training = self.training
        self.eval()
        feat = self.forward(x)
        self.train(was_training)
        return feat


def _init_reduce_block(block: nn.Sequential):
    conv, bn = block[0], block[1]
    nn.init.kaiming_normal_(conv.weight, a=0, mode="fan_out")
    nn.init.constant_(bn.weight, 1.0)
    nn.init.constant_(bn.bias, 0.0)


class PartReIDModel(nn.Module):
    """Backbone + глобальная ветка (GAP+BNNeck, как в ReIDModel) + N локальных
    веток по горизонтальным полосам карты признаков (в духе PCB, Sun et al.
    2018 "Beyond Part Models"): карта признаков режется на num_parts равных
    полос по высоте, каждая полоса пулится и учится своим ID-классификатором.

    Инференс (self.eval()) возвращает ОДИН тензор - конкатенацию
    [global_bn, local_bn_1, ..., local_bn_N] - точно как ReIDModel.forward
    в eval-режиме, поэтому extract_features/inference-код не меняется.
    Триплет-лосс считается только на глобальной ветке (как в классическом PCB).
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50_ibn_a", pretrained: bool = True,
                 num_parts: int = 3, part_dim: int = 256, last_stride: int = 2, pooling: str = "avg",
                 head: str = "linear", head_scale: float = 30.0, head_margin: float = 0.3,
                 img_size=None):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone, pretrained, last_stride, img_size)
        self.feat_dim = feat_dim
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.out_dim = feat_dim + num_parts * part_dim

        self.gap = _build_pooling(pooling)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = MarginClassifier(feat_dim, num_classes, head, head_scale, head_margin)

        self.part_pool = nn.AdaptiveAvgPool2d(1)  # полосы - всегда GAP, как в PCB
        self.part_reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim, part_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(part_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_parts)
        ])
        for block in self.part_reduce:
            _init_reduce_block(block)

        self.part_bottleneck = nn.ModuleList([nn.BatchNorm1d(part_dim) for _ in range(num_parts)])
        for bn in self.part_bottleneck:
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)

        # у полос та же голова, что у глобальной ветки (margin-softmax на 256-d косинусах работает так же)
        self.part_classifier = nn.ModuleList([
            MarginClassifier(part_dim, num_classes, head, head_scale, head_margin) for _ in range(num_parts)
        ])

    def _split_stripes(self, feat_map: torch.Tensor):
        h = feat_map.shape[2]
        bounds = [round(i * h / self.num_parts) for i in range(self.num_parts + 1)]
        bounds[-1] = h
        stripes = []
        for i in range(self.num_parts):
            h0, h1 = bounds[i], max(bounds[i + 1], bounds[i] + 1)
            stripes.append(feat_map[:, :, h0:h1, :])
        return stripes

    def forward(self, x, labels=None):
        feat_map = self.backbone(x)
        global_feat = self.gap(feat_map).flatten(1)
        global_bn = self.bottleneck(global_feat)

        local_bns = []
        local_cls = []
        for i, stripe in enumerate(self._split_stripes(feat_map)):
            reduced = self.part_reduce[i](stripe)
            pooled = self.part_pool(reduced).flatten(1)
            bn_feat = self.part_bottleneck[i](pooled)
            local_bns.append(bn_feat)
            if self.training:
                local_cls.append(self.part_classifier[i](bn_feat, labels))

        if self.training:
            cls_score = self.classifier(global_bn, labels)
            return global_feat, global_bn, cls_score, local_bns, local_cls

        return torch.cat([global_bn] + local_bns, dim=1)


def load_external_weights(model: nn.Module, path: str) -> None:
    """Инициализация из внешнего ReID-чекпоинта (model.init_weights) - домен-претрейн.

    Поддерживаются:
      - fast-reid (JDAI-CV/fast-reid model zoo, напр. veri_sbs_R50-ibn.pth, обучен на VeRi-776):
        backbone.* -> backbone.* (имена IBN-слоёв совпадают), heads.bottleneck.0.* -> bottleneck.*,
        heads.pool_layer.p -> gap.p (GeM). Non-local блоки (NL_*) и классификатор пропускаются.
      - наши собственные чекпоинты (ключи 1:1), классификаторы пропускаются (другое число классов).
    Грузится всё, что совпало по имени и форме; сколько - в логе.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    mapped = {}
    for k, v in sd.items():
        if k.startswith("heads.bottleneck.0."):
            mapped["bottleneck." + k[len("heads.bottleneck.0."):]] = v
        elif k == "heads.pool_layer.p":
            mapped["gap.p"] = v
        elif k.startswith("heads.") or k.startswith("pixel_"):
            continue
        elif "classifier" in k:
            continue
        else:
            mapped[k] = v
    own = model.state_dict()
    loadable = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    skipped = sorted({k.split(".")[1] if k.startswith("backbone.") else k.split(".")[0]
                      for k in mapped if k not in loadable})
    model.load_state_dict(loadable, strict=False)
    n_backbone_own = sum(k.startswith("backbone.") for k in own)
    n_backbone_loaded = sum(k.startswith("backbone.") for k in loadable)
    logger.info(f"init_weights {os.path.basename(path)}: загружено {len(loadable)} тензоров "
                f"(backbone {n_backbone_loaded}/{n_backbone_own}, "
                f"{'bottleneck ' if any(k.startswith('bottleneck.') for k in loadable) else ''}"
                f"{'gap.p ' if 'gap.p' in loadable else ''}), пропущено: {skipped}")
    if n_backbone_loaded < n_backbone_own:
        logger.warning(f"init_weights: в backbone не инициализировано {n_backbone_own - n_backbone_loaded} тензоров")


def build_model(model_cfg: dict, num_classes: int, pretrained: bool) -> nn.Module:
    """Единая точка сборки модели по секции model конфига (train.py / infer.py / app.py).

    Значения по умолчанию (last_stride=2, pooling=avg, num_parts=0) соответствуют
    поведению до появления этих опций, поэтому старые конфиги/чекпоинты грузятся как раньше.
    pretrained=True + model.init_weights: ImageNet-претрейн не качается, backbone/neck
    инициализируются из внешнего ReID-чекпоинта (см. load_external_weights).
    Атрибуты model.input_mean / input_std - нормализация входа для transforms
    (ImageNet по умолчанию, у timm-backbone'ов - из их pretrained_cfg).
    """
    init_weights = model_cfg.get("init_weights")
    common = dict(
        backbone=model_cfg["backbone"],
        pretrained=pretrained and not init_weights,
        last_stride=int(model_cfg.get("last_stride", 2)),
        pooling=model_cfg.get("pooling", "avg"),
        head=model_cfg.get("head", "linear"),
        head_scale=float(model_cfg.get("head_scale", 30.0)),
        head_margin=float(model_cfg.get("head_margin", 0.3)),
        img_size=model_cfg.get("img_size"),     # для ViT: если вход не равен родному разрешению
    )
    num_parts = int(model_cfg.get("num_parts", 0))
    if num_parts > 0:
        model = PartReIDModel(num_classes, num_parts=num_parts,
                              part_dim=int(model_cfg.get("part_dim", 256)), **common)
    else:
        model = ReIDModel(num_classes, **common)
    if model_cfg.get("grad_checkpoint", False):
        if not hasattr(model.backbone, "grad_checkpoint"):
            raise ValueError(f"grad_checkpoint поддерживается IBN- и timm-backbone'ами, не {model_cfg['backbone']}")
        model.backbone.grad_checkpoint = True
    if pretrained and init_weights:
        load_external_weights(model, init_weights)
    model.input_mean = tuple(getattr(model.backbone, "input_mean", IMAGENET_MEAN))
    model.input_std = tuple(getattr(model.backbone, "input_std", IMAGENET_STD))
    return model
