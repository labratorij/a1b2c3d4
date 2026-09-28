import logging

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger("reid")

IBN_MODEL_URLS = {
    "resnet50_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
    "resnet101_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth",
}


class IBN(nn.Module):
    def __init__(self, planes: int):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat([out1, out2], dim=1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ibn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if use_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNetIBN(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), last_stride: int = 2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1, use_ibn=True)
        self.layer2 = self._make_layer(128, layers[1], stride=2, use_ibn=True)
        self.layer3 = self._make_layer(256, layers[2], stride=2, use_ibn=True)
        self.layer4 = self._make_layer(512, layers[3], stride=last_stride, use_ibn=False)

        self.out_channels = 512 * BottleneckIBN.expansion
        self.grad_checkpoint = False

    def _make_layer(self, planes, blocks, stride, use_ibn):
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BottleneckIBN.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )
        layers = [BottleneckIBN(self.inplanes, planes, stride, downsample, use_ibn)]
        self.inplanes = planes * BottleneckIBN.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckIBN(self.inplanes, planes, use_ibn=use_ibn))
        return nn.Sequential(*layers)

    def _run_stage(self, stage, x):
        if self.grad_checkpoint and self.training and x.requires_grad:
            return checkpoint_sequential(stage, len(stage), x, use_reentrant=False)
        return stage(x)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self._run_stage(self.layer3, x)
        x = self._run_stage(self.layer4, x)
        return x


def _load_official_ibn(model: ResNetIBN, arch: str) -> bool:
    try:
        src = load_state_dict_from_url(IBN_MODEL_URLS[arch], progress=True, map_location="cpu")
    except Exception as e:  # noqa: BLE001 - любой сбой сети/файла -> fallback на torchvision
        logger.warning(f"не удалось загрузить официальные веса IBN-Net для {arch}: {e}")
        return False
    src = {k: v for k, v in src.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(src, strict=False)
    if missing or unexpected:
        logger.warning(f"IBN-Net {arch}: missing={missing[:5]}... unexpected={unexpected[:5]}...")
    logger.info(f"загружены официальные ImageNet-веса IBN-Net ({arch})")
    return True


def _load_pretrained_from_torchvision(model: ResNetIBN) -> None:
    src = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    dst = model.state_dict()

    for key in list(dst.keys()):
        if key in src and dst[key].shape == src[key].shape:
            dst[key] = src[key]
            continue
        if ".bn1.BN." in key:
            src_key = key.replace(".bn1.BN.", ".bn1.")
            if src_key in src:
                if dst[key].dim() == 0:
                    dst[key] = src[src_key].clone()
                else:
                    half = dst[key].shape[0]
                    dst[key] = src[src_key][-half:].clone()
    model.load_state_dict(dst)
    logger.info("IBN-a инициализирован из torchvision resnet50 (IN-часть без претрейна)")


def _build(arch: str, layers, pretrained: bool, last_stride: int) -> ResNetIBN:
    model = ResNetIBN(layers=layers, last_stride=last_stride)
    if pretrained and not _load_official_ibn(model, arch):
        if arch != "resnet50_ibn_a":
            raise RuntimeError(f"для {arch} нет fallback-претрейна из torchvision - нужны официальные веса IBN-Net")
        _load_pretrained_from_torchvision(model)
    return model


def resnet50_ibn_a(pretrained: bool = True, last_stride: int = 2) -> ResNetIBN:
    return _build("resnet50_ibn_a", (3, 4, 6, 3), pretrained, last_stride)


def resnet101_ibn_a(pretrained: bool = True, last_stride: int = 2) -> ResNetIBN:
    return _build("resnet101_ibn_a", (3, 4, 23, 3), pretrained, last_stride)
