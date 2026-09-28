import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .model import build_model
from .transforms import IMAGENET_MEAN, IMAGENET_STD
from .utils import load_checkpoint


def cfg_family(cfg: dict) -> str:
    if cfg.get("infer", {}).get("ensemble"):
        return "ensemble"
    return "transreid" if "transformer_type" in cfg.get("model", {}) else "resnet"


class ReIDWrapper(nn.Module):
    def __init__(self, model: nn.Module, forward_fn, image_size, input_mean, input_std, label_map, meta):
        super().__init__()
        self.model = model
        self._forward_fn = forward_fn
        self.image_size = tuple(image_size)
        self.input_mean = tuple(input_mean)
        self.input_std = tuple(input_std)
        self.label_map = label_map
        self.meta = meta

    def forward(self, x):
        return self._forward_fn(x)


class EnsembleWrapper(nn.Module):
    def __init__(self, members, names):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.names = list(names)
        first = members[0]
        self.image_size = first.image_size
        self.input_mean = first.input_mean
        self.input_std = first.input_std
        self.label_map = {}
        for m in members:
            self.label_map.update(m.label_map)
        self.meta = {"members": {n: m.meta for n, m in zip(names, members)}}
        self.out_dim = sum(getattr(m.model, "out_dim", 0) for m in members)

    def _adapt(self, x, member):
        if member.input_mean != self.input_mean or member.input_std != self.input_std:
            m0 = torch.tensor(self.input_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            s0 = torch.tensor(self.input_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            m1 = torch.tensor(member.input_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            s1 = torch.tensor(member.input_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x * s0 + m0 - m1) / s1
        if tuple(x.shape[-2:]) != member.image_size:
            x = F.interpolate(x, size=member.image_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        feats = [F.normalize(m(self._adapt(x, m)).float(), dim=1) for m in self.members]
        return torch.cat(feats, dim=1)


def _resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_reid_model(cfg: dict, ckpt_path, device: torch.device, base_dir: str = None):
    base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    family = cfg_family(cfg)

    if family == "ensemble":
        members, names = [], []
        for i, item in enumerate(cfg["infer"]["ensemble"]):
            with open(_resolve(item["config"], base_dir), "r", encoding="utf-8") as f:
                mcfg = yaml.safe_load(f)
            members.append(load_reid_model(mcfg, _resolve(item["checkpoint"], base_dir), device, base_dir))
            names.append(item.get("name") or os.path.splitext(os.path.basename(item["config"]))[0])
        return EnsembleWrapper(members, names)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if "label_map" not in ckpt:
        sd = ckpt.get("model", ckpt)
        clf = next((v for k, v in sd.items() if k.endswith("classifier.weight")), None)
        if clf is None:
            raise ValueError(f"{ckpt_path}: нет ни label_map, ни classifier.weight - "
                             f"не определить число классов")
        ckpt = {"model": sd, "label_map": {i: i for i in range(clf.shape[0])}}
    label_map = ckpt["label_map"]
    meta = {"epoch": ckpt.get("epoch"), "metrics": ckpt.get("metrics")}

    if family == "resnet":
        model = build_model(cfg["model"], len(label_map), pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ReIDWrapper(model, model, cfg["data"]["image_size"], model.input_mean, model.input_std, label_map, meta)

    from .transreid_model import build_transreid_model
    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    model, _, _ = build_transreid_model(len(label_map), 0, model_cfg,
                                        os.path.join(base_dir, "pretrained_cache", "transreid"))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return ReIDWrapper(model, lambda x: model(x, cam_label=None, view_label=None),
                       cfg["model"]["image_size"], IMAGENET_MEAN, IMAGENET_STD, label_map, meta)
