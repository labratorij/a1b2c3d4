import os
import sys

import torch

_TRANSREID_ROOT = os.environ.get(
    "TRANSREID_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "TransReID")),
)
if not os.path.isdir(_TRANSREID_ROOT):
    raise FileNotFoundError(
        f"TransReID repo not found at {_TRANSREID_ROOT}. "
        f"Клонируйте его туда либо укажите путь через переменную окружения TRANSREID_ROOT."
    )
if _TRANSREID_ROOT not in sys.path:
    sys.path.insert(0, _TRANSREID_ROOT)

from config import cfg as _default_cfg  # noqa: E402  (импорт из TransReID после правки sys.path)
from model.make_model import make_model as _make_model  # noqa: E402

_PRETRAIN_URLS = {
    "vit_base_patch16_224_TransReID": (
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_base_p16_224-80ecf9dd.pth",
        "jx_vit_base_p16_224-80ecf9dd.pth",
    ),
    "vit_small_patch16_224_TransReID": (
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/"
        "vit_small_p16_224-15ec54c9.pth",
        "vit_small_p16_224-15ec54c9.pth",
    ),
}


def _ensure_pretrained(transformer_type: str, cache_dir: str) -> str:
    if transformer_type not in _PRETRAIN_URLS:
        raise ValueError(f"нет известного URL претрейна для {transformer_type}; "
                          f"укажите model.pretrain_path в конфиге вручную")
    url, filename = _PRETRAIN_URLS[transformer_type]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        torch.hub.download_url_to_file(url, path)
    return path


def build_transreid_model(num_classes: int, num_cameras: int, model_cfg: dict, pretrain_cache_dir: str):
    cfg = _default_cfg.clone()
    cfg.MODEL.NAME = "transformer"
    cfg.MODEL.TRANSFORMER_TYPE = model_cfg.get("transformer_type", "vit_base_patch16_224_TransReID")
    cfg.MODEL.STRIDE_SIZE = list(model_cfg.get("stride_size", [16, 16]))
    cfg.MODEL.DROP_PATH = model_cfg.get("drop_path", 0.1)
    cfg.MODEL.DROP_OUT = model_cfg.get("drop_out", 0.0)
    cfg.MODEL.ATT_DROP_RATE = model_cfg.get("att_drop_rate", 0.0)

    cfg.MODEL.JPM = False
    cfg.MODEL.ID_LOSS_TYPE = "softmax"
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.NECK = "bnneck"
    cfg.MODEL.LAST_STRIDE = 1
    cfg.TEST.NECK_FEAT = "after"

    use_sie_camera = bool(model_cfg.get("sie_camera", False))
    cfg.MODEL.SIE_CAMERA = use_sie_camera
    cfg.MODEL.SIE_VIEW = False
    cfg.MODEL.SIE_COE = model_cfg.get("sie_coe", 3.0)

    size = list(model_cfg.get("image_size", [256, 256]))
    cfg.INPUT.SIZE_TRAIN = size
    cfg.INPUT.SIZE_TEST = size

    pretrained = model_cfg.get("pretrained", True)
    if pretrained:
        pretrain_path = model_cfg.get("pretrain_path") or _ensure_pretrained(
            cfg.MODEL.TRANSFORMER_TYPE, pretrain_cache_dir)
        cfg.MODEL.PRETRAIN_PATH = pretrain_path
        cfg.MODEL.PRETRAIN_CHOICE = "imagenet"
    else:
        cfg.MODEL.PRETRAIN_CHOICE = "none"

    camera_num = num_cameras if use_sie_camera else 0
    model = _make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    return model, cfg, use_sie_camera
