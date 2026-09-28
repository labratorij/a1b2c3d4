import io
import os
import sys
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from . import settings

if settings.SERVICE_DIR not in sys.path:
    sys.path.insert(0, settings.SERVICE_DIR)
os.environ.setdefault("TORCH_HOME", settings.TORCH_CACHE)
os.environ.setdefault("HF_HOME", os.path.join(settings.TORCH_CACHE, "hf"))

from reid.dataset import _crop_bbox                     # noqa: E402
from reid.gradcam import GradCAM                        # noqa: E402
from reid.features import forward_with_tta              # noqa: E402
from reid.model_loader import load_reid_model           # noqa: E402
from reid.reranking import re_ranking                   # noqa: E402
from reid.transforms import build_test_transforms       # noqa: E402


class Engine:

    def __init__(self, config_path: str = None, device: str = None):
        cfg_rel = config_path or settings.MODEL_CONFIG
        cfg_path = cfg_rel if os.path.isabs(cfg_rel) else os.path.join(settings.SERVICE_DIR, cfg_rel)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        dev = device or settings.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model = load_reid_model(self.cfg, self.cfg["infer"].get("checkpoint"),
                                     self.device, base_dir=settings.SERVICE_DIR)
        self.flip_tta = bool(self.cfg["infer"].get("flip_tta", False))
        self.bbox_margin = float(self.cfg["data"].get("bbox_margin", 0.1))
        self.transform = build_test_transforms(self.model.image_size,
                                               self.model.input_mean, self.model.input_std)
        self.config_path = cfg_rel
        self.members = list(getattr(self.model, "names", []))
        self._lock = threading.Lock()
        self._gradcam = None

    def describe(self) -> dict:
        return {
            "config": self.config_path,
            "device": str(self.device),
            "input_size": list(self.model.image_size),
            "flip_tta": self.flip_tta,
            "embedding_dim": int(getattr(self.model, "out_dim", 0)) or None,
            "ensemble_members": self.members or None,
            "bbox_margin": self.bbox_margin,
        }

    def prepare(self, image: Image.Image, bbox: Optional[Sequence[float]]) -> torch.Tensor:
        if bbox is not None:
            x, y, w, h = bbox
            image = _crop_bbox(image.convert("RGB"), x, y, w, h, self.bbox_margin)
        else:
            image = image.convert("RGB")
        return self.transform(image)

    @staticmethod
    def open_image(data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception as e:                                  # noqa: BLE001
            raise ValueError(f"не удалось прочитать изображение: {e}") from e

    @torch.no_grad()
    def embed_tensors(self, tensors: List[torch.Tensor], batch_size: int = None) -> np.ndarray:
        if not tensors:
            return np.zeros((0, 1), dtype=np.float32)
        bs = batch_size or int(self.cfg["infer"].get("batch_size", 16))
        use_amp = self.device.type == "cuda"
        out = []
        with self._lock:
            for i in range(0, len(tensors), bs):
                batch = torch.stack(tensors[i:i + bs]).to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    feat = forward_with_tta(self.model, batch, self.flip_tta)
                feat = torch.nn.functional.normalize(feat.float(), dim=1)
                out.append(feat.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def embed_images(self, images: Sequence[Tuple[Image.Image, Optional[Sequence[float]]]],
                     batch_size: int = None) -> np.ndarray:
        return self.embed_tensors([self.prepare(img, bbox) for img, bbox in images], batch_size)

    def embed_paths(self, paths: Sequence[str], bboxes: Sequence[Optional[Sequence[float]]],
                    batch_size: int = None, progress=None) -> Tuple[np.ndarray, List[int]]:
        bs = batch_size or int(self.cfg["infer"].get("batch_size", 16))
        feats, ok = [], []
        buf, buf_idx = [], []
        for i, (p, bb) in enumerate(zip(paths, bboxes)):
            try:
                with Image.open(p) as im:
                    buf.append(self.prepare(im, bb))
                buf_idx.append(i)
            except (FileNotFoundError, OSError):
                continue
            if len(buf) >= bs:
                feats.append(self.embed_tensors(buf, bs)); ok += buf_idx
                buf, buf_idx = [], []
                if progress:
                    progress(len(ok), len(paths))
        if buf:
            feats.append(self.embed_tensors(buf, bs)); ok += buf_idx
        if progress:
            progress(len(ok), len(paths))
        if not feats:
            return np.zeros((0, 1), dtype=np.float32), []
        return np.concatenate(feats, axis=0).astype(np.float32), ok

    def _cam_engine(self):
        if self._gradcam is None:
            with self._lock:
                if self._gradcam is None:
                    self._gradcam = GradCAM(self.model)
        return self._gradcam

    def _crop_and_tensor(self, image: Image.Image, bbox):
        crop = _crop_bbox(image.convert("RGB"), *bbox, self.bbox_margin) if bbox else image.convert("RGB")
        return crop, self.transform(crop).unsqueeze(0).to(self.device)

    def explain_pair(self, query: Image.Image, query_bbox, reference: Image.Image, reference_bbox):
        cam_engine = self._cam_engine()
        q_crop, q_x = self._crop_and_tensor(query, query_bbox)
        r_crop, r_x = self._crop_and_tensor(reference, reference_bbox)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                q_emb = torch.nn.functional.normalize(self.model(q_x).float(), dim=1)
                r_emb = torch.nn.functional.normalize(self.model(r_x).float(), dim=1)
        score = float((q_emb * r_emb).sum())
        with self._lock:
            cam_q, used_q = cam_engine.cam(q_x, r_emb, out_size=tuple(self.model.image_size))
            cam_r, used_r = cam_engine.cam(r_x, q_emb, out_size=tuple(self.model.image_size))
        picture = cam_engine.pair_figure(q_crop, cam_q, r_crop, cam_r)
        stats = {"members_used": min(used_q, used_r), "score": score,
                 "query": cam_engine.focus_stats(cam_q, q_crop.size[::-1]),
                 "reference": cam_engine.focus_stats(cam_r, r_crop.size[::-1])}
        return picture, stats

    def explain(self, image: Image.Image, bbox, reference: np.ndarray,
                mode: str = "side_by_side"):
        self._cam_engine()
        crop, x = self._crop_and_tensor(image, bbox)
        ref = torch.as_tensor(np.asarray(reference, dtype=np.float32)).reshape(1, -1)
        with self._lock:
            cam, used = self._gradcam.cam(x, ref, out_size=tuple(self.model.image_size))
        stats = self._gradcam.focus_stats(cam, crop.size[::-1])
        stats["members_used"] = used
        picture = (self._gradcam.side_by_side(crop, cam) if mode == "side_by_side"
                   else self._gradcam.overlay(crop, cam))
        return picture, stats

    def rerank(self, q_feat: np.ndarray, g_feat: np.ndarray) -> np.ndarray:
        inf = self.cfg["infer"]
        dist = re_ranking(q_feat, g_feat, inf["rerank_k1"], inf["rerank_k2"], inf["rerank_lambda"])
        return 1.0 - dist


_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine
