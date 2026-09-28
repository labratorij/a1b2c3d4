from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_LUT_STOPS = np.array([
    [0.00, 0, 0, 4], [0.15, 40, 11, 84], [0.30, 101, 21, 110], [0.45, 159, 42, 99],
    [0.60, 212, 72, 66], [0.75, 245, 125, 21], [0.90, 250, 193, 39], [1.00, 252, 255, 164],
], dtype=np.float32)


def _colormap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    pos, cols = _LUT_STOPS[:, 0], _LUT_STOPS[:, 1:]
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, pos, cols[:, ch])
    return out.astype(np.uint8)


class GradCAM:

    def __init__(self, model):
        self.model = model
        self.members = self._members(model)
        if not self.members:
            raise ValueError("ни один член модели не отдаёт свёрточную карту признаков — "
                             "Grad-CAM для такой конфигурации не построить")

    @staticmethod
    def _members(model) -> List:
        raw = list(getattr(model, "members", [])) or [model]
        out = []
        for m in raw:
            inner = getattr(m, "model", m)
            backbone = getattr(inner, "backbone", None)
            if backbone is not None:
                out.append((m, backbone))
        return out

    def _cam_for_member(self, member, backbone, x: torch.Tensor,
                        reference: torch.Tensor) -> Optional[np.ndarray]:
        acts = {}

        def hook(_module, _inp, out):
            acts["value"] = out
            out.retain_grad()

        handle = backbone.register_forward_hook(hook)
        try:
            xi = self.model._adapt(x, member) if hasattr(self.model, "_adapt") else x
            xi = xi.clone().requires_grad_(True)
            with torch.enable_grad():
                feat = member(xi) if callable(member) else member.model(xi)
                feat = F.normalize(feat.float(), dim=1)
                ref = F.normalize(reference.float().to(feat.device), dim=1)
                score = (feat * ref[:, :feat.shape[1]]).sum()
                score.backward()
            a = acts.get("value")
            if a is None or a.grad is None:
                return None
            weights = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * a).sum(dim=1, keepdim=True))
            return cam.detach().float().cpu().numpy()[0, 0]
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)

    def cam(self, x: torch.Tensor, reference: torch.Tensor,
            out_size: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        maps, used = [], 0
        for member, backbone in self.members:
            m = self._cam_for_member(member, backbone, x, reference)
            if m is None or not np.isfinite(m).all() or m.max() <= 0:
                continue
            t = torch.from_numpy(m)[None, None]
            t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=False)[0, 0]
            arr = t.numpy()
            arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
            maps.append(arr)
            used += 1
        if not maps:
            raise ValueError("Grad-CAM не дал ненулевой карты: возможно, эталонный эмбеддинг "
                             "ортогонален признаку запроса")
        cam = np.mean(maps, axis=0)
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cam, used

    @staticmethod
    def _resize_cam(cam: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        if cam.shape == hw:
            return cam
        t = torch.from_numpy(cam)[None, None]
        return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()

    @classmethod
    def overlay(cls, image: Image.Image, cam: np.ndarray, alpha: float = 0.65,
                desaturate: bool = True) -> Image.Image:
        rgb = image.convert("RGB")
        base = np.asarray(rgb.convert("L").convert("RGB") if desaturate else rgb, dtype=np.float32)
        cam = cls._resize_cam(cam, base.shape[:2])
        heat = _colormap(cam).astype(np.float32)
        a = (alpha * cam)[..., None]
        return Image.fromarray(np.clip(base * (1 - a) + heat * a, 0, 255).astype(np.uint8))

    @classmethod
    def side_by_side(cls, image: Image.Image, cam: np.ndarray, gap: int = 8) -> Image.Image:
        left = image.convert("RGB")
        right = cls.overlay(left, cam)
        w, h = left.size
        out = Image.new("RGB", (w * 2 + gap, h), (255, 255, 255))
        out.paste(left, (0, 0))
        out.paste(right, (w + gap, 0))
        return out

    @classmethod
    def pair_figure(cls, query: Image.Image, cam_q: np.ndarray,
                    ref: Image.Image, cam_r: np.ndarray, gap: int = 10,
                    row_height: int = 360) -> Image.Image:
        def row(img: Image.Image, cam: np.ndarray):
            over = cls.overlay(img, cam)
            k = row_height / img.height
            size = (max(1, int(img.width * k)), row_height)
            return img.convert("RGB").resize(size, Image.LANCZOS), over.resize(size, Image.LANCZOS)

        q_img, q_over = row(query, cam_q)
        r_img, r_over = row(ref, cam_r)
        w = max(q_img.width, r_img.width) * 2 + gap
        out = Image.new("RGB", (w, row_height * 2 + gap), (255, 255, 255))
        out.paste(q_img, (0, 0)); out.paste(q_over, (q_img.width + gap, 0))
        out.paste(r_img, (0, row_height + gap)); out.paste(r_over, (r_img.width + gap, row_height + gap))
        return out

    @classmethod
    def focus_stats(cls, cam: np.ndarray, hw: Tuple[int, int] = None) -> dict:
        c = cls._resize_cam(cam, hw) if hw else cam
        h, w = c.shape
        y0, y1 = int(h * 0.15), int(h * 0.85)
        x0, x1 = int(w * 0.15), int(w * 0.85)
        total = float(c.sum()) or 1.0
        center = float(c[y0:y1, x0:x1].sum())
        return {"center_share": center / total,
                "border_share": 1.0 - center / total,
                "peak_area": float((c > 0.5).mean())}
