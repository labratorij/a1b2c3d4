import os
import random
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _crop_coords(img_size, x, y, w, h, margin: float):
    """Координаты кропа с полем вокруг рамки; None, если рамка деградировала."""
    img_w, img_h = img_size
    mx, my = w * margin, h * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(img_w, int(x + w + mx)), min(img_h, int(y + h + my))
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def _crop_bbox(image: Image.Image, x, y, w, h, margin: float) -> Image.Image:
    box = _crop_coords(image.size, x, y, w, h, margin)
    return image if box is None else image.crop(box)


class MaskStore:
    """Маски ТС из precompute_masks.py (RLE по полному кадру).

    Хранятся по полному кадру, потому что кроп делается с полем и случайным сдвигом:
    маску нужно резать теми же координатами, что и изображение.
    """

    def __init__(self, path: str):
        d = np.load(path)
        self.index = {str(k): i for i, k in enumerate(d["ids"])}
        self.shapes, self.offsets, self.runs = d["shapes"], d["offsets"], d["runs"]

    def __contains__(self, image_id) -> bool:
        return str(image_id) in self.index

    def get(self, image_id) -> Optional[np.ndarray]:
        r = self._runs(image_id)
        if r is None:
            return None
        runs, shape = r
        return self._expand(runs, 0, shape[0] * shape[1]).reshape(shape)

    def crop(self, image_id, x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
        """Маска в координатах кропа, без разворачивания всего кадра.

        RLE идёт по строкам, поэтому нужные строки достаются напрямую: на 1080p это
        вдвое дешевле полного декода, а загрузчик здесь - узкое место.
        """
        r = self._runs(image_id)
        if r is None:
            return None
        runs, (h, w) = r
        y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
        if y1 <= y0:
            return None
        band = self._expand(runs, y0 * w, y1 * w).reshape(y1 - y0, w)
        return band[:, max(0, x0):min(x1, w)]

    def _runs(self, image_id):
        i = self.index.get(str(image_id))
        if i is None:
            return None
        return self.runs[self.offsets[i]:self.offsets[i + 1]], tuple(int(v) for v in self.shapes[i])

    @staticmethod
    def _expand(runs: np.ndarray, start: int, stop: int) -> np.ndarray:
        """Развернуть RLE только на полуинтервале [start, stop) плоского индекса."""
        ends = np.cumsum(runs, dtype=np.int64)
        first = int(np.searchsorted(ends, start, side="right"))
        last = int(np.searchsorted(ends, stop, side="left"))
        part = runs[first:last + 1].copy()
        if part.size == 0:
            return np.zeros(stop - start, dtype=bool)
        # обрезаем крайние серии по границам интервала
        part[0] -= start - (ends[first - 1] if first else 0)
        part[-1] -= max(0, int(ends[min(last, len(runs) - 1)]) - stop)
        # серии чередуются начиная с нулей, поэтому значение серии j равно j % 2
        values = np.resize([first % 2, 1 - first % 2], part.size).astype(np.uint8)
        return np.repeat(values, np.clip(part, 0, None)).astype(bool)


def apply_mask(crop: Image.Image, mask_crop: Optional[np.ndarray], mode: str,
               fill=(124, 116, 104)) -> Image.Image:
    """Подавление фона в кропе.

    hard   - фон заливается ровным цветом: максимальный эффект, но резкая граница
             сама по себе становится признаком
    soft   - фон затемняется, контекст (тени, положение на полосе) частично остаётся
    random - вариант для обучения: фон случайно заливается, зашумляется или остаётся
             как есть. Модель учится не опираться на фон, а на инференсе сегментация
             не нужна вовсе - задержка не растёт
    """
    if mask_crop is None or mode == "none":
        return crop
    # ветку "как есть" решаем до декодирования маски: она достаётся бесплатно
    if mode == "random":
        choice = random.random()
        if choice < 0.35:
            return crop
    if mask_crop.shape != (crop.height, crop.width):
        mask_crop = np.asarray(Image.fromarray(mask_crop.astype(np.uint8) * 255)
                               .resize(crop.size, Image.NEAREST)) > 127
    bg_idx = ~mask_crop
    if not bg_idx.any() or mask_crop.sum() == 0:
        return crop

    # правка только по пикселям фона и в uint8: по всему кропу во float32 это
    # втрое дороже, а загрузчик и без масок близок к пределу
    arr = np.array(crop, dtype=np.uint8)
    n = int(bg_idx.sum())
    if mode == "random":
        arr[bg_idx] = (np.random.randint(0, 256, 3, dtype=np.uint8) if choice < 0.7
                       else np.random.normal(128, 40, (n, 3)).clip(0, 255).astype(np.uint8))
    elif mode == "soft":
        arr[bg_idx] = (arr[bg_idx] * 0.35 + np.array(fill, dtype=np.float32) * 0.65).astype(np.uint8)
    else:
        arr[bg_idx] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(arr)


def _jitter_bbox(x, y, w, h, jitter: float):
    """Случайный сдвиг центра и масштаб рамки в пределах +-jitter (доля от w/h).

    Имитирует разброс детектора: на тесте рамки приходят не от той же разметки,
    что в train.csv, и модель, привыкшая к идеально центрированному кропу с полем
    ровно bbox_margin, теряет на сдвинутых/поджатых рамках.
    """
    dx = random.uniform(-jitter, jitter) * w
    dy = random.uniform(-jitter, jitter) * h
    sw = 1.0 + random.uniform(-jitter, jitter)
    sh = 1.0 + random.uniform(-jitter, jitter)
    cx, cy = x + w / 2 + dx, y + h / 2 + dy
    nw, nh = max(8.0, w * sw), max(8.0, h * sh)
    return cx - nw / 2, cy - nh / 2, nw, nh


def _crop_with_mask(image, image_id, x, y, w, h, margin, masks, mask_mode):
    box = _crop_coords(image.size, x, y, w, h, margin)
    crop = image if box is None else image.crop(box)
    if masks is None or mask_mode == "none" or box is None:
        return crop
    x0, y0, x1, y1 = box
    return apply_mask(crop, masks.crop(image_id, x0, y0, x1, y1), mask_mode)


class VehicleReIDTrainDataset(Dataset):
    """Train-датасет: кроп по bbox + vehicle_id -> contiguous label."""

    def __init__(self, df: pd.DataFrame, images_dir: str, label_map: dict,
                 bbox_margin: float = 0.1, transform=None, bbox_jitter: float = 0.0,
                 masks: "MaskStore" = None, mask_mode: str = "none"):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.label_map = label_map
        self.bbox_margin = bbox_margin
        self.transform = transform
        self.bbox_jitter = bbox_jitter   # 0 = рамка строго из train.csv (поведение по умолчанию)
        self.masks = masks
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_dir, f"{row.image_id}.jpg")
        image = Image.open(path).convert("RGB")
        x, y, w, h = row.x, row.y, row.w, row.h
        if self.bbox_jitter:
            x, y, w, h = _jitter_bbox(x, y, w, h, self.bbox_jitter)
        image = _crop_with_mask(image, row.image_id, x, y, w, h, self.bbox_margin,
                                self.masks, self.mask_mode)
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_map[row.vehicle_id]
        camera_id = int(row.camera_id) if "camera_id" in row else -1
        return image, label, camera_id, row.image_id


class VehicleReIDTestDataset(Dataset):
    """Test-датасет (query/gallery): кроп по bbox, без меток."""

    def __init__(self, df: pd.DataFrame, images_dir: str, bbox_margin: float = 0.1, transform=None,
                 masks: "MaskStore" = None, mask_mode: str = "none"):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.bbox_margin = bbox_margin
        self.transform = transform
        self.masks = masks
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_dir, f"{row.image_id}.jpg")
        image = Image.open(path).convert("RGB")
        image = _crop_with_mask(image, row.image_id, row.x, row.y, row.w, row.h,
                                self.bbox_margin, self.masks, self.mask_mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, row.image_id


def build_label_map(df: pd.DataFrame) -> dict:
    unique_ids = sorted(df.vehicle_id.unique().tolist())
    return {vid: i for i, vid in enumerate(unique_ids)}


def build_camera_map(df: pd.DataFrame) -> dict:
    """camera_id -> contiguous index, для SIE (side information embedding) в TransReID."""
    unique_cams = sorted(df.camera_id.unique().tolist())
    return {cid: i for i, cid in enumerate(unique_cams)}


def split_train_val_ids(df: pd.DataFrame, val_fraction: float, seed: int):
    """Отделяет часть vehicle_id целиком под внутреннюю валидацию (open-set,
    как в тестовом протоколе: id в train/val не пересекаются)."""
    rng = np.random.RandomState(seed)
    unique_ids = df.vehicle_id.unique()
    rng.shuffle(unique_ids)
    n_val = max(1, int(len(unique_ids) * val_fraction))
    val_ids = set(unique_ids[:n_val].tolist())
    train_ids = set(unique_ids[n_val:].tolist())

    train_df = df[df.vehicle_id.isin(train_ids)].reset_index(drop=True)
    val_df = df[df.vehicle_id.isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def make_val_query_gallery(val_df: pd.DataFrame, seed: int):
    """Из отложенных id формирует внутренние query/gallery для расчёта CMC/mAP:
    по одному случайному изображению каждого id - в query, остальные - в gallery."""
    rng = np.random.RandomState(seed)
    query_rows, gallery_rows = [], []
    for vid, group in val_df.groupby("vehicle_id"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        query_rows.append(idxs[0])
        gallery_rows.extend(idxs[1:])
    query_df = val_df.loc[query_rows].reset_index(drop=True)
    gallery_df = val_df.loc[gallery_rows].reset_index(drop=True)
    return query_df, gallery_df
