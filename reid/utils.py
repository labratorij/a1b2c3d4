import os
import random
import logging

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "reid") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


WEIGHTS_URL = "https://disk.yandex.ru/client/disk/LCT"


def load_checkpoint(path: str, map_location=None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"не найден файл весов: {path}\n"
            f"Весов нет в репозитории - скачайте их и положите в каталог weights/:\n"
            f"  {WEIGHTS_URL}\n"
            f"Нужны оба файла: bestv10_512.pth и r101_ibn_256.pth (по 92 МБ).")
    return torch.load(path, map_location=map_location, weights_only=False)
