import random
from io import BytesIO

import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# значения по умолчанию = поведение до появления секции data.aug, чтобы старые
# конфиги (v2...v7) воспроизводились без изменений
DEFAULT_AUG = {
    "color_jitter": [0.15, 0.15, 0.1, 0.02],   # brightness, contrast, saturation, hue
    "erasing_p": 0.5,
    "rotation": 0.0,                            # градусы, 0 = выключено
    "blur_p": 0.0,                              # вероятность гауссова размытия
    "jpeg_p": 0.0,                              # вероятность пережатия в JPEG
    "jpeg_quality": [40, 90],
    "grayscale_p": 0.0,                         # осторожно: цвет для ТС - сильный признак
}


class RandomJPEG:
    """Пережатие в JPEG со случайным качеством: в тесте кадры с камер разного качества,
    а train.csv и test_*.csv - одни и те же исходники, так что модель этого не видит."""

    def __init__(self, p: float, quality=(40, 90)):
        self.p = p
        self.quality = tuple(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self):
        return f"RandomJPEG(p={self.p}, quality={self.quality})"


def build_train_transforms(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD, aug=None):
    """aug - секция data.aug конфига; None/{} даёт исходный набор аугментаций."""
    a = dict(DEFAULT_AUG)
    a.update(aug or {})
    h, w = image_size
    fill = tuple(int(255 * m) for m in mean)      # заполнение углов при повороте - серый уровня среднего
    ops = [
        T.Resize((h, w)),
        T.Pad(10),
        T.RandomCrop((h, w)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if a["rotation"]:
        ops.append(T.RandomRotation(a["rotation"], fill=fill))
    if a["grayscale_p"]:
        ops.append(T.RandomGrayscale(p=a["grayscale_p"]))
    ops.append(T.ColorJitter(*a["color_jitter"]))
    if a["jpeg_p"]:
        ops.append(RandomJPEG(a["jpeg_p"], a["jpeg_quality"]))
    if a["blur_p"]:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))], p=a["blur_p"]))
    ops += [
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=a["erasing_p"], scale=(0.02, 0.2), value="random"),
    ]
    return T.Compose(ops)


def build_test_transforms(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    h, w = image_size
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
