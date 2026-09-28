import numpy as np
import torch


@torch.no_grad()
def forward_with_tta(model, images: torch.Tensor, flip_tta: bool) -> torch.Tensor:
    feat = model(images)
    if flip_tta:
        feat = feat + model(torch.flip(images, dims=[3]))
        feat = feat / 2
    return feat


@torch.no_grad()
def extract_features(model, loader, device, flip_tta: bool = False, normalize: bool = False):
    model.eval()
    use_amp = device.type == "cuda"
    feats, ids = [], []
    for images, image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            feat = forward_with_tta(model, images, flip_tta)
        feat = feat.float()
        if normalize:
            feat = torch.nn.functional.normalize(feat, dim=1)
        feats.append(feat.cpu().numpy())
        ids.extend(image_ids)
    return np.concatenate(feats, axis=0).astype(np.float32), list(ids)
