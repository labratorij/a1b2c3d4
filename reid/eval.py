import numpy as np


def compute_cmc_map(query_feat: np.ndarray, gallery_feat: np.ndarray,
                     query_ids: np.ndarray, gallery_ids: np.ndarray,
                     ranks=(1, 5, 10)):
    """CMC@rank и mAP для внутренней валидации (открытый набор идентичностей)."""
    qf = query_feat / np.clip(np.linalg.norm(query_feat, axis=1, keepdims=True), 1e-12, None)
    gf = gallery_feat / np.clip(np.linalg.norm(gallery_feat, axis=1, keepdims=True), 1e-12, None)
    sim = qf @ gf.T
    order = np.argsort(-sim, axis=1)

    n_q = qf.shape[0]
    cmc = np.zeros(max(ranks))
    aps = []

    for i in range(n_q):
        matches = (gallery_ids[order[i]] == query_ids[i]).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_match = np.argmax(matches)
        if first_match < len(cmc):
            cmc[first_match:] += 1

        rel_positions = np.where(matches == 1)[0]
        precisions = (np.arange(len(rel_positions)) + 1) / (rel_positions + 1)
        aps.append(precisions.mean())

    cmc = cmc / n_q
    result = {f"cmc@{r}": float(cmc[r - 1]) for r in ranks}
    result["mAP"] = float(np.mean(aps)) if aps else 0.0
    return result
