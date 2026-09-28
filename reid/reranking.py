import numpy as np


def _normalize(feat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.clip(norm, 1e-12, None)


def re_ranking(query_feat: np.ndarray, gallery_feat: np.ndarray,
               k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    query_feat = _normalize(query_feat.astype(np.float32))
    gallery_feat = _normalize(gallery_feat.astype(np.float32))

    all_feat = np.concatenate([query_feat, gallery_feat], axis=0)
    n_all = all_feat.shape[0]
    n_q = query_feat.shape[0]

    dist = 2 - 2 * (all_feat @ all_feat.T)
    dist = np.clip(dist, 0, None)

    original_dist = np.power(dist, 2).astype(np.float32)
    original_dist = original_dist / (original_dist.max(axis=0) + 1e-12)
    V = np.zeros_like(original_dist)
    initial_rank = np.argsort(original_dist, axis=1)

    for i in range(n_all):
        k_reciprocal_index = _k_reciprocal_neighbors(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal = _k_reciprocal_neighbors(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(n_all):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(n_all)]
    jaccard_dist = np.zeros((n_q, n_all), dtype=np.float32)
    for i in range(n_q):
        temp_min = np.zeros(n_all, dtype=np.float32)
        indNonZero = np.where(V[i] != 0)[0]
        for ind in indNonZero:
            temp_min[invIndex[ind]] += np.minimum(V[i, ind], V[invIndex[ind], ind])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:n_q] * lambda_value
    return final_dist[:, n_q:n_all]


def _k_reciprocal_neighbors(initial_rank: np.ndarray, i: int, k: int) -> np.ndarray:
    forward = initial_rank[i, :k + 1]
    backward = initial_rank[forward, :k + 1]
    fi = np.where(backward == i)[0]
    return forward[fi]
