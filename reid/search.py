"""Индекс поиска по базе признаков ТС - для эксплуатации, не для сабмишена.

Измерения, на которых построен этот модуль (база 10 000 объектов, см. EXPERIMENTS.md):

  * поиск - это матричное умножение, и он дёшев: 5.2 мс на полных 5632-d, 0.44 мс
    после PCA до 256-d. Векторная БД / ANN-индекс на таком размере только замедляют -
    они начинают выигрывать от ~100 тысяч векторов.
  * PCA 5632 -> 256 в онлайн-режиме стоит 0.3 pt mAP@10 (0.8597 -> 0.8570) при
    12-кратной экономии памяти и времени. Whitening применять НЕЛЬЗЯ: -6...-11 pt.
  * k-reciprocal re-ranking в онлайне неприменим: он O(N^2) по всей базе на каждый
    запрос (1.5 с при базе 5000) И его выигрыш существует только когда запросы идут
    пакетом - весь эффект берётся из связей между запросами, а не по галерее.
    На пакете из 1 запроса re-ranking не улучшает, а ухудшает результат.
  * эмбеддинги нельзя держать в float16 для счёта на CPU: в numpy нет BLAS для fp16,
    поиск замедляется в 25 раз. Хранить в fp16 можно, считать - в fp32.

Онлайн-режим (запрос за запросом):
    index = VehicleIndex.build(gallery_feat, gallery_ids, pca_dim=256)
    index.save("index.npz")
    ...
    ids, scores = index.search(query_feat, top_k=10)          # 0.44 мс при базе 10k

Пакетный режим (как в infer.py - даёт +3.4 pt, но нужен весь пакет запросов сразу):
    ids, scores = index.search(query_feats, top_k=10, rerank=RerankParams(10, 3, 0.5))
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .reranking import re_ranking


@dataclass
class RerankParams:
    """Параметры k-reciprocal re-ranking; подобраны на репетиции (см. EXPERIMENTS.md §6)."""
    k1: int = 10
    k2: int = 3
    lambda_value: float = 0.5


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_pca(x: np.ndarray, dim: int):
    """Проекция обучается на признаках базы. Возвращает (mean, components).

    Без whitening - оно проверено и ухудшает качество на 6-11 pt.
    """
    x = np.asarray(x, dtype=np.float32)
    if dim >= x.shape[1]:
        raise ValueError(f"pca_dim={dim} не меньше размерности признака {x.shape[1]}")
    if dim > min(x.shape):
        raise ValueError(f"pca_dim={dim} больше числа объектов базы ({x.shape[0]}) - "
                         f"проекцию не на чем обучить, возьмите больше объектов или меньший dim")
    mu = x.mean(axis=0, keepdims=True)
    # SVD вместо ковариационной матрицы: размерность признака велика (тысячи)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu.astype(np.float32), np.ascontiguousarray(vt[:dim], dtype=np.float32)


class VehicleIndex:
    """Плоский (brute-force) индекс: L2-нормированные признаки + косинусный поиск.

    Хранит матрицу в float32 C-порядка - так numpy отдаёт её в BLAS без копий.
    """

    def __init__(self, feats: np.ndarray, ids: Sequence, pca=None):
        self.ids = np.asarray(ids)
        self.pca = pca
        self.feats = np.ascontiguousarray(feats, dtype=np.float32)
        if len(self.ids) != len(self.feats):
            raise ValueError(f"{len(self.ids)} идентификаторов на {len(self.feats)} признаков")

    @classmethod
    def build(cls, feats: np.ndarray, ids: Sequence, pca_dim: Optional[int] = None) -> "VehicleIndex":
        feats = l2_normalize(feats)
        pca = None
        if pca_dim:
            pca = fit_pca(feats, pca_dim)
            feats = cls._project(feats, pca)
        return cls(feats, ids, pca)

    @staticmethod
    def _project(x: np.ndarray, pca) -> np.ndarray:
        mu, comp = pca
        return l2_normalize((x - mu) @ comp.T)

    def encode(self, feats: np.ndarray) -> np.ndarray:
        """Признаки запроса в то же пространство, что и база."""
        feats = l2_normalize(feats)
        return self._project(feats, self.pca) if self.pca is not None else feats

    def search(self, query_feats: np.ndarray, top_k: int = 10,
               rerank: Optional[RerankParams] = None):
        """-> (ids [n_query, top_k], scores [n_query, top_k]).

        rerank=None - онлайн-режим: косинус, argpartition вместо полной сортировки.
        rerank=RerankParams(...) - пакетный: ТОЛЬКО когда запросы идут все сразу
        (десятки-сотни). На одном-двух запросах re-ranking ухудшает результат.
        """
        q = self.encode(query_feats)
        k = min(top_k, len(self.feats))
        if rerank is None:
            sim = q @ self.feats.T
        else:
            if len(q) < 10:
                raise ValueError(f"re-ranking имеет смысл только на пакете запросов "
                                 f"(здесь {len(q)}): его эффект берётся из связей между "
                                 f"запросами. Для онлайна вызывайте без rerank.")
            sim = 1.0 - re_ranking(q, self.feats, rerank.k1, rerank.k2, rerank.lambda_value)
        # argpartition берёт top-k без полной сортировки базы, затем сортируем только k
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(len(sim))[:, None]
        order = part[rows, np.argsort(-sim[rows, part], axis=1)]
        return self.ids[order], np.take_along_axis(sim, order, axis=1)

    def search_one(self, feat: np.ndarray, top_k: int = 10, threshold: Optional[float] = None):
        """Один запрос -> [(id, score), ...]. threshold отсекает слабые совпадения
        (порог калибруется на размеченных парах, см. EXPERIMENTS.md §7.2)."""
        ids, scores = self.search(feat, top_k=top_k)
        pairs = list(zip(ids[0].tolist(), scores[0].tolist()))
        return [p for p in pairs if threshold is None or p[1] >= threshold]

    def add(self, feats: np.ndarray, ids: Sequence) -> None:
        """Добавить объекты. PCA-проекция не переобучается - она фиксируется при build."""
        self.feats = np.ascontiguousarray(np.vstack([self.feats, self.encode(feats)]), dtype=np.float32)
        self.ids = np.concatenate([self.ids, np.asarray(ids)])

    def remove(self, ids: Sequence) -> int:
        keep = ~np.isin(self.ids, np.asarray(ids))
        removed = int((~keep).sum())
        self.feats, self.ids = np.ascontiguousarray(self.feats[keep]), self.ids[keep]
        return removed

    def save(self, path: str) -> None:
        payload = {"feats": self.feats, "ids": self.ids}
        if self.pca is not None:
            payload["pca_mean"], payload["pca_comp"] = self.pca
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "VehicleIndex":
        d = np.load(path, allow_pickle=True)
        pca = (d["pca_mean"], d["pca_comp"]) if "pca_mean" in d.files else None
        return cls(d["feats"], d["ids"], pca)

    def __len__(self):
        return len(self.feats)

    def __repr__(self):
        dim = self.feats.shape[1] if len(self.feats) else 0
        pca = f", PCA {dim}-d" if self.pca is not None else ""
        return (f"VehicleIndex({len(self)} объектов, {dim}-d{pca}, "
                f"{self.feats.nbytes / 2**20:.1f} МБ)")
