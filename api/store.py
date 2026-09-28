import io
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _fit_pca(x: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu.astype(np.float32), np.ascontiguousarray(vt[:dim], dtype=np.float32)


class BaseStore:

    kind = "base"

    def fit_projection(self, feats: np.ndarray, dim: int) -> None:
        raise NotImplementedError

    def projection(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def set_meta(self, key: str, value: str) -> None:
        raise NotImplementedError

    def get_meta(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def project(self, feats: np.ndarray) -> np.ndarray:
        feats = _l2(feats)
        pca = self.projection()
        if pca is None:
            return feats
        mu, comp = pca
        if feats.shape[1] != comp.shape[1]:
            raise ValueError(f"признак {feats.shape[1]}-d не соответствует проекции "
                             f"{comp.shape[1]}-d: галерея построена другой моделью")
        return _l2((feats - mu) @ comp.T)

    def upsert(self, items: Sequence[dict], feats: np.ndarray) -> int:
        raise NotImplementedError

    def search(self, feat: np.ndarray, top_k: int) -> List[dict]:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def delete(self, image_ids: Sequence[str]) -> int:
        raise NotImplementedError


class SqliteStore(BaseStore):
    kind = "sqlite"

    def __init__(self, path: str):
        import sqlite3
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                image_id   TEXT PRIMARY KEY,
                vehicle_id TEXT,
                source     TEXT,
                dim        INTEGER NOT NULL,
                embedding  BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
            """
        )
        self._con.commit()
        self._cache = None

    def fit_projection(self, feats: np.ndarray, dim: int) -> None:
        mu, comp = _fit_pca(_l2(feats), dim)
        buf = io.BytesIO()
        np.savez(buf, mu=mu, comp=comp)
        self._con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('pca', ?)", (buf.getvalue(),))
        self._con.commit()
        self._cache = None

    def set_meta(self, key, value):
        self._con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                          (key, str(value).encode("utf-8")))
        self._con.commit()

    def get_meta(self, key):
        row = self._con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else bytes(row[0]).decode("utf-8")

    def projection(self):
        row = self._con.execute("SELECT value FROM meta WHERE key='pca'").fetchone()
        if row is None:
            return None
        d = np.load(io.BytesIO(row[0]))
        return d["mu"], d["comp"]

    def upsert(self, items, feats):
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        rows = [(it["image_id"], it.get("vehicle_id"), it.get("source"), feats.shape[1], feats[i].tobytes())
                for i, it in enumerate(items)]
        self._con.executemany(
            "INSERT OR REPLACE INTO gallery(image_id, vehicle_id, source, dim, embedding) VALUES (?,?,?,?,?)", rows)
        self._con.commit()
        self._cache = None
        return len(rows)

    def _matrix(self):
        if self._cache is None:
            rows = self._con.execute("SELECT image_id, vehicle_id, dim, embedding FROM gallery").fetchall()
            if not rows:
                self._cache = (np.array([]), np.array([]), np.zeros((0, 1), dtype=np.float32))
            else:
                dim = rows[0][2]
                mat = np.stack([np.frombuffer(r[3], dtype=np.float32, count=dim) for r in rows])
                self._cache = (np.array([r[0] for r in rows]),
                               np.array([r[1] for r in rows], dtype=object),
                               np.ascontiguousarray(mat, dtype=np.float32))
        return self._cache

    def search(self, feat, top_k):
        ids, vids, mat = self._matrix()
        if len(ids) == 0:
            return []
        q = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != mat.shape[1]:
            raise ValueError(f"размерность запроса {q.shape[1]} != размерности базы {mat.shape[1]}")
        sim = (q @ mat.T)[0]
        k = min(top_k, len(sim))
        part = np.argpartition(-sim, k - 1)[:k]
        order = part[np.argsort(-sim[part])]
        return [{"image_id": str(ids[i]), "vehicle_id": None if vids[i] is None else str(vids[i]),
                 "score": float(sim[i])} for i in order]

    def stats(self):
        n = self._con.execute("SELECT COUNT(*) FROM gallery").fetchone()[0]
        dim = self._con.execute("SELECT dim FROM gallery LIMIT 1").fetchone()
        nv = self._con.execute("SELECT COUNT(DISTINCT vehicle_id) FROM gallery WHERE vehicle_id IS NOT NULL").fetchone()[0]
        return {"backend": self.kind, "location": self.path, "items": n,
                "vehicles": nv, "dim": dim[0] if dim else None,
                "projection": self.projection() is not None}

    def clear(self):
        n = self._con.execute("SELECT COUNT(*) FROM gallery").fetchone()[0]
        self._con.executescript("DELETE FROM gallery; DELETE FROM meta;")
        self._con.commit()
        self._cache = None
        return n

    def delete(self, image_ids):
        cur = self._con.executemany("DELETE FROM gallery WHERE image_id=?", [(i,) for i in image_ids])
        self._con.commit()
        self._cache = None
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(image_ids)


class PostgresStore(BaseStore):
    kind = "postgres+pgvector"

    def __init__(self, dsn: str, dim: int):
        import psycopg
        self.dsn = dsn
        self.dim = dim
        self._con = psycopg.connect(dsn, autocommit=True)
        with self._con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS gallery (
                    image_id   TEXT PRIMARY KEY,
                    vehicle_id TEXT,
                    source     TEXT,
                    embedding  vector({dim}) NOT NULL
                )""")
            cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BYTEA)")
            cur.execute("CREATE INDEX IF NOT EXISTS gallery_emb_hnsw ON gallery "
                        "USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def _vec(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.7g}" for x in np.asarray(v, dtype=np.float32).ravel()) + "]"

    def fit_projection(self, feats, dim):
        mu, comp = _fit_pca(_l2(feats), dim)
        buf = io.BytesIO(); np.savez(buf, mu=mu, comp=comp)
        with self._con.cursor() as cur:
            cur.execute("INSERT INTO meta(key, value) VALUES ('pca', %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (buf.getvalue(),))

    def set_meta(self, key, value):
        with self._con.cursor() as cur:
            cur.execute("INSERT INTO meta(key, value) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                        (key, str(value).encode("utf-8")))

    def get_meta(self, key):
        with self._con.cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key=%s", (key,))
            row = cur.fetchone()
        return None if row is None else bytes(row[0]).decode("utf-8")

    def projection(self):
        with self._con.cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key='pca'")
            row = cur.fetchone()
        if row is None:
            return None
        d = np.load(io.BytesIO(bytes(row[0])))
        return d["mu"], d["comp"]

    def upsert(self, items, feats):
        feats = np.asarray(feats, dtype=np.float32)
        with self._con.cursor() as cur:
            cur.executemany(
                "INSERT INTO gallery(image_id, vehicle_id, source, embedding) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (image_id) DO UPDATE SET vehicle_id=EXCLUDED.vehicle_id, "
                "source=EXCLUDED.source, embedding=EXCLUDED.embedding",
                [(it["image_id"], it.get("vehicle_id"), it.get("source"), self._vec(feats[i]))
                 for i, it in enumerate(items)])
        return len(items)

    def search(self, feat, top_k):
        with self._con.cursor() as cur:
            cur.execute("SELECT image_id, vehicle_id, 1 - (embedding <=> %s) AS score FROM gallery "
                        "ORDER BY embedding <=> %s LIMIT %s",
                        (self._vec(feat), self._vec(feat), top_k))
            return [{"image_id": r[0], "vehicle_id": r[1], "score": float(r[2])} for r in cur.fetchall()]

    def stats(self):
        with self._con.cursor() as cur:
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT vehicle_id) FROM gallery")
            n, nv = cur.fetchone()
        return {"backend": self.kind, "location": self.dsn.rsplit("@", 1)[-1], "items": n,
                "vehicles": nv, "dim": self.dim, "projection": self.projection() is not None}

    def clear(self):
        with self._con.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM gallery")
            n = cur.fetchone()[0]
            cur.execute("TRUNCATE gallery")
            cur.execute("DELETE FROM meta")
        return n

    def delete(self, image_ids):
        with self._con.cursor() as cur:
            cur.execute("DELETE FROM gallery WHERE image_id = ANY(%s)", (list(image_ids),))
            return cur.rowcount


def build_store(database_url: str, sqlite_path: str, dim: int) -> BaseStore:
    if database_url:
        return PostgresStore(database_url, dim)
    return SqliteStore(sqlite_path)
