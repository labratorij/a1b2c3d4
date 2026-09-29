"""Замер скорости работы сервиса на текущей машине.

Повторяет сценарии из раздела «Время работы» README: загрузка модели, признак одного ТС,
пропускная способность пакетом, поиск по базе, переранжирование и оценка полного пакетного
прогона. Модель загружается в этом же процессе, backend запускать не нужно.

    python benchmark.py
    python benchmark.py --data-root D:/data/Датасет --full-batch
    python benchmark.py --device cpu --json data/outputs/benchmark.json
    python benchmark.py --api http://127.0.0.1:8000       # плюс полный HTTP-запрос /embed

Без данных замеры идут на синтетических кадрах 1920x1080 — скорость сети от содержимого
изображения не зависит, но чтение JPEG с диска в этом случае не учитывается.
"""
import argparse
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import urllib.request
import uuid

import numpy as np

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)


def summarize(times_s):
    ms = sorted(t * 1000 for t in times_s)
    p90 = ms[min(len(ms) - 1, int(round(0.9 * (len(ms) - 1))))]
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 3),
            "mean_ms": round(statistics.fmean(ms), 3), "p90_ms": round(p90, 3),
            "min_ms": round(ms[0], 3)}


def repeat(fn, n, warmup):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def system_info(torch):
    info = {"python": platform.python_version(), "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_threads": os.cpu_count(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_memory_gb"] = round(p.total_memory / 2 ** 30, 1)
    return info


def load_samples(data_root, n):
    """Реальные кадры и bbox из каталога с данными; None, если данных нет."""
    from api import discovery
    if not data_root or not os.path.isdir(data_root):
        return None
    report = discovery.inspect(os.path.normpath(data_root))
    csv_path = report["suggested"]["query_csv"] or report["suggested"]["bulk_csv"]
    if not csv_path or not report["images_dir"]:
        return None
    df = read_csv(csv_path).head(n)
    paths = [os.path.join(report["images_dir"], f"{i}.jpg") for i in df.image_id]
    bboxes = df[["x", "y", "w", "h"]].to_numpy().tolist()
    pairs = [(p, b) for p, b in zip(paths, bboxes) if os.path.exists(p)]
    return {"report": report, "pairs": pairs} if pairs else None


def read_csv(path):
    import pandas as pd
    cols = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip().split(",")
    return pd.read_csv(path) if first[0] == "image_id" else pd.read_csv(path, header=None, names=cols[:len(first)])


def synthetic_frame(rng):
    from PIL import Image
    arr = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    return Image.fromarray(arr), [600.0, 300.0, 640.0, 480.0]


def bench_search(sizes, dim, n_queries, top_k, rng):
    from api.store import SqliteStore
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            st = SqliteStore(os.path.join(tmp, f"g{size}.sqlite"))
            vecs = rng.standard_normal((size, dim)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            st.upsert([{"image_id": str(i)} for i in range(size)], vecs)
            queries = iter(rng.standard_normal((n_queries + 5, dim)).astype(np.float32))
            res[str(size)] = summarize(repeat(lambda: st.search(next(queries), top_k), n_queries, 5))
            st._con.close()
    return res


def multipart(fields, file_bytes):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
               f"Content-Type: image/jpeg\r\n\r\n".encode())
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def bench_http(api_url, jpeg, bbox, n, warmup):
    body, ctype = multipart({"bbox": ",".join(str(v) for v in bbox)}, jpeg)

    def call():
        req = urllib.request.Request(f"{api_url.rstrip('/')}/embed", data=body,
                                     headers={"Content-Type": ctype}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
    return summarize(repeat(call, n, warmup))


def print_report(r):
    s = r["system"]
    print("\n=== Машина ===")
    print(f"  CPU: {s['cpu']} ({s['cpu_threads']} потоков), torch {s['torch']}, CUDA {s['cuda']}")
    if s.get("gpu"):
        print(f"  GPU: {s['gpu']} ({s['gpu_memory_gb']} ГБ)")
    m = r["model"]
    print(f"  устройство: {m['device']}, вход {m['input_size']}, flip TTA {m['flip_tta']}, "
          f"признак {m['embedding_dim']}-d, батч {r['params']['batch_size']}")
    print(f"  данные: {r['params']['data']}")

    rows = [("загрузка модели", f"{r['load_seconds']:.1f} с")]
    e = r["embed_single"]
    rows.append(("признак одного ТС (медиана / p90)", f"{e['median_ms']:.1f} / {e['p90_ms']:.1f} мс"))
    if "http_embed" in r:
        h = r["http_embed"]
        rows.append(("HTTP POST /embed (медиана / p90)", f"{h['median_ms']:.1f} / {h['p90_ms']:.1f} мс"))
    t = r["throughput"]
    rows.append((f"пакетом по {t['batch_size']}", f"{t['images_per_s']:.1f} снимков/с "
                                                   f"({t['ms_per_image']:.1f} мс на снимок)"))
    for size, v in r["search"].items():
        rows.append((f"поиск по базе из {size} (PCA-{r['params']['search_dim']})", f"{v['median_ms']:.3f} мс"))
    rr = r["rerank"]
    rows.append((f"переранжирование {rr['n_query']}x{rr['n_gallery']}", f"{rr['seconds']:.2f} с"))
    b = r["batch_estimate"]
    rows.append((f"пакетная обработка {b['n_query']}x{b['n_gallery']} (оценка)", f"{b['seconds']:.0f} с"))
    if "batch_full" in r:
        f = r["batch_full"]
        rows.append((f"пакетная обработка {f['n_query']}x{f['n_gallery']} (реальный прогон)",
                     f"{f['seconds']:.0f} с"))

    print("\n=== Время работы ===")
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(w)}  {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="конфиг модели (по умолчанию MODEL_CONFIG)")
    ap.add_argument("--device", default=None, help="cuda / cpu (по умолчанию автоматически)")
    ap.add_argument("--data-root", default=None, help="каталог с данными (иначе DATA_ROOT)")
    ap.add_argument("--repeats", type=int, default=50, help="повторов для замеров задержки")
    ap.add_argument("--warmup", type=int, default=5, help="прогревочных прогонов")
    ap.add_argument("--throughput-images", type=int, default=120, help="снимков для замера пропускной способности")
    ap.add_argument("--batch-size", type=int, default=None, help="размер батча (по умолчанию из конфига)")
    ap.add_argument("--gallery-sizes", default="750,10000", help="размеры базы для замера поиска")
    ap.add_argument("--n-query", type=int, default=1110, help="запросов в оценке пакетной обработки")
    ap.add_argument("--n-gallery", type=int, default=750, help="объектов галереи в оценке пакетной обработки")
    ap.add_argument("--full-batch", action="store_true",
                    help="прогнать реальный пакет query/gallery из каталога данных (долго)")
    ap.add_argument("--api", default=None, help="адрес запущенного backend для замера HTTP /embed")
    ap.add_argument("--json", default=None, help="сохранить результаты в JSON")
    args = ap.parse_args()

    os.environ.setdefault("SERVICE_DIR", SERVICE_DIR)
    if args.data_root:
        os.environ["DATA_ROOT"] = os.path.abspath(args.data_root)

    import torch
    from api import settings
    from api.engine import Engine

    rng = np.random.default_rng(0)
    print("загрузка модели…", flush=True)
    t0 = time.perf_counter()
    eng = Engine(config_path=args.config, device=args.device)
    if eng.device.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    bs = args.batch_size or int(eng.cfg["infer"].get("batch_size", 16))

    samples = load_samples(settings.DATA_ROOT, max(args.throughput_images, args.repeats))
    if samples:
        from PIL import Image
        def get_frame(i):
            p, b = samples["pairs"][i % len(samples["pairs"])]
            with Image.open(p) as im:
                return im.copy(), b
        data_desc = f"{samples['report']['root']} ({len(samples['pairs'])} кадров)"
    else:
        frames = [synthetic_frame(rng) for _ in range(4)]
        get_frame = lambda i: frames[i % len(frames)]
        data_desc = "синтетические кадры 1920x1080 (данные не найдены)"

    result = {"system": system_info(torch), "model": eng.describe(),
              "params": {"batch_size": bs, "repeats": args.repeats, "warmup": args.warmup,
                         "data": data_desc, "search_dim": settings.PCA_DIM or None},
              "load_seconds": round(load_s, 2)}

    print("признак одного ТС…", flush=True)
    single = [get_frame(i) for i in range(8)]
    counter = iter(range(10 ** 9))
    def one():
        img, bb = single[next(counter) % len(single)]
        eng.embed_images([(img, bb)])
    result["embed_single"] = summarize(repeat(one, args.repeats, args.warmup))

    print("пропускная способность…", flush=True)
    n = args.throughput_images
    if samples:
        paths = [samples["pairs"][i % len(samples["pairs"])][0] for i in range(n)]
        boxes = [samples["pairs"][i % len(samples["pairs"])][1] for i in range(n)]
        eng.embed_paths(paths[:bs], boxes[:bs], batch_size=bs)
        t0 = time.perf_counter()
        feats, _ = eng.embed_paths(paths, boxes, batch_size=bs)
    else:
        tensors = [eng.prepare(*get_frame(i)) for i in range(n)]
        eng.embed_tensors(tensors[:bs], bs)
        t0 = time.perf_counter()
        feats = eng.embed_tensors(tensors, bs)
    dt = time.perf_counter() - t0
    result["throughput"] = {"batch_size": bs, "images": n, "seconds": round(dt, 3),
                            "images_per_s": round(n / dt, 2), "ms_per_image": round(1000 * dt / n, 2)}
    emb_dim = int(feats.shape[1])

    if args.api:
        print("HTTP /embed…", flush=True)
        img, bb = get_frame(0)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        result["http_embed"] = bench_http(args.api, buf.getvalue(), bb, args.repeats, args.warmup)

    print("поиск по базе…", flush=True)
    sizes = [int(s) for s in args.gallery_sizes.split(",") if s.strip()]
    search_dim = settings.PCA_DIM or emb_dim
    result["search"] = bench_search(sizes, search_dim, args.repeats * 4,
                                    settings.DEFAULT_TOP_K, rng)

    print("переранжирование…", flush=True)
    qf = rng.standard_normal((args.n_query, emb_dim)).astype(np.float32)
    gf = rng.standard_normal((args.n_gallery, emb_dim)).astype(np.float32)
    t0 = time.perf_counter()
    eng.rerank(qf, gf)
    rr_s = time.perf_counter() - t0
    result["rerank"] = {"n_query": args.n_query, "n_gallery": args.n_gallery, "seconds": round(rr_s, 2)}
    per_img = result["throughput"]["seconds"] / n
    result["batch_estimate"] = {"n_query": args.n_query, "n_gallery": args.n_gallery,
                                "seconds": round((args.n_query + args.n_gallery) * per_img + rr_s, 1)}

    if args.full_batch:
        report = samples["report"] if samples else None
        q_csv = report and report["suggested"]["query_csv"]
        g_csv = report and report["suggested"]["gallery_csv"]
        if not (q_csv and g_csv):
            print("[!] --full-batch: в каталоге данных не найдены CSV запросов и галереи — пропуск",
                  file=sys.stderr)
        else:
            print("реальный пакетный прогон…", flush=True)
            mk = lambda df: ([os.path.join(report["images_dir"], f"{i}.jpg") for i in df.image_id],
                             df[["x", "y", "w", "h"]].to_numpy().tolist())
            qdf, gdf = read_csv(q_csv), read_csv(g_csv)
            t0 = time.perf_counter()
            qf, q_ok = eng.embed_paths(*mk(qdf), batch_size=bs)
            gf, g_ok = eng.embed_paths(*mk(gdf), batch_size=bs)
            t_embed = time.perf_counter() - t0
            eng.rerank(qf, gf)
            total = time.perf_counter() - t0
            result["batch_full"] = {"n_query": len(q_ok), "n_gallery": len(g_ok),
                                    "embed_seconds": round(t_embed, 1), "seconds": round(total, 1)}

    print_report(result)
    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(SERVICE_DIR, args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nрезультаты: {path}")


if __name__ == "__main__":
    main()
