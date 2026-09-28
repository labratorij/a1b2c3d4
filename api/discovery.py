import csv
import os
import re
import shutil
import zipfile
from typing import List, Optional, Tuple

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_COLUMNS = {"image_id", "x", "y", "w", "h"}
FALLBACK_COLUMNS = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]

ROLE_PATTERNS = (
    ("query", ("query", "probe", "запрос", "zapros")),
    ("gallery", ("gallery", "galer", "галере", "база", "baza")),
    ("train", ("train", "обуч", "obuch")),
)


def _count_images(path: str) -> int:
    n = 0
    with os.scandir(path) as it:
        for e in it:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXT:
                n += 1
    return n


def find_images_dir(root: str) -> Optional[str]:
    candidates = [os.path.join(root, "images"), root]
    with os.scandir(root) as it:
        candidates += sorted(e.path for e in it if e.is_dir())
    seen = set()
    best, best_n = None, 0
    for path in candidates:
        real = os.path.normpath(path)
        if real in seen or not os.path.isdir(real):
            continue
        seen.add(real)
        n = _count_images(real)
        if n > best_n:
            best, best_n = real, n
        if real == os.path.normpath(os.path.join(root, "images")) and n:
            return real
    return best


def _role_of(name: str) -> str:
    stem = os.path.splitext(name)[0].lower()
    for role, needles in ROLE_PATTERNS:
        if any(n in stem for n in needles):
            return role
    return "other"


def read_annotations(path: str):
    import pandas as pd

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first = next(csv.reader(f), [])
    if not first:
        raise ValueError(f"{os.path.basename(path)} пуст")
    if first[0].strip() == "image_id":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", header=None, names=FALLBACK_COLUMNS[:len(first)])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"в {os.path.basename(path)} нет колонок: {sorted(missing)}")
    return df


def describe_csv(path: str) -> dict:
    info = {"name": os.path.basename(path), "path": path, "role": _role_of(os.path.basename(path)),
            "rows": None, "has_vehicle_id": False, "usable": False, "problem": None,
            "inferred_role": False}
    try:
        df = read_annotations(path)
    except Exception as e:
        info["problem"] = str(e)
        return info
    info["rows"] = int(len(df))
    info["has_vehicle_id"] = "vehicle_id" in df.columns
    info["usable"] = True
    return info


def scan_csvs(root: str) -> List[dict]:
    with os.scandir(root) as it:
        paths = sorted(e.path for e in it if e.is_file() and e.name.lower().endswith(".csv"))
    return [describe_csv(p) for p in paths]


def infer_missing_roles(csvs: List[dict]) -> None:
    usable = [c for c in csvs if c["usable"]]
    unknown = [c for c in usable if c["role"] == "other"]
    if len(usable) != 2 or len(unknown) != 1:
        return
    known = next(c for c in usable if c["role"] != "other")
    pair = {"query": "gallery", "gallery": "query"}.get(known["role"])
    if pair:
        unknown[0]["role"] = pair
        unknown[0]["inferred_role"] = True


def pick(csvs: List[dict], role: str) -> Optional[dict]:
    matching = [c for c in csvs if c["usable"] and c["role"] == role]
    if not matching:
        return None
    matching.sort(key=lambda c: (not c["name"].lower().startswith("test"), c["name"]))
    return matching[0]


def pick_for_gallery(csvs: List[dict]) -> Optional[dict]:
    usable = [c for c in csvs if c["usable"]]
    labelled = [c for c in usable if c["has_vehicle_id"] and c["role"] in ("gallery", "train")]
    if labelled:
        labelled.sort(key=lambda c: (c["role"] != "gallery", c["name"]))
        return labelled[0]
    return pick(csvs, "gallery") or (usable[0] if len(usable) == 1 else None)


SAFE_NAME = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._-]+")


def safe_name(name: str, fallback: str = "file") -> str:
    base = os.path.basename(str(name).replace("\\", "/").rstrip("/"))
    base = SAFE_NAME.sub("_", base).strip("._-")
    return base or fallback


def workspace_path(upload_dir: str, name: str) -> str:
    return os.path.join(upload_dir, safe_name(name, "dataset"))


def place(root: str, filename: str, data: bytes) -> Optional[str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXT:
        target = os.path.join(root, "images", safe_name(filename))
    elif ext == ".csv":
        target = os.path.join(root, safe_name(filename))
    else:
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(data)
    return target


def unpack_zip(root: str, data: bytes) -> Tuple[int, int]:
    import io as _io

    written = skipped = 0
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                skipped += 1
                continue
            if place(root, os.path.basename(name), z.read(info)) is None:
                skipped += 1
            else:
                written += 1
    return written, skipped


def list_workspaces(upload_dir: str) -> List[dict]:
    if not os.path.isdir(upload_dir):
        return []
    out = []
    with os.scandir(upload_dir) as it:
        for e in sorted(it, key=lambda e: e.name):
            if not e.is_dir():
                continue
            images = find_images_dir(e.path)
            out.append({"name": e.name, "path": e.path,
                        "images": _count_images(images) if images else 0,
                        "csvs": len([c for c in scan_csvs(e.path) if c["usable"]])})
    return out


def drop_workspace(upload_dir: str, name: str) -> bool:
    path = workspace_path(upload_dir, name)
    if not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True


def inspect(root: str) -> dict:
    images_dir = find_images_dir(root)
    csvs = scan_csvs(root)
    infer_missing_roles(csvs)
    query, gallery = pick(csvs, "query"), pick(csvs, "gallery")
    bulk = pick_for_gallery(csvs)
    problems = []
    if images_dir is None:
        problems.append("не найден каталог с изображениями: ни сам путь, ни его подкаталоги "
                        "не содержат файлов " + "/".join(sorted(IMAGE_EXT)))
    if not any(c["usable"] for c in csvs):
        problems.append("не найдено ни одного CSV с колонками image_id,x,y,w,h")
    return {
        "root": root,
        "images_dir": images_dir,
        "images": _count_images(images_dir) if images_dir else 0,
        "csvs": csvs,
        "suggested": {
            "query_csv": query["path"] if query else None,
            "gallery_csv": gallery["path"] if gallery else None,
            "bulk_csv": bulk["path"] if bulk else None,
        },
        "problems": problems,
    }
