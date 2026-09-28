import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))


def wait_for(url: str, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    print(f"[!] {label} не ответил за {timeout:.0f} с", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000, help="порт backend")
    ap.add_argument("--ui-port", type=int, default=8501, help="порт веб-интерфейса")
    ap.add_argument("--host", default="127.0.0.1", help="адрес прослушивания")
    ap.add_argument("--api-only", action="store_true", help="не запускать интерфейс")
    ap.add_argument("--ui-only", action="store_true", help="не запускать backend (уже запущен)")
    ap.add_argument("--data-root", default=None, help="каталог с данными (иначе service/data/dataset)")
    ap.add_argument("--reload", action="store_true", help="перезапуск backend при правке кода")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("SERVICE_DIR", SERVICE_DIR)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.data_root:
        env["DATA_ROOT"] = os.path.abspath(args.data_root)
        env["UI_DATA_ROOT"] = env["DATA_ROOT"]
        env["UI_IMAGES_DIR"] = os.path.join(env["DATA_ROOT"], "images")
    api_url = f"http://{args.host}:{args.port}"
    env.setdefault("API_URL", api_url)

    procs = []
    try:
        if not args.ui_only:
            cmd = [sys.executable, "-m", "uvicorn", "api.main:app",
                   "--host", args.host, "--port", str(args.port), "--timeout-keep-alive", "120"]
            if args.reload:
                cmd.append("--reload")
            print(f"[1/2] backend:   {api_url}/docs   (Swagger)")
            procs.append(subprocess.Popen(cmd, cwd=SERVICE_DIR, env=env))
            if not wait_for(f"{api_url}/openapi.json", 180, "backend"):
                raise SystemExit(1)

        if not args.api_only:
            cmd = [sys.executable, "-m", "streamlit", "run", os.path.join("ui", "app.py"),
                   "--server.port", str(args.ui_port), "--server.address", args.host,
                   "--server.headless", "true", "--browser.gatherUsageStats", "false"]
            print(f"[2/2] интерфейс: http://{args.host}:{args.ui_port}")
            procs.append(subprocess.Popen(cmd, cwd=SERVICE_DIR, env=env))

        print("\nCtrl+C — остановить\n")
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"[!] процесс завершился с кодом {p.returncode}", file=sys.stderr)
                    raise SystemExit(p.returncode or 1)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nостановка…")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
