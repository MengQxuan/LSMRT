import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


ROOT_API = "https://api.osf.io/v2/nodes/bj5w8/files/osfstorage/"


def fetch_json(url: str, retries: int = 8, timeout: int = 40):
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last_err = e
            print(
                f"[retry] fetch_json failed ({i+1}/{retries}) url={url} err={repr(e)}",
                flush=True,
            )
            sleep_s = min(8, 1.5 * (i + 1))
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch JSON: {url}\n{last_err}")


def crawl_manifest():
    q = deque([ROOT_API])
    seen_folder_api = set()
    seen_page_api = set()
    files = []
    page_cnt = 0
    while q:
        folder_api = q.popleft()
        if folder_api in seen_folder_api:
            continue
        seen_folder_api.add(folder_api)

        page_api = folder_api
        while page_api:
            if page_api in seen_page_api:
                break
            seen_page_api.add(page_api)
            payload = fetch_json(page_api)
            page_cnt += 1
            if page_cnt % 20 == 0:
                print(
                    f"[crawl] pages={page_cnt} folders={len(seen_folder_api)} files={len(files)}",
                    flush=True,
                )
            for item in payload["data"]:
                attr = item["attributes"]
                kind = attr["kind"]
                if kind == "folder":
                    child_url = item["relationships"]["files"]["links"]["related"]["href"]
                    q.append(child_url)
                elif kind == "file":
                    files.append(
                        {
                            "name": attr["name"],
                            "size": int(attr.get("size") or 0),
                            "materialized_path": attr["materialized_path"],  # e.g. /Vertical_Pokes/...
                            "download_url": item["links"]["download"],
                        }
                    )
            page_api = payload.get("links", {}).get("next")

    # De-duplicate by relative path.
    dedup = {}
    for f in files:
        dedup[f["materialized_path"]] = f
    return list(dedup.values())


def safe_relpath(materialized_path: str):
    # materialized_path always starts with "/"
    rel = materialized_path.lstrip("/")
    if not rel:
        raise ValueError(f"Bad materialized path: {materialized_path}")
    rel = rel.replace("\\", "/")
    return rel


def download_file(url: str, dst: Path, expected_size: int, timeout: int = 120):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == expected_size:
        return "skip"

    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, 6):
        req = urllib.request.Request(
            url, headers={"User-Agent": "LSMRT-RobotImpact-Downloader/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            got = tmp.stat().st_size
            if expected_size > 0 and got != expected_size:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Size mismatch for {dst}: got={got}, expected={expected_size}"
                )
            os.replace(tmp, dst)
            return "ok"
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == 5:
                raise
            sleep_s = min(8, 1.5 * attempt)
            print(
                f"[retry] download failed ({attempt}/5) file={dst.name} err={repr(e)}",
                flush=True,
            )
            time.sleep(sleep_s)


def main():
    ap = argparse.ArgumentParser("Download OSF Robot_impact_Data (bj5w8)")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/root/mqx/LSMRT/data/YCB-impact-sounds/Robot_impact_Data",
    )
    ap.add_argument("--manifest_only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Crawling OSF manifest...", flush=True)
    files = crawl_manifest()
    total_bytes = sum(x["size"] for x in files)
    print(
        f"files={len(files)} total_bytes={total_bytes} total_gb={total_bytes / 1024**3:.2f}",
        flush=True,
    )

    manifest_path = out_dir / "manifest_robot_impact_data.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)
    print(f"Manifest written: {manifest_path}", flush=True)

    if args.manifest_only:
        return

    print("[2/3] Downloading files...", flush=True)
    ok = 0
    skip = 0
    fail = 0
    for i, item in enumerate(files, 1):
        rel = safe_relpath(item["materialized_path"])
        dst = out_dir / rel
        try:
            status = download_file(item["download_url"], dst, item["size"])
            if status == "ok":
                ok += 1
            else:
                skip += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {i}/{len(files)} {rel}: {e}", file=sys.stderr, flush=True)
        if i % 25 == 0 or i == len(files):
            print(
                f"progress {i}/{len(files)} ok={ok} skip={skip} fail={fail}",
                flush=True,
            )

    print("[3/3] Done.", flush=True)
    print(f"ok={ok} skip={skip} fail={fail}", flush=True)
    if fail > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
