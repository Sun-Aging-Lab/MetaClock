#!/usr/bin/env python3
"""Fetch the two processed GSE223748 files required by the species workflow."""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from repository_paths import relative_path

from .species_config import (
    GSE223748_BETA_PATH,
    GSE223748_DOWNLOAD_DIR,
    GSE223748_SERIES_PATH,
)


ACCESSION = "GSE223748"
USER_AGENT = "MetaClock-GSE223748-fetch/1.0"
CHUNK_SIZE = 8 * 1024 * 1024
REPORT_INTERVAL = 256 * 1024 * 1024
MAX_ATTEMPTS = 5
FILES = (
    (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223748/"
        "suppl/GSE223748_datBetaNormalized.csv.gz",
        GSE223748_BETA_PATH,
    ),
    (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223748/"
        "matrix/GSE223748_series_matrix.txt.gz",
        GSE223748_SERIES_PATH,
    ),
)


def total_size(headers: object, offset: int) -> int | None:
    content_range = headers.get("Content-Range")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    content_length = headers.get("Content-Length")
    if content_length and str(content_length).isdigit():
        return offset + int(content_length)
    return None


def require_gzip(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            raise ValueError(f"Downloaded file is not gzip data: {path}")


def download_once(url: str, destination: Path) -> dict[str, object]:
    if destination.is_file() and destination.stat().st_size > 0:
        require_gzip(destination)
        print(f"[reuse] {destination}")
        return {
            "url": url,
            "filename": destination.name,
            "local_path": relative_path(destination),
            "size_bytes": destination.stat().st_size,
            "status": "reused",
        }

    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)

    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        expected = total_size(response.headers, offset)
        downloaded = offset
        next_report = downloaded + REPORT_INTERVAL
        print(f"[download] {url}")
        with partial.open(mode) as handle:
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    if expected:
                        print(f"  {downloaded / 1e9:.2f}/{expected / 1e9:.2f} GB")
                    else:
                        print(f"  {downloaded / 1e9:.2f} GB")
                    next_report += REPORT_INTERVAL
            handle.flush()
            os.fsync(handle.fileno())
        if expected is not None and partial.stat().st_size != expected:
            raise IOError(
                f"Incomplete download for {destination.name}: "
                f"{partial.stat().st_size} of {expected} bytes"
            )
        response_metadata = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }

    require_gzip(partial)
    os.replace(partial, destination)
    return {
        "url": url,
        "filename": destination.name,
        "local_path": relative_path(destination),
        "size_bytes": destination.stat().st_size,
        "status": "downloaded",
        **response_metadata,
    }


def download(url: str, destination: Path) -> dict[str, object]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return download_once(url, destination)
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = 5 * attempt
            print(f"[retry] {destination.name} in {delay}s ({attempt}/{MAX_ATTEMPTS})")
            time.sleep(delay)
    raise RuntimeError("Unreachable download retry state")


def main() -> None:
    GSE223748_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = GSE223748_DOWNLOAD_DIR / ".fetch.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = [download(url, destination) for url, destination in FILES]
        manifest = {
            "accession": ACCESSION,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": records,
        }
        manifest_path = GSE223748_DOWNLOAD_DIR / "download_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
