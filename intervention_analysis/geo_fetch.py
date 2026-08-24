"""GEO metadata retrieval helpers used by the frozen intervention workflow."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GEO_ENDPOINT = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
USER_AGENT = "MetaClockIntervention/1.0"


class RateLimiter:
    """Thread-safe limiter for NCBI metadata requests."""

    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                time.sleep(delay)
            self.next_time = max(now, self.next_time) + self.interval


def _fetch_text(gse: str, target: str, limiter: RateLimiter) -> tuple[str, str]:
    query = urlencode({"acc": gse, "targ": target, "view": "brief", "form": "text"})
    url = f"{GEO_ENDPOINT}?{query}"
    for attempt in range(6):
        limiter.wait()
        try:
            with urlopen(
                Request(url, headers={"User-Agent": USER_AGENT}), timeout=120
            ) as response:
                return url, response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
        except (
            ConnectionError,
            IncompleteRead,
            RemoteDisconnected,
            TimeoutError,
            URLError,
        ):
            if attempt == 5:
                raise
        time.sleep(2**attempt)
    raise RuntimeError(f"{gse}: unreachable retry state")


def _append_field(fields: dict[str, list[str]], key: str, value: str) -> None:
    fields.setdefault(key, []).append(value)


def _parse_series(text: str, expected_gse: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    accession = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("^SERIES = "):
            accession = line.split(" = ", 1)[1].strip()
        elif line.startswith("!Series_") and " = " in line:
            key, value = line[1:].split(" = ", 1)
            _append_field(fields, key, value)
    if accession != expected_gse:
        raise ValueError(
            f"{expected_gse}: expected SERIES metadata, received {accession or 'none'}"
        )
    return fields


def _parse_samples(text: str, expected_gse: str) -> list[dict[str, Any]]:
    if "!Sample_table_begin" in text or "<html" in text.lower():
        raise ValueError("unexpected data table or HTML in GEO brief response")
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("^SAMPLE = "):
            current = {"sample_id": line.split(" = ", 1)[1].strip(), "fields": {}}
            samples.append(current)
        elif current is not None and line.startswith("!Sample_") and " = " in line:
            key, value = line[1:].split(" = ", 1)
            _append_field(current["fields"], key, value)
    sample_ids = [sample["sample_id"] for sample in samples]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("GEO sample response is empty or contains duplicate GSM IDs")
    for sample in samples:
        fields = sample["fields"]
        if fields.get("Sample_geo_accession") != [sample["sample_id"]]:
            raise ValueError(f"{sample['sample_id']}: GEO accession mismatch")
        if expected_gse not in fields.get("Sample_series_id", []):
            raise ValueError(
                f"{sample['sample_id']}: missing {expected_gse} series link"
            )
    return samples


def fetch_gse(gse: str, output_dir: Path, limiter: RateLimiter) -> dict[str, Any]:
    """Fetch and cache validated GEO series/sample metadata."""
    output = output_dir / f"{gse}.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        return {"gse_id": gse, "status": "cached", "n_samples": len(payload["samples"])}
    series_url, series_text = _fetch_text(gse, "self", limiter)
    sample_url, sample_text = _fetch_text(gse, "gsm", limiter)
    series = _parse_series(series_text, gse)
    samples = _parse_samples(sample_text, gse)
    series_sample_ids = set(series.get("Series_sample_id", []))
    sample_ids = {sample["sample_id"] for sample in samples}
    if series_sample_ids and series_sample_ids != sample_ids:
        raise ValueError(
            f"{gse}: Series/Sample GSM mismatch ({len(series_sample_ids)} vs {len(sample_ids)})"
        )
    payload = {
        "gse_id": gse,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "series_url": series_url,
        "sample_url": sample_url,
        "series": series,
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return {"gse_id": gse, "status": "downloaded", "n_samples": len(samples)}
