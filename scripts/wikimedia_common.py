"""Shared network and file helpers for the Wikipedia corpus collector."""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
REST = "https://en.wikipedia.org/w/rest.php/v1/page"
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DATA = PROJECT_DIR / "data"
# Set ADM_DATA_ROOT to a non-synced directory to avoid massive OneDrive syncs.
DATA_DIR = Path(os.environ.get("ADM_DATA_ROOT", str(PROJECT_DATA))).expanduser().resolve()
if os.name == "nt":
    default_db = Path(os.environ.get("LOCALAPPDATA", str(DATA_DIR))) / "adm-hackathon" / "crawler.sqlite3"
else:
    default_db = DATA_DIR / "crawler.sqlite3"
DB_PATH = Path(os.environ.get("ADM_CRAWLER_DB", str(default_db))).expanduser().resolve()
CATALOG_PATH = PROJECT_DATA / "category_catalog.json"


class StopCrawl(Exception):
    """Stop safely without marking the current operation as complete."""


class WikiClient:
    def __init__(self, delay: float = 1.0, deadline: float | None = None):
        user_agent = os.environ.get("WIKIMEDIA_USER_AGENT", "").strip()
        if not user_agent or "your-contact" in user_agent:
            raise SystemExit("Set WIKIMEDIA_USER_AGENT to an identifiable contact (see README).")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self.deadline = deadline
        self.requests_made = 0
        self._next_allowed = 0.0

    def wait(self, seconds: float):
        if self.deadline is not None and time.monotonic() + seconds >= self.deadline:
            raise StopCrawl("Time budget reached before next request")
        time.sleep(max(0.0, seconds))

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        for attempt in range(5):
            self.wait(max(0.0, self._next_allowed - time.monotonic()))
            if self.deadline is not None and time.monotonic() >= self.deadline - 45:
                raise StopCrawl("Time budget reached")
            try:
                response = self.session.get(url, params=params, timeout=(12, 35))
                self.requests_made += 1
                self._next_allowed = time.monotonic() + self.delay
                if response.status_code in (429, 503):
                    raw = response.headers.get("Retry-After", "")
                    try:
                        pause = max(5.0, float(raw))
                    except ValueError:
                        try:
                            when = parsedate_to_datetime(raw)
                            pause = max(5.0, (when - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError, OverflowError):
                            pause = min(120.0, 5.0 * 2 ** attempt)
                    self.wait(min(pause, 600.0))
                    continue
                response.raise_for_status()
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 4:
                    raise
                self.wait(min(5 * 2 ** attempt, 60))
        raise requests.exceptions.RequestException("Temporary rate limit / unavailable after retries")

    def query(self, **params) -> dict:
        params = {"action": "query", "format": "json", "formatversion": 2,
                  "maxlag": 5, **params}
        for attempt in range(5):
            result = self.get(API, params=params).json()
            error = result.get("error", {})
            if error.get("code") == "maxlag":
                self.wait(min(5 * 2 ** attempt, 60))
                continue
            if error:
                raise ValueError(f"MediaWiki API error: {error}")
            return result
        raise requests.exceptions.RequestException("API maxlag persisted after retries")


def write_json_atomic(path: Path, content: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def normalized_title(title: str) -> str:
    return " ".join(title.replace("_", " ").split())


def choose_random(items, n: int, seed: int):
    chosen = list(dict.fromkeys(items))
    random.Random(seed).shuffle(chosen)
    return chosen[:n]
