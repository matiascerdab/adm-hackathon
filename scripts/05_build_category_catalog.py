"""Build a reproducible catalog of up to 50 topics in each of 50 areas.

Usage: uv run python scripts/05_build_category_catalog.py --hours 5
The catalog records actual API-discovered subcategories; it never fabricates them.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from collections import deque

from wikimedia_common import CATALOG_PATH, StopCrawl, WikiClient, choose_random, write_json_atomic

# Human-readable area names and plausible en.wikipedia Category: root titles.
# An area with a nonexistent/empty root is marked underfilled and reported.
AREA_ROOTS = {
    "Physics": "Physics", "Chemistry": "Chemistry", "Biology": "Biology",
    "Genetics": "Genetics", "Ecology": "Ecology", "Astronomy": "Astronomy",
    "Geology": "Geology", "Meteorology": "Meteorology", "Oceanography": "Oceanography",
    "Paleontology": "Paleontology", "Computer science": "Computer science",
    "Artificial intelligence": "Artificial intelligence", "Mathematics": "Mathematics",
    "Statistics": "Statistics", "Civil engineering": "Civil engineering",
    "Mechanical engineering": "Mechanical engineering",
    "Electrical engineering": "Electrical engineering", "Medicine": "Medicine",
    "Agriculture": "Agriculture", "Materials science": "Materials science",
    "History": "History", "Geography": "Geography", "Economics": "Economics",
    "Political science": "Political science", "Sociology": "Sociology",
    "Psychology": "Psychology", "Philosophy": "Philosophy", "Law": "Law",
    "Education": "Education", "Anthropology": "Anthropology",
    "Literature": "Literature", "Film": "Film", "Music": "Music",
    "Painting": "Painting", "Sculpture": "Sculpture", "Architecture": "Architecture",
    "Photography": "Photography", "Theatre": "Theatre", "Linguistics": "Linguistics",
    "Mythology": "Mythology", "Association football": "Association football",
    "Basketball": "Basketball", "Tennis": "Tennis", "Athletics": "Sport of athletics",
    "Motorsport": "Motorsport", "Cycling": "Cycling", "Swimming": "Swimming",
    "Olympic Games": "Olympic Games", "Transport": "Transport", "Cuisine": "Cuisine"
}
assert len(AREA_ROOTS) == 50

BAD_TERMS = (
    "wikipedia", "articles needing", "articles with", "pages with", "pages using",
    "stubs", "stub", "maintenance", "templates", "wikidata", "wikiproject",
    "articles by", "cleanup", "redirects", "disambiguation", "tracking categories",
    "unsourced", "citation needed", "articles requiring", "articles lacking",
    "lists of", "list of", "categories by", "categories requiring", "articles containing",
)


def eligible_category(cat: str) -> bool:
    s = cat.removeprefix("Category:").casefold()
    return not any(bad in s for bad in BAD_TERMS)


def stable_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)


def fetch_subcategories(client: WikiClient, category: str, max_pages: int = 4):
    """Discover up to 4*500 subcategories with proper API continuation."""
    found = []
    continuation = {}
    for _ in range(max_pages):
        data = client.query(list="categorymembers", cmtitle=category,
                            cmtype="subcat", cmlimit=500,
                            cmprop="title|ids", **continuation)
        for member in data.get("query", {}).get("categorymembers", []):
            if member.get("ns") == 14 and eligible_category(member["title"]):
                found.append(member["title"])
        continuation = data.get("continue", {})
        if not continuation:
            break
    return list(dict.fromkeys(found))


def build_area(client: WikiClient, area: str, root: str, count: int):
    root = "Category:" + root
    first = fetch_subcategories(client, root)
    pool = list(first)
    # Broaden only when the root has too few subcategories.
    # Avoid an unbounded traversal of the Wikipedia category graph.
    if len(pool) < count:
        for child in choose_random(first, min(len(first), 20), stable_seed(area)):
            pool.extend(fetch_subcategories(client, child, max_pages=1))
            if len(set(pool)) >= count * 2:
                break
    pool = [x for x in dict.fromkeys(pool) if x != root and eligible_category(x)]
    chosen = choose_random(pool, count, stable_seed(area))
    return {"root": root, "topics": chosen,
            "available_topics_found": len(pool),
            "underfilled": len(chosen) < count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--topics-per-area", type=int, default=50)
    args = parser.parse_args()
    start = time.monotonic()
    client = WikiClient(delay=args.delay, deadline=start + args.hours * 3600)

    # Resume completed areas when rebuilding catalog.
    import json
    if CATALOG_PATH.exists():
        with CATALOG_PATH.open(encoding="utf-8") as f:
            catalog = json.load(f)
        if catalog.get("topics_per_area") != args.topics_per_area:
            raise SystemExit("Existing catalog was created using a different --topics-per-area")
    else:
        catalog = {"version": 1, "topics_per_area": args.topics_per_area,
                   "areas": {}}

    try:
        for area, root in AREA_ROOTS.items():
            if area in catalog["areas"]:
                continue
            entry = build_area(client, area, root, args.topics_per_area)
            catalog["areas"][area] = entry
            write_json_atomic(CATALOG_PATH, catalog)
            print(f"{len(catalog['areas'])}/50 {area}: {len(entry['topics'])} topics; "
                  f"underfilled={entry['underfilled']}", flush=True)
    except (StopCrawl, KeyboardInterrupt):
        print("Paused; the completed areas are saved.")
    finally:
        write_json_atomic(CATALOG_PATH, catalog)
        print(f"Saved {len(catalog['areas'])}/50 areas at {CATALOG_PATH}")
        print(f"API requests in this run: {client.requests_made}")


if __name__ == "__main__":
    main()
