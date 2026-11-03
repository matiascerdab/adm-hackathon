
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
STATE_PATH = PROJECT_DIR / "data" / "crawler_state.json"
LINKS_PATH = RAW_DIR / "article_links.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_ARTICLES = 100
MIN_WORDS = 150
MAX_LINKS_PER_ARTICLE = 400
REQUEST_DELAY = 1.0

API_URL = "https://en.wikipedia.org/w/api.php"
REST_URL = "https://en.wikipedia.org/w/rest.php/v1/page"

session = requests.Session()
session.headers.update({
    "User-Agent": "ADM-Hackathon/0.1 (your-contact-email)"
})


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def save_json(path, data):
    """Write data as JSON."""
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    temporary_path.replace(path)


def get_response(url, params=None):
    """Make a request with a delay and limited retries."""

    for attempt in range(3):
        try:
            response = session.get(
                url,
                params=params,
                timeout=31
            )

            if response.status_code in (429, 503):
                retry_after = response.headers.get(
                    "Retry-After", "5"
                )

                try:
                    delay = max(5, float(retry_after))
                except ValueError:
                    delay = 5 * (2 ** attempt)

                time.sleep(delay)
                continue

            response.raise_for_status()
            return response

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(5 * (2 ** attempt))

        finally:
            time.sleep(REQUEST_DELAY)

    raise requests.exceptions.RequestException(
        f"Request failed after retries: {url}"
    )


def download_article(title):
    """Retrieve the article and extract plain text."""

    formatted_title = quote(
        title.replace(" ", "_"),
        safe=""
    )

    url = f"{REST_URL}/{formatted_title}/with_html"

    response = get_response(url)
    article = response.json()

    html = article["html"]
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "sup"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    if len(text.split()) < MIN_WORDS:
        return None

    page_id = article["id"]

    raw_path = RAW_DIR / f"{page_id}.json"
    processed_path = PROCESSED_DIR / f"{page_id}.json"

    processed_article = {
        "document_id": page_id,
        "title": article["title"],
        "text": text,
        "word_count": len(text.split())
    }

    save_json(raw_path, article)
    save_json(processed_path, processed_article)

    return processed_article


def get_article_links(title):
    """Retrieve outgoing links, including paginated results."""

    titles = []
    continuation = {}

    while len(titles) < MAX_LINKS_PER_ARTICLE:

        params = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "titles": title,
            "plnamespace": 0,
            "pllimit": min(
                500,
                MAX_LINKS_PER_ARTICLE - len(titles)
            ),
            **continuation
        }

        response = get_response(API_URL, params=params)
        data = response.json()

        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            for link in page.get("links", []):
                titles.append(link["title"])

        if "continue" not in data:
            break

        continuation = data["continue"]

    return titles


# --------------------------------------------------
# 3. INITIALIZE OR RESTORE BFS
# --------------------------------------------------

if STATE_PATH.exists():

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    queue = deque(state["queue"])
    visited = set(state["visited"])
    discovered = set(state["discovered"])

    print("Resuming previous crawl.")

else:

    with open(LINKS_PATH, "r", encoding="utf-8") as f:
        titles = json.load(f)

    queue = deque(dict.fromkeys(titles))
    visited = set()
    discovered = set(queue)

    print("Starting new crawl.")


downloaded_count = len(list(PROCESSED_DIR.glob("*.json")))

print("Already downloaded:", downloaded_count)
print("Articles in queue:", len(queue))


# --------------------------------------------------
# 4. BFS CRAWLING LOOP
# --------------------------------------------------

while queue and downloaded_count < MAX_ARTICLES:

    current_title = queue[0]

    print(f"\nProcessing: {current_title}")

    if current_title not in visited:

        try:
            # Download and save article
            article = download_article(current_title)

            if article is not None:

                page_id = article["document_id"]
                processed_path = (
                    PROCESSED_DIR / f"{page_id}.json"
                )

                # Count distinct saved documents
                downloaded_count = len(
                    list(PROCESSED_DIR.glob("*.json"))
                )

                print(
                    f"Saved: {article['title']} "
                    f"({article['word_count']} words)"
                )

                # Discover more Wikipedia articles
                new_links = get_article_links(current_title)

                added = 0

                for title in new_links:

                    if title not in discovered:
                        queue.append(title)
                        discovered.add(title)
                        added += 1

                print("New links added:", added)

            else:
                print("Skipped: article too short.")

        except requests.exceptions.RequestException as error:
            print("Request error:", error)

        except (ValueError, KeyError, TypeError) as error:
            print("Invalid article data:", error)

        visited.add(current_title)

    # Remove processed title from the queue
    queue.popleft()

    # Save progress after every article
    save_json(STATE_PATH, {
        "queue": list(queue),
        "visited": sorted(visited),
        "discovered": sorted(discovered)
    })

    print("Downloaded:", downloaded_count)
    print("Queue size:", len(queue))


print("\nCrawling finished.")
print("Total downloaded articles:", downloaded_count)
print("Remaining titles:", len(queue))
