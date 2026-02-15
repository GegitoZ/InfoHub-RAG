import re
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from playwright.sync_api import sync_playwright

DOC_RE = re.compile(r"/ka/workspace/document/[0-9a-fA-F-]{36}")
BASE = "https://infohub.rs.ge"

OUT_FILE = Path("data/doc_links.json")

MAX_DOCS = 80          # how many document links you want total
TAKE = 10              # results per page (InfoHub seems to use 10)
MAX_PAGES = 20         # safety cap so we don't loop forever

# Put YOUR working search URL here (must be the one that shows docs)
START_URL = "https://infohub.rs.ge/ka/search?types=1&types=15&types=16&types=17&types=75&types=76&types=77"


def set_skip_take(url: str, skip: int, take: int) -> str:
    """Return same URL but with updated skip/take query params."""
    parts = urlparse(url)
    q = parse_qs(parts.query)

    # force skip/take
    q["skip"] = [str(skip)]
    q["take"] = [str(take)]

    # rebuild query string (doseq keeps repeated params like types=1&types=15...)
    new_query = urlencode(q, doseq=True)
    return urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, new_query, parts.fragment))


def collect_links():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    found = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page_num = 0
        while len(found) < MAX_DOCS and page_num < MAX_PAGES:
            skip = page_num * TAKE
            url = set_skip_take(START_URL, skip=skip, take=TAKE)

            print(f"Opening page {page_num + 1}: skip={skip} take={TAKE}")
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1200)

            html = page.content()
            matches = DOC_RE.findall(html)

            # If this page contains no doc links, we likely reached the end
            if not matches:
                print("No more document links found on this page. Stopping.")
                break

            before = len(found)
            for rel in matches:
                found.add(BASE + rel)
                if len(found) >= MAX_DOCS:
                    break

            added = len(found) - before
            print(f"Added {added} links (total {len(found)})")

            page_num += 1

        browser.close()

    links = sorted(found)[:MAX_DOCS]
    OUT_FILE.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(links)} links to {OUT_FILE}")


if __name__ == "__main__":
    collect_links()
