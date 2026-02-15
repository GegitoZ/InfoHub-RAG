import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

RAW_DIR = Path("data/raw_docs")
RAW_DIR.mkdir(parents=True, exist_ok=True)

WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    # normalize whitespace
    return WS_RE.sub(" ", text).strip()


def fetch_document(page, url: str):
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(800)

    title = page.title()

    # Grab visible text from the rendered page
    text = page.evaluate("() => document.body.innerText")
    text = clean_text(text)

    return {
        "url": url,
        "title": title,
        "text": text
    }


def main():
    links = json.load(open("data/doc_links.json", "r", encoding="utf-8"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        saved = 0
        for i, url in enumerate(links, start=1):
            try:
                doc = fetch_document(page, url)

                # Skip very short pages (menus, empty, etc.)
                if len(doc["text"]) < 800:
                    print(f"[{i}/{len(links)}] skipped (too short)")
                    continue

                out_path = RAW_DIR / f"doc_{saved:04d}.json"
                out_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                saved += 1
                print(f"[{i}/{len(links)}] saved {out_path.name}")

            except Exception as e:
                print(f"[{i}/{len(links)}] FAILED {url}: {e}")

        browser.close()

    print(f"\nDone. Saved {saved} documents into {RAW_DIR}")


if __name__ == "__main__":
    main()
