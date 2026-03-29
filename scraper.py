"""
NTU Sign Bank Scraper
---------------------
python -m venv .venv
.venv\Scripts\activate.bat
pip install requests beautifulsoup4

Output structure:
    output/
        gifs/
            Abuse.gif
            ...
        manifest.json   ← { "Abuse": "gifs/Abuse.gif", ... }
"""

import json
import os
import time

import requests
from bs4 import BeautifulSoup


# ── Config ────────────────────────────────────────────────────────────────────

INDEX_URL   = "https://blogs.ntu.edu.sg/sgslsignbank/signs/"
OUTPUT_DIR  = "output"
GIF_DIR     = os.path.join(OUTPUT_DIR, "gifs")
MANIFEST    = os.path.join(OUTPUT_DIR, "manifest.json")
REQUEST_DELAY = 1.0  # seconds between requests — be polite

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup | None:
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [Error] {url} — {e}")
        return None


def download_file(url: str, dest: str) -> bool:
    try:
        resp = SESSION.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"  [Error] {url} — {e}")
        return False


# ── Scraping ──────────────────────────────────────────────────────────────────

def scrape_index() -> list[tuple[str, str]]:
    """Return [(word, sign_page_url), ...] from the index page."""
    print(f"Fetching index: {INDEX_URL}")
    soup = get_soup(INDEX_URL)
    if not soup:
        raise RuntimeError("Failed to fetch index page.")

    container = soup.select_one("#post-2063 div.row.text-center.mb-5")
    if not container:
        # Fallback: grab all sign buttons anywhere on the page
        print("  Warning: target container not found, falling back to page-wide search.")
        links = soup.find_all("a", class_=lambda c: c and "sign" in c and "btn-red" in c)
    else:
        links = container.find_all("a", class_=lambda c: c and "sign" in c and "btn-red" in c)

    signs = [(a.get_text(strip=True), a["href"]) for a in links if a.get("href")]
    print(f"Found {len(signs)} signs.")
    return signs


def get_gif_url(sign_page_url: str) -> str | None:
    """Visit a sign detail page and return the absolute GIF src."""
    soup = get_soup(sign_page_url)
    if not soup:
        return None

    # Primary selector matching the known page structure
    img = soup.select_one("div.col-lg-7 img.img-fluid")

    # Fallback: any img whose alt ends with '-demo'
    if not img:
        img = soup.find("img", alt=lambda a: a and a.lower().endswith("-demo"))

    # Last resort: first .gif img on the page
    if not img:
        img = soup.find("img", src=lambda s: s and s.lower().endswith(".gif"))

    if not img or not img.get("src"):
        return None

    src = img["src"]
    # Resolve relative URLs
    if not src.startswith("http"):
        from urllib.parse import urljoin
        src = urljoin(sign_page_url, src)

    return src


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GIF_DIR, exist_ok=True)

    # Resume support — load existing manifest
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest: dict = json.load(f)
        print(f"Resuming — {len(manifest)} signs already processed.")
    else:
        manifest = {}

    signs = scrape_index()

    for i, (word, sign_url) in enumerate(signs, 1):
        if word in manifest:
            print(f"[{i}/{len(signs)}] Skipping '{word}' (already done)")
            continue

        print(f"[{i}/{len(signs)}] {word}")
        time.sleep(REQUEST_DELAY)

        gif_url = get_gif_url(sign_url)

        if not gif_url:
            print(f"  ⚠ No GIF found for '{word}'")
            manifest[word] = None
        else:
            safe_name = word.replace("/", "-").replace("\\", "-").strip()
            gif_path  = os.path.join(GIF_DIR, f"{safe_name}.gif")
            time.sleep(REQUEST_DELAY)

            if download_file(gif_url, gif_path):
                rel_path = os.path.relpath(gif_path, OUTPUT_DIR)
                manifest[word] = rel_path
                size_kb  = os.path.getsize(gif_path) // 1024
                print(f"  ✓ {rel_path} ({size_kb} KB)")
            else:
                print(f"  ✗ Download failed for '{word}'")
                manifest[word] = None

        # Persist after every sign so a crash loses nothing
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    succeeded = sum(1 for v in manifest.values() if v)
    print(f"\nDone. {succeeded}/{len(manifest)} signs downloaded.")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()