import asyncio
import csv
import os
import yt_dlp
import pandas as pd
from playwright.async_api import async_playwright

# ================= CONFIGURATION =================
OUTPUT_DIR = "output/hksl_sign_language"
BASE_URL = "https://www.cslds.org/hkslbrowser/"
LIST_URL = f"{BASE_URL}databank.jsp?page=1&gcid=-1&gcvalue=(All%20vocabulary)&ns=-1&hs=-1"
CSV_FILENAME = "hksl_video_manifest.csv"
BROWSER_FOR_COOKIES = "chrome"  # or 'firefox', 'edge'
# =================================================

async def get_all_gloss_links(page):
    """Navigates pagination and collects all detail page URLs."""
    glosses = []
    print("--- Phase 1: Collecting Gloss Links ---")
    
    while True:
        await page.wait_for_selector("#search-result")
        links = await page.query_selector_all("#search-result div.row.equal a.btn-primary")
        
        for link in links:
            name = await link.inner_text()
            href = await link.get_attribute("href")
            glosses.append({"name": name.strip(), "url": f"{BASE_URL}{href}"})
        
        # Robust pagination: find the 'Next' button at the end of the list
        next_button = await page.query_selector("#paging ul li:last-child a")
        if next_button:
            text = await next_button.inner_text()
            if "Next" in text or "»" in text:
                print(f"Collected {len(glosses)} links... moving to next page.")
                await next_button.click()
                await page.wait_for_load_state("networkidle")
            else:
                break
        else:
            break
    return glosses

async def get_panopto_embeds(page, gloss_list):
    """Visits each detail page to find the Panopto iframe source."""
    print("\n--- Phase 2: Extracting Video Embed URLs ---")
    results = []
    for i, gloss in enumerate(gloss_list):
        try:
            await page.goto(gloss['url'], timeout=60000)
            iframe_selector = "iframe.resp-iframe-100"
            await page.wait_for_selector(iframe_selector, timeout=5000)
            
            iframe = await page.query_selector(iframe_selector)
            if iframe:
                src = await iframe.get_attribute("src")
                results.append({"name": gloss['name'], "embed_url": src})
                print(f"[{i+1}/{len(gloss_list)}] Found embed for: {gloss['name']}")
        except Exception:
            print(f"[{i+1}/{len(gloss_list)}] Skipping {gloss['name']} (No video/timeout)")
            
    return results

def download_videos(results_list):
    """Uses yt-dlp to download and merge the HLS streams."""
    print("\n--- Phase 3: Downloading Videos via yt-dlp ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'cookiesfrombrowser': (BROWSER_FOR_COOKIES,),
        'quiet': True,
        'no_warnings': True,
    }

    for i, item in enumerate(results_list):
        # Sanitize filename
        safe_name = "".join([c for c in item['name'] if c.isalnum() or c in (' ', '-', '_')]).strip()
        ydl_opts['outtmpl'] = f"{OUTPUT_DIR}/{safe_name}.%(ext)s"
        
        print(f"[{i+1}/{len(results_list)}] Downloading MP4: {item['name']}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([item['embed_url']])
        except Exception as e:
            print(f"Failed download for {item['name']}: {e}")

async def main():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        page = await context.new_page()

        # Step 1 & 2: Scrape metadata
        await page.goto(LIST_URL)
        gloss_links = await get_all_gloss_links(page)
        video_data = await get_panopto_embeds(page, gloss_links)
        
        # Save CSV backup
        with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "embed_url"])
            writer.writeheader()
            writer.writerows(video_data)
        
        await browser.close()

    # Step 3: Automated Download
    download_videos(video_data)
    print(f"\nTask Complete! Videos are in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    asyncio.run(main())