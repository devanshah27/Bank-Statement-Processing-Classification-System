"""Vendor script to download HTMX and Chart.js locally into public/ directory."""

import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

VENDORS = {
    "htmx.min.js": "https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js",
    "chart.min.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js",
}


def download_vendors():
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    for filename, url in VENDORS.items():
        dest = PUBLIC_DIR / filename
        if dest.is_file() and dest.stat().st_size > 1000:
            logger.info("%s already exists (%d bytes)", filename, dest.stat().st_size)
            continue

        logger.info("Downloading %s from %s...", filename, url)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
                out_file.write(response.read())
            logger.info("Successfully downloaded %s (%d bytes)", filename, dest.stat().st_size)
        except Exception as e:
            logger.warning("Could not download %s from network: %s. Creating local fallback stub.", filename, e)
            if not dest.exists():
                dest.write_text(f"/* Fallback for {filename} */\nconsole.log('{filename} loaded');", encoding="utf-8")


if __name__ == "__main__":
    download_vendors()

