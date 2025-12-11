#!/usr/bin/env python3
"""
Enhanced homepage crawler with Selenium support for dynamic pages
"""

import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from datetime import datetime
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CRAWLED_DIR = PROCESSED_DIR / "crawled_homepages"

# Ensure directories exist
CRAWLED_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MAX_PROFESSORS_TO_CRAWL = None  # None for full crawl
START_IDX = 0
BATCH_SIZE = None  # None for full crawl
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 2
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def get_page_content_requests(url, timeout=REQUEST_TIMEOUT):
    """Get page content using requests (for static pages)"""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)

    return text


def get_page_content_selenium(url, timeout=REQUEST_TIMEOUT):
    """Get page content using Selenium (for dynamic pages)"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)

        # Wait for page to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Additional wait for dynamic content
        time.sleep(2)

        # Get page source
        page_source = driver.page_source

        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    finally:
        driver.quit()


def crawl_homepage(name, homepage, use_selenium=False):
    """
    Crawl a professor's homepage

    Args:
        name: Professor name
        homepage: Homepage URL
        use_selenium: Whether to use Selenium (for dynamic pages)

    Returns:
        dict: Crawl result
    """
    result = {
        "name": name,
        "homepage": homepage,
        "status": "pending",
        "content": "",
        "content_length": 0,
        "method": "unknown",
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not homepage or homepage.strip() == "":
        result["status"] = "no_homepage"
        return result

    # Try requests first
    for attempt in range(MAX_RETRIES):
        try:
            if not use_selenium:
                content = get_page_content_requests(homepage)
                result["method"] = "requests"
            else:
                content = get_page_content_selenium(homepage)
                result["method"] = "selenium"

            result["content"] = content
            result["content_length"] = len(content)
            result["status"] = "success"

            logger.info(
                f"✓ Crawled: {name} ({result['content_length']} chars, {result['method']})"
            )
            return result

        except Exception as e:
            logger.warning(
                f"✗ Attempt {attempt + 1}/{MAX_RETRIES} failed for {name}: {e}"
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                # If requests failed all retries, try selenium
                if not use_selenium:
                    logger.info(f"Retrying {name} with Selenium...")
                    try:
                        content = get_page_content_selenium(homepage)
                        result["content"] = content
                        result["content_length"] = len(content)
                        result["status"] = "success"
                        result["method"] = "selenium"
                        logger.info(
                            f"✓ Crawled with Selenium: {name} ({result['content_length']} chars)"
                        )
                        return result
                    except Exception as e2:
                        logger.error(f"✗ Selenium also failed for {name}: {e2}")

                result["status"] = "failed"
                result["error"] = str(e)

    return result


def save_result(result):
    """Save crawl result to individual file"""
    filename = result["name"].replace(" ", "_").replace("/", "_") + ".json"
    filepath = CRAWLED_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    """Main crawling function"""
    logger.info("=" * 80)
    logger.info("Enhanced Homepage Crawler")
    logger.info("=" * 80)

    # Load professors
    professors_file = PROCESSED_DIR / "professors.json"
    with open(professors_file, "r", encoding="utf-8") as f:
        professors = json.load(f)

    total = len(professors)
    logger.info(f"Loaded {total} professors")

    # Determine crawl range
    if MAX_PROFESSORS_TO_CRAWL is not None:
        end_idx = min(START_IDX + MAX_PROFESSORS_TO_CRAWL, total)
    elif BATCH_SIZE is not None:
        end_idx = min(START_IDX + BATCH_SIZE, total)
    else:
        end_idx = total

    to_crawl = professors[START_IDX:end_idx]

    logger.info(f"Crawling professors {START_IDX} to {end_idx} ({len(to_crawl)} total)")
    logger.info("=" * 80)

    # Crawl
    results = []
    success_count = 0
    failed_count = 0
    no_homepage_count = 0

    for i, prof in enumerate(to_crawl, 1):
        logger.info(f"\n[{i}/{len(to_crawl)}] Processing: {prof['name']}")

        result = crawl_homepage(prof["name"], prof.get("homepage", ""))

        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "no_homepage":
            no_homepage_count += 1
        else:
            failed_count += 1

        results.append(result)

        # Save individual result
        if result["status"] == "success":
            save_result(result)

        # Progress update
        if i % 100 == 0:
            logger.info(f"\nProgress: {i}/{len(to_crawl)} ({i/len(to_crawl)*100:.1f}%)")
            logger.info(
                f"Success: {success_count}, Failed: {failed_count}, No Homepage: {no_homepage_count}"
            )

        # Small delay to be polite
        time.sleep(0.5)

    # Save aggregated results
    batch_file = PROCESSED_DIR / f"crawled_batch_{START_IDX}_{end_idx}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Crawling Complete!")
    logger.info("=" * 80)
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Success: {success_count} ({success_count/len(results)*100:.1f}%)")
    logger.info(f"Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    logger.info(
        f"No homepage: {no_homepage_count} ({no_homepage_count/len(results)*100:.1f}%)"
    )
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Individual files: {CRAWLED_DIR}")
    logger.info(f"  - Batch file: {batch_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
