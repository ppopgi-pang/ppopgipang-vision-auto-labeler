from typing import List
from modules.crawler.base import Crawler
from domain.image import ImageItem

from config import settings
from playwright.sync_api import sync_playwright
import time

class NaverCrawler(Crawler):
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        print(f"[NaverCrawler] Fetching for {keywords}...")
        
        image_items = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            for keyword in keywords:
                try:
                    # Navigate to Naver Images
                    page.goto(f"https://search.naver.com/search.naver?where=image&sm=tab_jum&query={keyword}")
                    
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2000)

                    # Scroll to load more images
                    # Scroll to load more images
                    last_height = page.evaluate("document.body.scrollHeight")
                    scroll_attempts = 0
                    max_scrolls = 100
                    
                    while scroll_attempts < max_scrolls:
                        page.keyboard.press("End")
                        page.wait_for_timeout(1000)
                        
                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            # Try waiting a bit longer
                            page.wait_for_timeout(1000)
                            new_height = page.evaluate("document.body.scrollHeight")
                            if new_height == last_height:
                                print("[NaverCrawler] Reached end of page.")
                                break
                                
                        last_height = new_height
                        scroll_attempts += 1

                    # Naver image selector often involves ._image or ._img
                    # Check for .tile_item img
                    
                    # Updated selector: .tile_item is the container, img is the child. ._image class is removed.
                    image_elements = page.locator(".tile_item img").all()
                    print(f"DEBUG: Found {len(image_elements)} potential image elements.")
                    
                    count = 0
                    for element in image_elements:
                        # Removed limit to fetch all available images
                        # if count >= 10: break
                            
                        try:
                           src = element.get_attribute("src")
                           # Naver sometimes puts real src in data-src (lazy load) but after scrolling it should be in src
                           if not src or "data:image" in src:
                               src = element.get_attribute("data-src")
                            
                           if src and src.startswith("http"):
                                image_items.append(ImageItem(
                                    url=src,
                                    keyword=keyword,
                                    source="naver"
                                ))
                                count += 1
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            
                except Exception as e:
                    print(f"Error crawling keyword {keyword}: {e}")

            browser.close()
            
        return image_items
