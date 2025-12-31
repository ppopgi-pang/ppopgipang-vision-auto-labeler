from typing import List
from modules.crawler.base import Crawler
from domain.image import ImageItem

from config import settings
from playwright.sync_api import sync_playwright

class GoogleCrawler(Crawler):
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        print(f"[GoogleCrawler] Fetching for {keywords}...")
        
        image_items = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True) # Set to True for production
            page = browser.new_page()
            
            for keyword in keywords:
                try:
                    # Navigate to Google Images
                    page.goto("https://www.google.com/imghp?hl=en&ogbl")
                    
                    # Accept cookies if present (EU specific but good practice)
                    try:
                        page.locator("button:has-text('Accept all')").click(timeout=2000)
                    except:
                        pass

                    # Search
                    search_box = page.locator("textarea[name='q']")
                    search_box.fill(keyword)
                    search_box.press("Enter")
                    
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2000) # Explicit wait for results

                    # Scroll to load more images
                    # Scroll to load more images
                    last_height = page.evaluate("document.body.scrollHeight")
                    scroll_attempts = 0
                    max_scrolls = 100  # Safety limit
                    
                    while scroll_attempts < max_scrolls:
                        page.keyboard.press("End")
                        page.wait_for_timeout(1000)
                        
                        # Check for "Show more results" button and click if visible
                        # Common selectors for the button: .mye4qd
                        try:
                            # Try multiple possible selectors or text
                            more_button = page.locator(".mye4qd, input[value='Show more results'], input[type='button'][value='Show more results']")
                            if more_button.is_visible():
                                print("[GoogleCrawler] Clicking 'Show more results' button...")
                                more_button.click()
                                page.wait_for_timeout(2000)
                        except Exception:
                            pass

                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            # Try waiting a bit longer to see if it's just slow network
                            page.wait_for_timeout(1000)
                            new_height = page.evaluate("document.body.scrollHeight")
                            if new_height == last_height:
                                print("[GoogleCrawler] Reached end of page or no new content.")
                                break
                        
                        last_height = new_height
                        scroll_attempts += 1

                    # Extract image URLs
                    # Google structure: div.isv-r contains the result. img inside.
                    # img.rg_i is the common class for the thumbnail.
                    
                    # Updated selector based on browser inspection (2024-05)
                    image_elements = page.locator("img.YQ4gaf").all()
                    print(f"DEBUG: Found {len(image_elements)} potential 'img.YQ4gaf' elements.")
                    
                    count = 0
                    for element in image_elements:
                        # Removed limit to fetch all available images
                        # if count >= 10: break
                            
                        try:
                           src = element.get_attribute("src")
                           if not src:
                               src = element.get_attribute("data-src")
                               
                           if src and src.startswith("http"):
                                image_items.append(ImageItem(
                                    url=src,
                                    keyword=keyword,
                                    source="google"
                                ))
                                count += 1
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            
                except Exception as e:
                    print(f"Error crawling keyword {keyword}: {e}")                            
                except Exception as e:
                    print(f"Error crawling keyword {keyword}: {e}")

            browser.close()
            
        return image_items
