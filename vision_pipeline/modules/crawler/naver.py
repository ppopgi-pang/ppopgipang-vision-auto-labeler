from typing import List
from modules.crawler.base import Crawler
from domain.image import ImageItem

from config import settings
from playwright.sync_api import sync_playwright
from modules.crawler.utils import run_sync_in_thread_if_event_loop
import time

class NaverCrawler(Crawler):
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        return run_sync_in_thread_if_event_loop(self._fetch_sync, keywords)

    def _fetch_sync(self, keywords: List[str]) -> List[ImageItem]:
        print(f"[NaverCrawler] Fetching for {keywords}...")
        
        image_items = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            for keyword in keywords:
                try:
                    # 네이버 이미지 검색으로 이동
                    page.goto(f"https://search.naver.com/search.naver?where=image&sm=tab_jum&query={keyword}")

                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2000)

                    # 더 많은 이미지를 로드하기 위해 스크롤
                    last_height = page.evaluate("document.body.scrollHeight")
                    scroll_attempts = 0
                    max_scrolls = 100

                    while scroll_attempts < max_scrolls:
                        page.keyboard.press("End")
                        page.wait_for_timeout(1000)

                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            # 조금 더 대기
                            page.wait_for_timeout(1000)
                            new_height = page.evaluate("document.body.scrollHeight")
                            if new_height == last_height:
                                print("[NaverCrawler] Reached end of page.")
                                break

                        last_height = new_height
                        scroll_attempts += 1

                    # 네이버 이미지 선택자는 보통 ._image 또는 ._img를 포함
                    # .tile_item img 확인

                    # 업데이트된 선택자: .tile_item이 컨테이너이고 img가 자식. ._image 클래스는 제거됨.
                    image_elements = page.locator(".tile_item img").all()
                    print(f"DEBUG: Found {len(image_elements)} potential image elements.")

                    count = 0
                    for element in image_elements:
                        # 사용 가능한 모든 이미지를 가져오기 위해 제한 제거
                        # if count >= 10: break

                        try:
                           src = element.get_attribute("src")
                           # 네이버는 때때로 실제 src를 data-src에 넣음 (lazy load) 하지만 스크롤 후에는 src에 있어야 함
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
