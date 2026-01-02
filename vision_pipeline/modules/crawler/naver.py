from typing import List
from modules.crawler.base import Crawler
from domain.image import ImageItem

from config import settings
from playwright.sync_api import sync_playwright
from modules.crawler.utils import run_sync_in_thread_if_event_loop
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class NaverCrawler(Crawler):
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        return run_sync_in_thread_if_event_loop(self._fetch_sync, keywords)

    def _fetch_single_keyword(self, keyword: str) -> List[ImageItem]:
        """단일 키워드에 대한 크롤링 수행"""
        image_items = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

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
                        page.wait_for_timeout(1000)
                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            print(f"[NaverCrawler-{keyword}] 페이지 끝에 도달")
                            break

                    last_height = new_height
                    scroll_attempts += 1

                # 이미지 URL 추출
                image_elements = page.locator(".tile_item img").all()
                print(f"[NaverCrawler-{keyword}] {len(image_elements)}개의 이미지 요소 발견")

                count = 0
                for element in image_elements:
                    try:
                       src = element.get_attribute("src")
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
                        print(f"[NaverCrawler-{keyword}] 이미지 처리 오류: {e}")

                print(f"[NaverCrawler-{keyword}] 완료: {len(image_items)}개 이미지 수집")

            except Exception as e:
                print(f"[NaverCrawler-{keyword}] 크롤링 오류: {e}")
            finally:
                browser.close()

        return image_items

    def _fetch_sync(self, keywords: List[str]) -> List[ImageItem]:
        print(f"[NaverCrawler] {len(keywords)}개 키워드에 대해 병렬 검색 시작...")

        image_items = []

        # 키워드를 병렬로 처리
        max_workers = min(len(keywords), 5)  # 최대 5개 동시 실행

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_keyword, keyword): keyword
                for keyword in keywords
            }

            for future in as_completed(futures):
                keyword = futures[future]
                try:
                    result = future.result()
                    image_items.extend(result)
                except Exception as e:
                    print(f"[NaverCrawler-{keyword}] 처리 실패: {e}")

        print(f"[NaverCrawler] 총 {len(image_items)}개 이미지 수집 완료")
        return image_items
