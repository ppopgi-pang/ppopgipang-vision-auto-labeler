from typing import List
from modules.crawler.base import Crawler
from domain.image import ImageItem

from config import settings
from playwright.sync_api import sync_playwright
from modules.crawler.utils import run_sync_in_thread_if_event_loop
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class GoogleCrawler(Crawler):
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        return run_sync_in_thread_if_event_loop(self._fetch_sync, keywords)

    def _fetch_single_keyword_with_retry(self, keyword: str, max_retries: int = 3) -> List[ImageItem]:
        """재시도 로직이 있는 키워드 크롤링"""
        for attempt in range(max_retries):
            try:
                return self._fetch_single_keyword(keyword)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1초, 2초, 4초
                    print(f"[GoogleCrawler-{keyword}] 시도 {attempt + 1}/{max_retries} 실패: {e}")
                    print(f"[GoogleCrawler-{keyword}] {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"[GoogleCrawler-{keyword}] 모든 재시도 실패: {e}")
                    return []
        return []

    def _fetch_single_keyword(self, keyword: str) -> List[ImageItem]:
        """단일 키워드에 대한 크롤링 수행"""
        image_items = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                # Google 이미지 검색으로 직접 이동 (더 빠름)
                from urllib.parse import quote
                encoded_keyword = quote(keyword)
                search_url = f"https://www.google.com/search?q={encoded_keyword}&tbm=isch&hl=ko"
                
                # 타임아웃을 60초로 늘리고, domcontentloaded로 변경 (더 빠름)
                page.goto(search_url, timeout=60000, wait_until="domcontentloaded")

                # 이미지가 로드될 때까지 잠시 대기
                page.wait_for_timeout(3000)

                # 더 많은 이미지를 로드하기 위해 스크롤
                last_height = page.evaluate("document.body.scrollHeight")
                scroll_attempts = 0
                max_scrolls = 50  # 적당한 스크롤 횟수
                no_change_count = 0 
                max_no_change = 3

                while scroll_attempts < max_scrolls:
                    page.keyboard.press("End")
                    page.wait_for_timeout(1500)

                    # "더 보기" 버튼을 확인하고 표시되면 클릭
                    try:
                        # 다양한 선택자 시도 (a 태그 기반 - 최신 구글)
                        more_button_selectors = [
                            "a.T7sFge",
                            "a[jsname='oHxHid']",
                            "a:has-text('결과 더보기')",
                            "a:has-text('Show more results')",
                            ".mye4qd",
                        ]
                        
                        for selector in more_button_selectors:
                            try:
                                more_button = page.locator(selector).first
                                if more_button.is_visible(timeout=500):
                                    print(f"[GoogleCrawler-{keyword}] '더 보기' 버튼 발견 ({selector}), 클릭 중...")
                                    more_button.click(timeout=3000)
                                    page.wait_for_timeout(2000)
                                    no_change_count = 0
                                    break
                            except:
                                continue
                    except Exception:
                        pass

                    new_height = page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        no_change_count += 1
                        if no_change_count >= max_no_change:
                             print(f"[GoogleCrawler-{keyword}] 페이지 끝에 도달 (높이 변화 없음)")
                             break
                        else:
                             page.wait_for_timeout(2000)
                    else:
                        last_height = new_height
                        no_change_count = 0
                    
                    scroll_attempts += 1

                print(f"[GoogleCrawler-{keyword}] 스크롤 완료, 이미지 추출 시작...")

                # 이미지 URL 추출 - 여러 방법 시도
                # 방법 1: 썸네일 컨테이너에서 이미지 추출
                image_elements = page.locator("div.F0uyec img, img.YQ4gaf").all()
                print(f"[GoogleCrawler-{keyword}] {len(image_elements)}개의 이미지 요소 발견")

                seen_urls = set()
                for element in image_elements:
                    try:
                        src = element.get_attribute("src")
                        if not src:
                            src = element.get_attribute("data-src")

                        if src:
                            # HTTP URL인 경우 (스크롤 후 로딩된 이미지)
                            if src.startswith("http"):
                                if src not in seen_urls:
                                    seen_urls.add(src)
                                    image_items.append(ImageItem(
                                        url=src,
                                        keyword=keyword,
                                        source="google"
                                    ))
                            # Base64 인코딩된 경우 (초기 로드 이미지) - 작은 썸네일이므로 스킵할 수도 있음
                            # 고화질 이미지가 필요하면 썸네일을 클릭해야 하지만, 기본적으로는 base64도 수집
                            elif src.startswith("data:image"):
                                # base64 이미지는 썸네일이라 품질이 낮음 - HTTP URL로 된 것만 수집
                                # 필요시 아래 주석 해제하여 base64도 수집 가능
                                # image_items.append(ImageItem(
                                #     url=src,
                                #     keyword=keyword,
                                #     source="google"
                                # ))
                                pass
                    except Exception as e:
                        print(f"[GoogleCrawler-{keyword}] 이미지 처리 오류: {e}")

                # 만약 HTTP URL이 너무 적으면 (10개 미만), 고해상도 이미지 추출 시도
                if len(image_items) < 10:
                    print(f"[GoogleCrawler-{keyword}] HTTP URL 이미지가 부족함, 고해상도 이미지 추출 시도...")
                    try:
                        # 썸네일 클릭하여 고해상도 이미지 URL 추출
                        thumbnails = page.locator("div.F0uyec").all()[:min(50, len(image_elements))]
                        for i, thumbnail in enumerate(thumbnails):
                            try:
                                thumbnail.click(timeout=2000)
                                page.wait_for_timeout(1000)
                                
                                # 고해상도 이미지 찾기
                                large_img = page.locator("img.sFlh5c, img.pT0Scc, .ivg-i img").first
                                if large_img.is_visible(timeout=500):
                                    large_src = large_img.get_attribute("src")
                                    if large_src and large_src.startswith("http") and large_src not in seen_urls:
                                        seen_urls.add(large_src)
                                        image_items.append(ImageItem(
                                            url=large_src,
                                            keyword=keyword,
                                            source="google"
                                        ))
                            except:
                                continue
                    except Exception as e:
                        print(f"[GoogleCrawler-{keyword}] 고해상도 이미지 추출 실패: {e}")

                print(f"[GoogleCrawler-{keyword}] 완료: {len(image_items)}개 이미지 수집")

            except Exception as e:
                print(f"[GoogleCrawler-{keyword}] 크롤링 오류: {e}")
            finally:
                browser.close()

        return image_items

    def _fetch_sync(self, keywords: List[str]) -> List[ImageItem]:
        print(f"[GoogleCrawler] {len(keywords)}개 키워드에 대해 병렬 검색 시작...")

        image_items = []

        # 키워드를 병렬로 처리 (코랩 T4 환경에서 10개 동시 실행 가능)
        max_workers = min(len(keywords), 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_keyword_with_retry, keyword): keyword
                for keyword in keywords
            }

            for future in as_completed(futures):
                keyword = futures[future]
                result = future.result()  # 재시도 로직 내부에서 이미 예외 처리됨
                image_items.extend(result)

        print(f"[GoogleCrawler] 총 {len(image_items)}개 이미지 수집 완료")
        return image_items
