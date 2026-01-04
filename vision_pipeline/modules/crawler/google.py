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
                # Google 이미지 검색으로 이동
                page.goto("https://www.google.com/imghp?hl=en&ogbl")

                # 쿠키 동의 버튼이 있으면 클릭
                try:
                    page.locator("button:has-text('Accept all')").click(timeout=2000)
                except:
                    pass

                # 검색
                search_box = page.locator("textarea[name='q']")
                search_box.fill(keyword)
                search_box.press("Enter")

                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(2000)

                # 더 많은 이미지를 로드하기 위해 스크롤
                last_height = page.evaluate("document.body.scrollHeight")
                scroll_attempts = 0
                max_scrolls = 500  # 스크롤 제한 대폭 증가
                no_change_count = 0 
                max_no_change = 3   # 높이 변화 없음 허용 횟수

                while scroll_attempts < max_scrolls:
                    page.keyboard.press("End")
                    page.wait_for_timeout(1000)

                    # 가끔 스크롤을 살짝 올려서 lazy loading 트리거
                    if scroll_attempts % 5 == 0:
                        page.mouse.wheel(0, -500)
                        page.wait_for_timeout(500)
                        page.keyboard.press("End")
                        page.wait_for_timeout(1000)

                    # "더 보기" 버튼을 확인하고 표시되면 클릭
                    try:
                        # 다양한 선택자 시도
                        more_button_selectors = [
                            ".mye4qd", 
                            "input[value='Show more results']", 
                            "input[type='button'][value='Show more results']",
                            "div[data-l='Show more results']",
                            # Browser analysis findings:
                            "a.T7sFge",  # New Google Images button class
                            "a[jsname='oHxHid']", # New Google Images button jsname
                            "span:has-text('Show more results')",
                            "span:has-text('결과 더보기')",
                            # Generic text fallback for buttons/links
                            "a:has-text('Show more results')",
                            "a:has-text('결과 더보기')",
                            "div:has-text('결과 더보기')"
                        ]
                        
                        clicked = False
                        for selector in more_button_selectors:
                            more_button = page.locator(selector).first
                            if more_button.is_visible():
                                print(f"[GoogleCrawler-{keyword}] '더 보기' 버튼 발견 ({selector}), 클릭 중...")
                                try:
                                    more_button.click(timeout=3000)
                                    page.wait_for_timeout(2000)
                                    clicked = True
                                    no_change_count = 0 # 버튼 클릭했으면 카운트 초기화
                                    break
                                except:
                                    continue
                        
                        if not clicked:
                            # 한국어 버전 대응 (구형/변형)
                            more_button_kr = page.locator("input[value='결과 더보기']").first
                            if more_button_kr.is_visible():
                                print(f"[GoogleCrawler-{keyword}] '결과 더보기' 버튼 클릭 중...")
                                more_button_kr.click(timeout=3000)
                                page.wait_for_timeout(2000)
                                no_change_count = 0
                    except Exception:
                        pass


                    new_height = page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        no_change_count += 1
                        # 변화가 없어도 바로 종료하지 않고 몇 번 더 시도
                        if no_change_count >= max_no_change:
                             print(f"[GoogleCrawler-{keyword}] 페이지 끝에 도달 (높이 변화 없음)")
                             break
                        else:
                             print(f"[GoogleCrawler-{keyword}] 높이 변화 없음 ({no_change_count}/{max_no_change}), 대기 후 재시도...")
                             page.wait_for_timeout(2000)
                    else:
                        last_height = new_height
                        no_change_count = 0 # 높이 변화 있으면 카운트 초기화
                    
                    scroll_attempts += 1

                # 이미지 URL 추출
                image_elements = page.locator("img.YQ4gaf").all()
                print(f"[GoogleCrawler-{keyword}] {len(image_elements)}개의 이미지 요소 발견")

                count = 0
                for element in image_elements:
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
                        print(f"[GoogleCrawler-{keyword}] 이미지 처리 오류: {e}")

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
