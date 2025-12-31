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
            browser = p.chromium.launch(headless=True)  # 프로덕션 환경에서는 True로 설정
            page = browser.new_page()

            for keyword in keywords:
                try:
                    # Google 이미지 검색으로 이동
                    page.goto("https://www.google.com/imghp?hl=en&ogbl")

                    # 쿠키 동의 버튼이 있으면 클릭 (EU 지역용이지만 일반적인 좋은 관행)
                    try:
                        page.locator("button:has-text('Accept all')").click(timeout=2000)
                    except:
                        pass

                    # 검색
                    search_box = page.locator("textarea[name='q']")
                    search_box.fill(keyword)
                    search_box.press("Enter")

                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2000)  # 결과 로딩을 위한 명시적 대기

                    # 더 많은 이미지를 로드하기 위해 스크롤
                    last_height = page.evaluate("document.body.scrollHeight")
                    scroll_attempts = 0
                    max_scrolls = 100  # 안전 제한

                    while scroll_attempts < max_scrolls:
                        page.keyboard.press("End")
                        page.wait_for_timeout(1000)

                        # "더 보기" 버튼을 확인하고 표시되면 클릭
                        # 버튼의 일반적인 선택자: .mye4qd
                        try:
                            # 여러 가능한 선택자 또는 텍스트 시도
                            more_button = page.locator(".mye4qd, input[value='Show more results'], input[type='button'][value='Show more results']")
                            if more_button.is_visible():
                                print("[GoogleCrawler] Clicking 'Show more results' button...")
                                more_button.click()
                                page.wait_for_timeout(2000)
                        except Exception:
                            pass

                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            # 느린 네트워크인지 확인하기 위해 조금 더 대기
                            page.wait_for_timeout(1000)
                            new_height = page.evaluate("document.body.scrollHeight")
                            if new_height == last_height:
                                print("[GoogleCrawler] Reached end of page or no new content.")
                                break

                        last_height = new_height
                        scroll_attempts += 1

                    # 이미지 URL 추출
                    # Google 구조: div.isv-r이 결과를 포함. 내부에 img 태그.
                    # img.rg_i는 썸네일의 일반적인 클래스.

                    # 브라우저 검사에 기반한 업데이트된 선택자 (2024-05)
                    image_elements = page.locator("img.YQ4gaf").all()
                    print(f"DEBUG: Found {len(image_elements)} potential 'img.YQ4gaf' elements.")

                    count = 0
                    for element in image_elements:
                        # 사용 가능한 모든 이미지를 가져오기 위해 제한 제거
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

            browser.close()
            
        return image_items
