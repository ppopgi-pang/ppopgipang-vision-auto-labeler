# ppopgipang-vision-auto-labeler
이미지 크롤링부터 자동 필터링, 객체 탐지, LLM 기반 라벨 검증까지 연결하는 비전 데이터 파이프라인

## 설치

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Playwright 브라우저 설치
```bash
playwright install
```

### 3. 환경 변수 설정
`.env` 파일을 프로젝트 루트에 생성하고 다음 내용을 추가:
```bash
OPENAI_API_KEY=your_api_key_here
```

## 실행 방법
```bash
python scripts/run_pipeline.py --keywords "하츠네 미쿠 피규어" "하츠네 미쿠 인형" --target "miku"
```