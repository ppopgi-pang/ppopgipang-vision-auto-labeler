import yaml
from pathlib import Path
from domain.job import Job
from pipelines.crawl_pipeline import CrawlPipeline
from pipelines.filter_pipeline import FilterPipeline
from pipelines.detect_pipeline import DetectPipeline
from pipelines.verify_pipeline import VerifyPipeline

class PipelineRunner:
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"경고: 설정 파일 {config_file}을 찾을 수 없습니다. 기본값을 사용합니다.")
            self.config = {}
            
        self.stage_config = self.config.get("stages", {})
        
        # 파이프라인 초기화
        # (지연 로딩이 더 나을 수 있지만, 현재로서는 간단한 초기화로 충분)
        self.crawl_pipeline = CrawlPipeline()
        self.filter_pipeline = FilterPipeline()
        self.detect_pipeline = DetectPipeline()
        self.verify_pipeline = VerifyPipeline()

    def _update_job_status(self, job: Job, status: str):
        job.status = status
        print(f"[Job] Status -> {status}")

    def run(self, job: Job):
        self._update_job_status(job, "running")
        print(f"[{job.job_id}] {job.keywords}에 대한 Pipeline Runner 시작...")

        # 1. 크롤링
        if self.stage_config.get("crawl", True):
            print("\n>>> 단계 1: 크롤링")
            self._update_job_status(job, "crawling")
            images = self.crawl_pipeline.run(job)
        else:
            print("\n>>> 단계 1: 크롤링 (건너뜀)")
            self._update_job_status(job, "crawling_skipped")
            # 건너뛸 경우 이전 단계에서 로드해야 하는가?
            # 현재로서는 건너뛰면 기존 데이터를 로드하지 않는 한 진행할 수 없다고 가정.
            # 하지만 일반적으로 이 MVP에서는 모두 실행하거나 아무것도 실행하지 않는다고 가정.
            images = []

        if not images and self.stage_config.get("crawl", True):
            self._update_job_status(job, "stopped_no_images")
            print("이미지를 찾거나 크롤링하지 못했습니다. 중지합니다.")
            return

        # 2. 필터링
        if self.stage_config.get("filter", True):
            print("\n>>> 단계 2: 필터링")
            self._update_job_status(job, "filtering")
            # 분류기가 대상 클래스에 대해 확인하도록 보장
            # 분류기 설정에 대상 클래스를 동적으로 주입해야 할 수도 있음?
            # 또는 누락된 경우 images.keyword를 job.target_class로 업데이트
            for img in images:
                if not img.keyword:
                    img.keyword = job.target_class

            filtered_images = self.filter_pipeline.run(images)
        else:
             print("\n>>> 단계 2: 필터링 (건너뜀)")
             self._update_job_status(job, "filtering_skipped")
             filtered_images = images  # 건너뛸 경우 통과

        if not filtered_images:
            self._update_job_status(job, "stopped_no_filtered")
            print("필터링 후 남은 이미지가 없습니다. 중지합니다.")
            return

        # 3. 객체 탐지
        if self.stage_config.get("detect", True):
            print("\n>>> 단계 3: 객체 탐지")
            self._update_job_status(job, "detecting")
            detection_results = self.detect_pipeline.run(filtered_images)
        else:
            print("\n>>> 단계 3: 객체 탐지 (건너뜀)")
            self._update_job_status(job, "detecting_skipped")
            detection_results = []

        if not detection_results:
             self._update_job_status(job, "stopped_no_detections")
             print("탐지된 객체가 없습니다. 중지합니다.")
             return

        # 4. 검증
        if self.stage_config.get("verify", True):
             print("\n>>> 단계 4: 검증")
             self._update_job_status(job, "verifying")
             verification_results = self.verify_pipeline.run(detection_results)
        else:
             print("\n>>> 단계 4: 검증 (건너뜀)")
             self._update_job_status(job, "verifying_skipped")
             verification_results = []

        print(f"\n[{job.job_id}] 파이프라인 완료.")

        # 최종 요약
        verified_count = len([r for r in verification_results if r.verified])
        print(f"최종 결과: 초기 {len(images)}개 이미지 중 {verified_count}개 검증된 레이블.")
        self._update_job_status(job, "completed")
