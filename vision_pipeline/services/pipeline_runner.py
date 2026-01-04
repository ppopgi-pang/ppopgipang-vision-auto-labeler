import yaml
import os
from pathlib import Path
from tqdm import tqdm
from domain.job import Job
from pipelines.crawl_pipeline import CrawlPipeline
from pipelines.filter_pipeline import FilterPipeline
from pipelines.detect_pipeline import DetectPipeline
from pipelines.verify_pipeline import VerifyPipeline
from modules.storage.image_store import ImageStore
from config import settings

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

        # 활성화된 단계 계산
        stages = []
        if self.stage_config.get("crawl", True):
            stages.append(("크롤링", "crawl"))
        if self.stage_config.get("filter", True):
            stages.append(("필터링", "filter"))
        if self.stage_config.get("detect", True):
            stages.append(("객체 탐지", "detect"))
        if self.stage_config.get("verify", True):
            stages.append(("검증", "verify"))

        # 전체 파이프라인 프로그레스바 (position=0: 최상위, leave=True: 완료 후에도 유지)
        with tqdm(total=len(stages), desc="전체 파이프라인 진행", unit="단계", position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            images = []
            filtered_images = []
            detection_results = []
            verification_results = []

            # 1. 크롤링
            if self.stage_config.get("crawl", True):
                # raw 디렉토리가 이미 존재하는지 확인
                raw_dir = os.path.join(settings.output_dir, "raw")
                if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
                    # 파일 수 확인
                    raw_files = []
                    for root, dirs, files in os.walk(raw_dir):
                        raw_files.extend([f for f in files if os.path.isfile(os.path.join(root, f))])

                    if raw_files:
                        print(f"\n[INFO] {raw_dir} 디렉토리가 존재합니다 ({len(raw_files)}개 파일).")
                        print("[INFO] 크롤링 단계를 스킵하고 기존 데이터로 파이프라인을 진행합니다.")
                        self._update_job_status(job, "crawling_skipped")
                        image_store = ImageStore()
                        images = image_store.load_raw()
                        pbar.update(1)
                    else:
                        pbar.set_description("단계 1/4: 크롤링")
                        self._update_job_status(job, "crawling")
                        images = self.crawl_pipeline.run(job)
                        pbar.update(1)
                else:
                    pbar.set_description("단계 1/4: 크롤링")
                    self._update_job_status(job, "crawling")
                    images = self.crawl_pipeline.run(job)
                    pbar.update(1)

                if not images:
                    self._update_job_status(job, "stopped_no_images")
                    print("\n이미지를 찾거나 크롤링하지 못했습니다. 중지합니다.")
                    return
            else:
                self._update_job_status(job, "crawling_skipped")
                images = []

            # 2. 필터링
            if self.stage_config.get("filter", True):
                pbar.set_description("단계 2/4: 필터링")
                self._update_job_status(job, "filtering")
                # 분류기가 대상 클래스에 대해 확인하도록 보장
                for img in images:
                    if not img.keyword:
                        img.keyword = job.target_class

                filtered_images = self.filter_pipeline.run(images)
                pbar.update(1)

                if not filtered_images:
                    self._update_job_status(job, "stopped_no_filtered")
                    print("\n필터링 후 남은 이미지가 없습니다. 중지합니다.")
                    return
            else:
                self._update_job_status(job, "filtering_skipped")
                filtered_images = images

            # 3. 객체 탐지
            if self.stage_config.get("detect", True):
                pbar.set_description("단계 3/4: 객체 탐지")
                self._update_job_status(job, "detecting")
                detection_results = self.detect_pipeline.run(filtered_images)
                pbar.update(1)

                if not detection_results:
                    self._update_job_status(job, "stopped_no_detections")
                    print("\n탐지된 객체가 없습니다. 중지합니다.")
                    return
            else:
                self._update_job_status(job, "detecting_skipped")
                detection_results = []

            # 4. 검증
            if self.stage_config.get("verify", True):
                pbar.set_description("단계 4/4: 검증")
                self._update_job_status(job, "verifying")
                verification_results = self.verify_pipeline.run(detection_results)
                pbar.update(1)
            else:
                self._update_job_status(job, "verifying_skipped")
                verification_results = []

        print(f"\n[{job.job_id}] 파이프라인 완료.")

        # 최종 요약
        verified_count = len([r for r in verification_results if r.verified])
        print(f"최종 결과: 초기 {len(images)}개 이미지 중 {verified_count}개 검증된 레이블.")
        self._update_job_status(job, "completed")
