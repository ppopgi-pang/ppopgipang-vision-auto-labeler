import imagehash
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from modules.filter.base import FilterStep
from domain.image import ImageItem
from config import settings
import numpy as np


def _compute_hash_worker(img_path: str, hash_size: int) -> tuple[imagehash.ImageHash | None, str | None]:
    """
    ProcessPoolExecutor를 위한 모듈 레벨 함수 (picklable) - CPU fallback용
    """
    if not img_path or not Path(img_path).exists():
        return None, f"[Deduplicator] specific path not found: {img_path}"

    try:
        with Image.open(img_path) as pil_img:
            current_hash = imagehash.phash(pil_img, hash_size=hash_size)
        return current_hash, None
    except Exception as e:
        return None, f"[Deduplicator] Error processing {img_path}: {e}"


class Deduplicator(FilterStep):
    def __init__(self, config: dict):
        self.config = config
        self.hash_size = config.get("hash_size", 8)
        self.threshold = config.get("threshold", 5)
        self.seen_hashes = []  # GPU 모드: tensor 해시, CPU 모드: (image_item, hash_obj) 튜플
        self.use_gpu = config.get("use_gpu", True)  # 기본값 True로 변경 (대용량 처리)
        self.device = "cpu"

        # GPU 사용 시 장치 설정
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[Deduplicator] GPU 모드 활성화 (device: {self.device})")
                print(f"[Deduplicator] 대용량 이미지 처리에 최적화됨 (배치 GPU 해싱)")
            else:
                print("[Deduplicator] CUDA를 사용할 수 없어 CPU 모드로 fallback합니다.")
                self.use_gpu = False

        # 이미지 전처리 파이프라인 (GPU 전송 전)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32), antialias=True),
            transforms.ToTensor(),
        ])

    def compute_perceptual_hash_gpu(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        GPU에서 배치 이미지에 대해 perceptual hash 계산
        Args:
            image_tensors: [batch_size, 1, 32, 32] grayscale 이미지
        Returns:
            [batch_size, hash_size*hash_size] 비트 해시 (0 또는 1)
        """
        batch_size = image_tensors.shape[0]

        # 2D DCT 근사 (PyTorch에는 내장 DCT가 없으므로 코사인 변환 행렬 사용)
        # 실제 pHash는 DCT를 사용하지만, 여기서는 평균 풀링으로 근사
        # 더 정확한 구현을 원하면 torch-dct 라이브러리 사용 가능

        # 간단한 방식: 8x8로 평균 풀링 (저주파 특징 추출)
        pooled = F.adaptive_avg_pool2d(image_tensors, (self.hash_size, self.hash_size))
        pooled = pooled.view(batch_size, -1)  # [batch_size, hash_size*hash_size]

        # 중앙값 계산
        median = pooled.median(dim=1, keepdim=True)[0]

        # 중앙값보다 크면 1, 작으면 0
        hash_bits = (pooled > median).float()

        return hash_bits

    def compute_hamming_distance_gpu(self, hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
        """
        GPU에서 해밍 거리 계산
        Args:
            hash1: [N, hash_dim] 또는 [hash_dim]
            hash2: [M, hash_dim] 또는 [hash_dim]
        Returns:
            [N, M] 또는 스칼라 해밍 거리
        """
        # XOR 연산 후 합계 (다른 비트 개수)
        if hash1.dim() == 1:
            hash1 = hash1.unsqueeze(0)
        if hash2.dim() == 1:
            hash2 = hash2.unsqueeze(0)

        # Broadcasting을 사용한 해밍 거리 계산
        # [N, 1, hash_dim] XOR [1, M, hash_dim] -> [N, M, hash_dim]
        xor = (hash1.unsqueeze(1) != hash2.unsqueeze(0)).float()
        distances = xor.sum(dim=2)  # [N, M]

        return distances

    def run_gpu(self, images: list[ImageItem]) -> list[ImageItem]:
        """GPU 기반 중복 제거 (대용량 처리 최적화)"""
        unique_images = []
        duplicates = 0

        print(f"[Deduplicator] GPU 모드로 {len(images)} 이미지 처리 중...")
        total = len(images)
        if total == 0:
            return unique_images

        # 배치 크기 설정 (GPU 메모리에 따라 조정)
        batch_size = self.config.get("batch_size", 128)

        # GPU 메모리 관리를 위한 청크 크기 (이 개수마다 GPU에서 CPU로 해시 이동)
        gpu_hash_limit = self.config.get("gpu_hash_limit", 10000)

        # 전체 해시를 저장할 리스트
        seen_hash_tensors_gpu = []  # GPU에 있는 최근 해시들
        seen_hash_tensors_cpu = []  # CPU로 이동된 해시들 (오래된 것들)
        seen_items = []  # 대응하는 ImageItem

        # 배치 단위로 처리
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = images[batch_start:batch_end]

            # 배치 이미지 로드 및 전처리
            batch_tensors = []
            valid_items = []

            for img_item in batch_items:
                if not img_item.path or not Path(img_item.path).exists():
                    continue

                try:
                    with Image.open(img_item.path) as pil_img:
                        tensor = self.transform(pil_img)  # [1, 32, 32]
                        batch_tensors.append(tensor)
                        valid_items.append(img_item)
                except Exception as e:
                    print(f"\n[Deduplicator] 이미지 로드 오류 {img_item.path}: {e}")
                    continue

            if not batch_tensors:
                continue

            # 배치를 GPU로 전송
            batch_tensor = torch.stack(batch_tensors).to(self.device)  # [batch, 1, 32, 32]

            # GPU에서 해시 계산
            with torch.no_grad():
                current_hashes = self.compute_perceptual_hash_gpu(batch_tensor)  # [batch, hash_dim]

            # 각 이미지에 대해 중복 검사
            for idx, (img_item, img_hash) in enumerate(zip(valid_items, current_hashes)):
                is_duplicate = False

                # CPU에 저장된 해시와 비교 (있는 경우)
                if seen_hash_tensors_cpu:
                    cpu_hashes = torch.stack(seen_hash_tensors_cpu).to(self.device)
                    distances_cpu = self.compute_hamming_distance_gpu(img_hash, cpu_hashes)
                    min_distance_cpu = distances_cpu.min().item()

                    if min_distance_cpu <= self.threshold:
                        is_duplicate = True
                        duplicates += 1

                # GPU에 있는 최근 해시와 비교 (중복이 아닌 경우만)
                if not is_duplicate and seen_hash_tensors_gpu:
                    gpu_hashes = torch.stack(seen_hash_tensors_gpu)
                    distances_gpu = self.compute_hamming_distance_gpu(img_hash, gpu_hashes)
                    min_distance_gpu = distances_gpu.min().item()

                    if min_distance_gpu <= self.threshold:
                        is_duplicate = True
                        duplicates += 1

                if not is_duplicate:
                    seen_hash_tensors_gpu.append(img_hash)
                    seen_items.append(img_item)
                    unique_images.append(img_item)

                    # GPU 해시가 너무 많이 쌓이면 CPU로 이동
                    if len(seen_hash_tensors_gpu) >= gpu_hash_limit:
                        print(f"\n[Deduplicator] GPU 메모리 관리: {len(seen_hash_tensors_gpu)}개 해시를 CPU로 이동 중...")
                        # GPU 해시를 CPU로 이동
                        for h in seen_hash_tensors_gpu:
                            seen_hash_tensors_cpu.append(h.cpu())
                        seen_hash_tensors_gpu = []
                        # GPU 메모리 정리
                        torch.cuda.empty_cache()

            # 진행상황 출력
            processed = batch_end
            gpu_count = len(seen_hash_tensors_gpu)
            cpu_count = len(seen_hash_tensors_cpu)
            print(f"[Deduplicator] GPU 처리: {processed}/{total} (유지: {len(unique_images)}, 중복: {duplicates}, GPU해시: {gpu_count}, CPU해시: {cpu_count})...", end="\r", flush=True)

        # GPU 메모리 정리
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print()
        print(f"[Deduplicator] GPU 처리 완료. {duplicates}개 중복 제거. {len(unique_images)}개 유지.")
        return unique_images

    def run_cpu(self, images: list[ImageItem]) -> list[ImageItem]:
        """CPU 기반 중복 제거 (기존 방식)"""
        unique_images = []
        duplicates = 0

        print(f"[Deduplicator] CPU 모드로 {len(images)} 이미지 처리 중 (ProcessPoolExecutor)...")
        total = len(images)
        if total == 0:
            return unique_images

        max_workers = max(1, min(int(getattr(settings, "max_workers", 64)) // 4, 16))
        hash_results: list[tuple[ImageItem, imagehash.ImageHash | None, str | None]] = [None] * total

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_hash_worker, img_item.path, self.hash_size): idx
                for idx, img_item in enumerate(images)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                img_item = images[idx]
                try:
                    current_hash, error = future.result()
                except Exception as e:
                    current_hash, error = None, f"[Deduplicator] Error processing {img_item.path}: {e}"

                hash_results[idx] = (img_item, current_hash, error)
                completed += 1
                print(f"[Deduplicator] CPU 처리: {completed}/{total}...", end="\r", flush=True)

        print()
        for img_item, current_hash, error in hash_results:
            if error:
                print(error)
                continue

            is_duplicate = False
            for seen_item, seen_hash in self.seen_hashes:
                if current_hash - seen_hash <= self.threshold:
                    is_duplicate = True
                    duplicates += 1
                    break

            if not is_duplicate:
                self.seen_hashes.append((img_item, current_hash))
                unique_images.append(img_item)

        print(f"[Deduplicator] CPU 처리 완료. {duplicates}개 중복 제거. {len(unique_images)}개 유지.")
        return unique_images

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        """
        중복 제거 실행
        GPU 사용 가능 시 GPU 모드로, 불가능 시 CPU 모드로 처리
        """
        if self.use_gpu and self.device == "cuda":
            return self.run_gpu(images)
        else:
            return self.run_cpu(images)
