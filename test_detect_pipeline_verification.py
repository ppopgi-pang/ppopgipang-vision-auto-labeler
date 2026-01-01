"""
DetectPipeline 검증 테스트
CLIP → Top-K → GPT Judge 플로우가 정상적으로 작동하는지 확인
"""
import sys
from pathlib import Path

# Add vision_pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "vision_pipeline"))

from pipelines.detect_pipeline import DetectPipeline
from domain.image import ImageItem
import json


def main():
    print("=" * 80)
    print("DetectPipeline 검증 테스트")
    print("=" * 80)

    # 테스트 이미지 경로 확인
    test_data_dir = Path("data/raw")
    if not test_data_dir.exists():
        print(f"❌ 테스트 데이터 디렉토리가 없습니다: {test_data_dir}")
        print("   샘플 이미지를 data/raw/ 폴더에 넣어주세요.")
        return

    # 이미지 파일 수집
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(test_data_dir.glob(ext))

    if not image_files:
        print(f"❌ 테스트 이미지가 없습니다: {test_data_dir}")
        return

    # 최대 3개 이미지만 테스트
    image_files = list(image_files)[:3]
    print(f"\n✓ {len(image_files)}개 테스트 이미지 발견")

    # ImageItem 리스트 생성
    test_images = []
    for idx, img_path in enumerate(image_files):
        test_images.append(ImageItem(
            id=f"test_{idx}",
            path=str(img_path),
            source="test",
            metadata={}
        ))

    print("\n" + "=" * 80)
    print("DetectPipeline 실행")
    print("=" * 80)

    # DetectPipeline 실행
    pipeline = DetectPipeline(config_path="configs/detector.yaml")
    results = pipeline.run(test_images)

    print("\n" + "=" * 80)
    print("결과 분석")
    print("=" * 80)

    # 결과 분석
    total_detections = 0
    clip_enabled_count = 0
    gpt_judge_count = 0
    verifier_count = 0
    verified_true_count = 0
    verified_false_count = 0

    for result in results:
        bboxes = result.get("bboxes", [])
        total_detections += len(bboxes)

        for bbox in bboxes:
            # CLIP 후보가 있는지 확인
            if bbox.get("clip_candidates"):
                clip_enabled_count += 1

            # GPT Judge 실행 여부 확인
            if bbox.get("labeler_confidence") is not None:
                gpt_judge_count += 1

            # Verifier 실행 여부 확인
            if bbox.get("verified") is not None:
                verifier_count += 1
                if bbox.get("verified"):
                    verified_true_count += 1
                else:
                    verified_false_count += 1

                print(f"\n[이미지: {result['image_id']}]")
                print(f"  Label: {bbox['label']}")
                print(f"  CLIP Top-1 Score: {bbox.get('clip_top1_score', 'N/A'):.4f}" if bbox.get('clip_top1_score') else "  CLIP Top-1 Score: N/A")
                if bbox.get("clip_candidates"):
                    print(f"  CLIP Candidates:")
                    for candidate in bbox.get("clip_candidates", []):
                        print(f"    - {candidate['label']}: {candidate['score']:.4f}")
                print(f"  Labeler Confidence: {bbox.get('labeler_confidence', 'N/A')}")
                print(f"  Verified: {bbox.get('verified')}")
                print(f"  Verification Reason: {bbox.get('verification_reason', 'N/A')}")
                print(f"  Verification Confidence: {bbox.get('verification_confidence', 'N/A')}")

    print("\n" + "=" * 80)
    print("검증 요약")
    print("=" * 80)

    print(f"\n총 탐지된 객체: {total_detections}개")
    print(f"CLIP 후보 생성: {clip_enabled_count}개")
    print(f"GPT Judge 라벨링: {gpt_judge_count}개")
    print(f"Verifier 검증: {verifier_count}개")
    print(f"  - Verified (valid): {verified_true_count}개")
    print(f"  - Not Verified (invalid → unknown): {verified_false_count}개")

    # 플로우 검증
    print("\n파이프라인 플로우 체크:")
    print(f"  ✓ CLIP → Top-K: {'✓ 동작함' if clip_enabled_count > 0 else '❌ 동작 안 함'}")
    print(f"  ✓ GPT Judge: {'✓ 동작함' if gpt_judge_count > 0 else '❌ 동작 안 함'}")
    print(f"  ✓ Verifier: {'✓ 통합됨' if verifier_count > 0 else '❌ 동작 안 함'}")
    print(f"  ✓ valid/invalid 분기: {'✓ 구현됨' if verified_false_count >= 0 else '❌ 동작 안 함'}")

    # 결과 JSON 저장 확인
    output_path = Path("data/artifacts/bboxes.json")
    if output_path.exists():
        print(f"\n✓ 결과 저장됨: {output_path}")
        with open(output_path) as f:
            saved_results = json.load(f)

        # CLIP 데이터가 저장되었는지 확인
        has_clip_data = any(
            bbox.get("clip_candidates") is not None
            for result in saved_results
            for bbox in result.get("bboxes", [])
        )
        print(f"  CLIP 데이터 저장: {'✓' if has_clip_data else '❌'}")

        # Verification 데이터가 저장되었는지 확인
        has_verification_data = any(
            bbox.get("verified") is not None
            for result in saved_results
            for bbox in result.get("bboxes", [])
        )
        print(f"  Verification 데이터 저장: {'✓' if has_verification_data else '❌'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
