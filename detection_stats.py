"""
detection_stats.py

result_resolution/ 을 순회하며 각 영상의 smas_thick 검출 성공률 출력.
성공률 = area > 0 인 frame 수 / 전체 frame 수

사용법:
    python detection_stats.py
    python detection_stats.py --data_root result_resolution
"""

import argparse
import json
from pathlib import Path


def calc_stats(result_dir: Path) -> tuple[int, int]:
    """(검출 성공 frame 수, 전체 frame 수) 반환."""
    mask_dir = result_dir / "masks"
    if not mask_dir.exists():
        return 0, 0

    total = 0
    detected = 0
    for json_path in sorted(mask_dir.glob("*.json")):
        with open(json_path) as f:
            ann = json.load(f)
        area = ann["annotations"][0]["area"]
        total += 1
        if area > 0:
            detected += 1

    return detected, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="result_resolution")
    args = parser.parse_args()

    base = Path(args.data_root)
    result_dirs = sorted(base.glob("output*"),
                         key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0))

    print(f"{'폴더':<12} {'성공':>6} {'전체':>6} {'성공률':>8}")
    print("-" * 38)

    total_det = 0
    total_all = 0

    for result_dir in result_dirs:
        if not result_dir.is_dir():
            continue
        detected, total = calc_stats(result_dir)
        if total == 0:
            continue

        rate = detected / total * 100
        total_det += detected
        total_all += total

        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"{result_dir.name:<12} {detected:>6} {total:>6} {rate:>7.1f}%  {bar}")

    if total_all > 0:
        overall = total_det / total_all * 100
        print("-" * 38)
        print(f"{'전체 합계':<12} {total_det:>6} {total_all:>6} {overall:>7.1f}%")


if __name__ == "__main__":
    main()