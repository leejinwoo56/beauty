"""
run_batch.py

24/ 폴더의 Image1 ~ Image44 를 순회하며 analog_detect_video.py 실행.
mp4/jpg 모두 지원, 출력 파일 확장자는 입력과 동일하게 유지.

사용법:
    python run_batch.py
    python run_batch.py --input_dir 24 --output_base result_resolution --roi 50 70 1100 700
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   default="24")
    parser.add_argument("--output_base", default="test")
    parser.add_argument("--roi", nargs=4, type=int,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        default=[50, 70, 1100, 700])
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=44)
    parser.add_argument("--files", nargs="+", type=int,
                        help="특정 번호만 실행 (예: --files 15 17 32). 지정 시 start/end 무시")
    args = parser.parse_args()

    input_dir   = Path(args.input_dir)
    output_base = Path(args.output_base)
    roi_args    = [str(v) for v in args.roi]

    targets = args.files if args.files else list(range(args.start, args.end + 1))

    skipped = []
    done    = []

    for i in targets:
        # mp4 우선, 없으면 jpg 시도
        input_path = None
        for ext in ("mp4", "MP4", "jpg", "JPG", "jpeg", "JPEG"):
            candidate = input_dir / f"Image{i}.{ext}"
            if candidate.exists():
                input_path = candidate
                break

        if input_path is None:
            print(f"[skip] Image{i}: 파일 없음")
            skipped.append(i)
            continue

        ext        = input_path.suffix          # .mp4 or .jpg
        output_dir = output_base / f"output{i}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"output{i}{ext}"

        print(f"\n[{i}/{len(targets)}] {input_path} → {output_path}")

        cmd = [
            sys.executable, "analog_detect_video.py",
            "--input",  str(input_path),
            "--output", str(output_path),
            "--roi",    *roi_args,
        ]
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  [error] returncode={result.returncode}")
            skipped.append(i)
        else:
            done.append(i)

    print(f"\n[완료] 성공: {len(done)}개 / 스킵: {len(skipped)}개")
    if skipped:
        print(f"  스킵 목록: {skipped}")


if __name__ == "__main__":
    main()