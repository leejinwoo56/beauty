#!/bin/bash
set -e

# ── 1. 작업 공간 초기화 ────────────────────────────────────────────────────────
echo "[Sandbox] 작업 공간 초기화 중..."
rm -rf /workspace/code
cp -r /image/code /workspace/code
echo "[Sandbox] 초기화 완료."

# ── 2. 날짜 기반 체크포인트 출력 디렉토리 생성 ────────────────────────────────
# 오늘 날짜로 하위 폴더 생성 (예: /checkpoints/output/0420/)
DATE=$(date +%m%d)
mkdir -p /checkpoints/output/$DATE
mkdir -p /workspace/saved
mkdir -p /workspace/saved/infer_results

# ── 3. /workspace/ 안의 model_checkpoints 폴더 제거 ──────────────────────────
rm -rf /workspace/code/medsam_finetune_lora/model_checkpoints

# ── 4. 학습/추론 스크립트 경로 패치 ──────────────────────────────────────────
echo "[Sandbox] 스크립트 경로 패치 중..."
python3 - <<PYEOF
import os

date = "$DATE"

# ── train_lora_unimatch.py 패치 ──────────────────────────────────────────────
path = "/workspace/code/medsam_finetune_lora/train_lora_unimatch.py"
if os.path.exists(path):
    with open(path, "r") as f:
        content = f.read()
    replacements = [
        ('ANN_ROOT        = str(_ROOT / "labeling_data" / "propagated_masks")',
         'ANN_ROOT        = "/data/labeling_data/propagated_masks"'),
        ('FRAMES_ROOT     = str(_ROOT / "analog" / "analog_result_24_v1")',
         'FRAMES_ROOT     = "/data/analog_result_24_v1"'),
        ('OUTPUT_DIR      = str(_HERE / "model_checkpoints" / "train_lora_unimatch_aug_exclude5053_188")',
         f'OUTPUT_DIR      = "/checkpoints/output/{date}/train_lora_unimatch"'),
        ('SAM2_CKPT       = str(_ROOT / "MedSAM2" / "checkpoints" / "MedSAM2_US_Heart.pt")',
         'SAM2_CKPT       = "/image/code/MedSAM2/checkpoints/MedSAM2_US_Heart.pt"'),
        ('pin_memory=True', 'pin_memory=False'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("[Sandbox] train_lora_unimatch.py 패치 완료")

# ── train_lora.py 패치 ───────────────────────────────────────────────────────
path = "/workspace/code/medsam_finetune_lora/train_lora.py"
if os.path.exists(path):
    with open(path, "r") as f:
        content = f.read()
    replacements = [
        ('default="./MedSAM2/checkpoints/MedSAM2_latest.pt"',
         'default="/image/code/MedSAM2/checkpoints/MedSAM2_latest.pt"'),
        ('default="./lora_checkpoints"',
         f'default="/checkpoints/output/{date}/lora_checkpoints"'),
        ('pin_memory=True', 'pin_memory=False'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("[Sandbox] train_lora.py 패치 완료")

# ── infer_lora.py 패치 ───────────────────────────────────────────────────────
path = "/workspace/code/medsam_finetune_lora/infer_lora.py"
if os.path.exists(path):
    with open(path, "r") as f:
        content = f.read()
    replacements = [
        ('SAM2_CKPT   = str(_MEDSAM2 / "checkpoints" / "MedSAM2_US_Heart.pt")',
         'SAM2_CKPT   = "/image/code/MedSAM2/checkpoints/MedSAM2_US_Heart.pt"'),
        ('LORA_CKPT = str(_HERE / "model_checkpoints" / "train_lora_unimatch_aug_exclude5053_188" / "best_lora_unimatch.pt")',
         'LORA_CKPT = "/image/code/medsam_finetune_lora/model_checkpoints/train_lora_unimatch_aug_exclude5053_188/best_lora_unimatch.pt"'),
        ('FRAMES_ROOT = str(_ROOT / "analog" / "analog_result_24_v1")',
         'FRAMES_ROOT = "/data/analog_result_24_v1"'),
        ('OUTPUT_ROOT = str(_HERE / "output_lora" /"train_lora_unimatch_aug_exclude5053_188")',
         'OUTPUT_ROOT = "/workspace/saved/infer_results"'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("[Sandbox] infer_lora.py 패치 완료")
PYEOF

# ── 5. .pt 파일 감시 데몬 ─────────────────────────────────────────────────────
echo "[Sandbox] 체크포인트 차단 데몬 시작..."
(
  while true; do
    FILEPATH=$(inotifywait -q -r \
      -e close_write -e moved_to \
      --include '.*\.pt$' \
      --format '%w%f' \
      /workspace/ 2>/dev/null)
    if [ -n "$FILEPATH" ]; then
      rm -f "$FILEPATH"
      echo "[Sandbox] .pt 파일 저장 차단됨: $FILEPATH"
    fi
  done
) &

echo "[Sandbox] JupyterLab 시작 중..."
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --notebook-dir=/workspace \
  --config=/etc/jupyter/jupyter_server_config.py
