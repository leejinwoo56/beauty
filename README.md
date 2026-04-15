# SMAS Layer Detection in Aesthetic Ultrasound

피부 미용 초음파 영상에서 **SMAS 층을 탐지**하기 위한 알고리즘 개발 프로젝트입니다.

---


## 접근 순서

| 단계 | 방법 | 결과 |
|------|------|------|
| 1 | SAM — 범용 segmentation | 층 구조 추적 불가 |
| 2 | MedSAM + sliding window box prompt | 연속성 유지 실패 |
| 3 | SAMUS / AutoSAMUS — 초음파 특화 모델 | 병변 분할엔 강하나 근막층 탐지 부적합 |
| 4 | **Analog detection** — rule-based 구조 추적 | 근막층 초기 탐지, 이미지에 따른 편차 존재 |
| 5 | MedSAM2 LoRA fine-tuning (UniMatch V2 semi-supervised) | **최종 채택, 영상 단위 적용** |

---

## 모델 구조 (MedSAM2 + LoRA)

### 베이스: MedSAM2
SAM2를 의료 초음파 도메인에 fine-tuning한 모델.
```
Image Encoder (Hiera backbone)
    └─ MultiScaleAttention × N
Prompt Encoder
Mask Decoder
```

### LoRA 적용
Hiera backbone의 모든 `MultiScaleAttention.qkv`에 삽입. Q·V에만 delta 적용, K는 frozen 유지.
```
Q_out = Q_orig + lora_q_B( lora_q_A(x) ) × (alpha/r)
V_out = V_orig + lora_v_B( lora_v_A(x) ) × (alpha/r)
K_out = K_orig  (변경 없음)

rank r=4, alpha=4  |  학습 대상: LoRA 가중치 + Mask Decoder (~전체의 수%)
```

### SAMed 기본 프롬프트
```
default_prompt_embedding: nn.Parameter [1, 2, 256]
→ Mask Decoder에 task-specific embedding으로 주입 (별도 box prompt 없이 동작)
```

### 학습: UniMatch V2 Semi-Supervised
```
L_total = (L_sup + L_unsup) / 2

L_sup  : labeled frame → BCE + Dice (GT 마스크)
L_unsup: EMA teacher pseudo-label (confidence > 0.95) + Complementary Dropout
         → confidence-masked BCE × 2

EMA decay = 0.996
```

---

## 웹 서비스

FastAPI(백엔드) + React(프론트엔드)로 구성된 데모 웹 애플리케이션을 AWS EC2에 배포했습니다.

- 영상 업로드 → SMAS 탐지 추론 → 결과 영상 반환
- MedSAM2 LoRA 모델 기반 추론 (`webpage/backend/inference.py`)

---

## 결과

<!-- ==================== RESULT VIDEO HERE ==================== -->





<img width="600" height="360" alt="result-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/9438c7ce-87ef-4fc0-bbda-4b8c42ac7223" />








<!-- ========================================================== -->

---

## 주요 파일

```
analog/
  analog_detect.py          # rule-based SMAS 탐지
  analog_detect_video.py    # 영상 단위 적용

medsam_finetune_lora/
  train_lora_unimatch.py    # MedSAM2 LoRA + UniMatch V2 semi-supervised 학습
  infer_lora.py             # 이미지 추론
  infer_lora_video.py       # 영상 추론

webpage/
  backend/inference.py      # FastAPI 추론 서버
  frontend/                 # React 프론트엔드
```

---

## Tech Stack

Python · PyTorch · OpenCV · FastAPI · React · AWS EC2
