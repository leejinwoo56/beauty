# SMAS Layer Detection in Aesthetic Ultrasound

피부 미용 초음파 영상에서 **SMAS 층을 탐지**하기 위한 알고리즘을 개발한 프로젝트입니다.  
이 프로젝트의 핵심은 단순히 하나의 모델을 적용하는 것이 아니라, **문제가 잘 풀리지 않을 때 다양한 접근을 빠르게 시도하고, 실패 원인을 분석해 최종적으로 동작하는 해결책까지 도달한 과정**에 있습니다.

---

## Project Overview

초기 목표는 초음파 이미지에서 특정 근막층인 **SMAS(Superficial Musculoaponeurotic System)** 를 안정적으로 찾아내는 것이었습니다.  
처음에는 최신 segmentation foundation model을 활용하면 빠르게 해결할 수 있을 것이라 판단해 다음과 같은 방법들을 순차적으로 검토했습니다.

- **SAM**: 범용 segmentation foundation model
- **MedSAM**: 의료 영상 특화 SAM
- **SAMUS / AutoSAMUS**: 초음파 도메인 특화 모델
- **Brightness / rule-based method**: 영상의 밝기 구조와 층 경계를 직접 이용하는 비AI 방식
- **Analog detection pipeline**: 최종적으로 수렴한 구조 기반 탐지 방식

---

## Why This Project Matters

이 프로젝트는 의료 영상 도메인의 과제이지만, 제가 보여주고 싶은 핵심 역량은 도메인 자체보다 **문제 해결 방식**입니다.

이 과정을 통해 다음을 경험했습니다.

- pretrained 모델이 실제 문제와 맞지 않을 때의 한계 파악
- 논문/오픈소스 기반 모델을 직접 적용하고 비교 실험 수행
- 실패 원인을 분석하며 문제를 다시 정의
- AI 접근에 집착하지 않고, 더 적합한 비AI 방법으로 방향 전환
- 이미지 수준을 넘어 영상 단위까지 확장 가능한 결과 도출

즉, **정답이 불명확한 문제에서 가설을 세우고, 검증하고, 실패하면 빠르게 다른 방향으로 전환하는 능력**을 보여주는 프로젝트입니다.

---

## Approach & Troubleshooting

### 1) SAM 적용
처음에는 범용 **SAM**을 사용해 초음파 이미지 전체에서 마스크를 자동 생성했습니다.  
이 단계에서는 이미지 내 여러 구조가 분리되었지만, **SMAS처럼 길게 이어지는 특정 층 구조를 안정적으로 찾는 문제**에는 적합하지 않았습니다.

### 2) MedSAM 적용
이후 **MedSAM**을 적용해 box prompt 기반 segmentation을 수행했습니다.  
또한 layer 후보 구간을 먼저 찾은 뒤, 슬라이딩 window 방식으로 여러 box를 생성해 자동으로 층을 분할하려는 시도도 진행했습니다.

하지만 이 과정에서 다음과 같은 한계를 확인했습니다.

- 원하는 층이 아닌 다른 고밝기 구조에 반응하는 경우가 많음
- 이미지 전체에서 **연속적인 layer 구조**를 안정적으로 유지하지 못함
- 결과를 보정하기 위한 후처리 의존도가 점점 커짐

### 3) SAMUS / AutoSAMUS 적용
초음파 도메인에 더 가까운 모델인 **SAMUS / AutoSAMUS**도 적용했습니다.  
도메인 특화 모델이라 더 나은 결과를 기대했지만, 이 역시 **병변 분할**에는 강점이 있어도,  
**전체 이미지에서 특정 근막층을 구조적으로 탐지하는 과제**에는 충분하지 않았습니다.

### 4) Rule-based 접근으로 전환
이후에는 문제를 다시 정의했습니다.

이 과제는 “객체를 잘 분할하는 문제”라기보다,  
**초음파 영상에서 특정 깊이와 밝기 패턴을 가지는 층 구조를 일관되게 추적하는 문제**라고 판단했습니다.

그래서 딥러닝 모델 대신 다음과 같은 비AI 방식을 시도했습니다.

- brightness profile 분석
- 경계선(seed) 탐지
- peak line propagation
- 상대 깊이와 연속성 기반 layer 추적
- 지방층/근막층 구분을 위한 구조적 규칙 적용

처음 시도한 `bmode_brightness_classifier`는 완전한 해결책은 아니었지만,  
이 과정을 통해 **모델보다 영상 구조 자체를 직접 해석하는 방향**이 더 적합하다는 확신을 얻었습니다.

### 5) Final: Analog Detection
최종적으로는 `analog_detect.py` 기반의 **structure-aware analog detection pipeline**으로 수렴했습니다.

이 방식은 다음 요소를 결합합니다.

- 전처리(CLAHE, blur)
- 표피/진피 경계 탐지
- brightness peak 검출
- 핵심 layer line 선택
- layer 간 상대 위치와 연속성 기반 추적
- SMAS 내부 밝은 구조 강조

그 결과, 모델에 의존하지 않고도 **실제로 활용 가능한 수준의 SMAS 탐지 결과**를 얻을 수 있었고,  
마지막에는 이를 **영상 전체 프레임**에 적용하는 단계까지 확장했습니다.

---

## Result



<!-- ==================== RESULT VIDEO HERE ==================== -->
https://github.com/user-attachments/assets/0a8385ff-eab6-4ba5-8db1-1d95b9d0f74a
<!-- ========================================================== -->

최종적으로 이 프로젝트를 통해 다음을 달성했습니다.

- 여러 segmentation foundation model의 실제 적용 가능성 검증
- 초음파 영상 구조에 맞는 문제 재정의
- rule-based / non-AI 방식으로의 성공적인 전환
- 단일 이미지가 아닌 **영상 단위 적용 가능 파이프라인** 구현

---

## Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **SAM / MedSAM / SAMUS 계열 모델 실험**
- **Image processing / signal-inspired rule-based detection**

---

## Key Takeaways

이 프로젝트에서 가장 크게 배운 점은,  
**좋은 문제 해결은 특정 모델을 고집하는 것이 아니라, 문제의 본질에 맞는 방법을 끝까지 찾아가는 것**이라는 점이었습니다.

처음에는 최신 모델을 적용하는 방식으로 시작했지만,  
실험을 반복할수록 이 과제의 핵심은 “분할”이 아니라 “구조 추적”이라는 점을 깨달았습니다.  
그리고 그에 맞춰 접근 방식을 과감하게 바꾸었을 때, 비로소 실제로 동작하는 결과를 만들 수 있었습니다.

---

## Files

- `basic_sam.py` : 기본 SAM 자동 마스크 생성 실험
- `medsam_box_interactive.py` : MedSAM box prompt 기반 인터랙티브 실험
- `medsam_layer_segmentation.py` : layer 구간 추정 + MedSAM 자동 segmentation
- `autosamus.py` : 초음파 도메인 특화 SAMUS/AutoSAMUS 실험
- `bmode_brightness_classifier.py` : brightness 기반 rule-based 분류 시도
- `analog_detect.py` : 최종 구조 기반 SMAS 탐지 파이프라인
- `analog_detect_video.py` : 최종 방법의 영상 적용 버전

---

## Relevance

비록 이 프로젝트는 의료 초음파 도메인에서 수행되었지만,  
제가 이 프로젝트를 통해 보여주고 싶은 역량은 **문제 해결력, 실험 설계 능력, 실패 분석 능력, 방향 전환 능력**입니다.

이는 추천 시스템, CTR 예측, 개인화 광고와 같은 머신러닝 문제에서도 동일하게 중요한 역량이라고 생각합니다.

