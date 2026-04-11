# 프로젝트 진행상황

## 프로젝트 개요
MedSAM2 기반 SMAS층 감지 웹페이지
- 영상 업로드 → SMAS 탐지 → 원본/결과 영상 나란히 표시
- 목표: GitHub CI/CD + AWS 배포 실습

## 기술 스택
- Backend: Python + FastAPI
- Frontend: React (Vite)
- 배포: AWS EC2
- CI/CD: GitHub Actions

---

## 디렉토리 구조
```
beauty/
  webpage/
    backend/
      main.py         ← FastAPI 서버 (업로드·상태·결과 API)
      inference.py    ← MedSAM2 LoRA 추론 로직
      requirements.txt
      uploads/        ← 업로드된 원본 영상 (자동 생성, gitignore)
      results/        ← 결과 영상 (자동 생성, gitignore)
    frontend/         ← React (Vite) 앱
      src/
        api.js
        App.jsx
        App.css
        components/
          UploadZone.jsx
          StatusBadge.jsx
          VideoPanel.jsx
    .gitignore
    progress.txt
    trouble.txt
```

---

## 진행 단계

### Phase 1: 환경 세팅 ✅
- [x] 프로젝트 구조 설계
- [x] requirements.txt 작성
- [x] React + Vite 프로젝트 초기화
- [x] Python 가상환경 + 패키지 설치

### Phase 2: Backend 구현 ✅
- [x] FastAPI 기본 서버 구축 (main.py)
- [x] 영상 업로드 API (POST /api/upload)
- [x] 추론 상태 확인 API (GET /api/status/{job_id})
- [x] 결과 영상 다운로드 API (GET /api/result/{job_id})
- [x] infer_lora_video.py 로직 분리 (inference.py)
- [x] 백그라운드 추론 (ThreadPoolExecutor)
- [x] ffmpeg H.264 재인코딩 (브라우저 호환)
- [x] 로컬 실행 테스트 완료

### Phase 3: Frontend 구현 (React) ✅
- [x] Vite + React 프로젝트 생성
- [x] API 호출 함수 모음 (api.js)
- [x] 드래그&드롭 업로드 컴포넌트 (UploadZone.jsx)
- [x] 추론 상태 표시 컴포넌트 (StatusBadge.jsx)
- [x] 원본/결과 영상 비교 패널 (VideoPanel.jsx)
- [x] 메인 앱 상태 관리 + 2초 폴링 (App.jsx)
- [x] 다크 테마 UI 스타일링
- [x] 로컬 실행 테스트 완료

### Phase 4: GitHub 연동 🔄
- [x] .gitignore 작성
- [x] beauty 레포지토리에 webpage/ 초기 버전 push
- [ ] GitHub Actions workflow 작성 (CI/CD)
- [ ] main 브랜치 push 시 AWS EC2 자동 배포 설정
- [ ] EC2 SSH 키 GitHub Secrets 등록

### Phase 5: AWS 배포 ✅
- [x] EC2 인스턴스 생성 (t3.micro, Ubuntu 22.04, ap-northeast-2)
- [x] SSH 접속 (PowerShell + pem 파일)
- [x] 서버 환경 세팅 (Python, Node.js v20, nginx, ffmpeg)
- [x] MedSAM2 클론 + 모델 체크포인트 scp 전송
- [x] Python venv + 패키지 설치 (FastAPI, PyTorch CPU, peft 등)
- [x] React 빌드: npm run build → /var/www/html/ 복사
- [x] FastAPI 배포: systemd 서비스 등록 (beauty.service)
- [x] nginx 리버스 프록시 설정 (/api/ → 8001)
- [x] 브라우저 접속 확인 (http://3.35.16.120)
- [x] 추론 테스트 완료

### Phase 4: GitHub Actions CI/CD ✅
- [x] .gitignore 작성
- [x] beauty 레포지토리에 webpage/ 초기 버전 push
- [x] GitHub Actions workflow 작성 (deploy.yml)
- [x] EC2 SSH 키 GitHub Secrets 등록 (EC2_HOST, EC2_USER, EC2_KEY)
- [x] main 브랜치 push 시 AWS EC2 자동 배포 확인

---

## 현재 상태
**모든 Phase 완료 — 로컬 개발 → GitHub → EC2 자동 배포 파이프라인 구축 완료**

## 트러블슈팅 기록
- 포트 8000 충돌 (WSL 프로세스) → 8001로 변경
- OpenCV mp4v 코덱 브라우저 미지원 → ffmpeg H.264 재인코딩으로 해결
- UniMatch CONF_THRESH=0.95 → unsupervised loss 무력화 → 0.5로 수정
- EC2 nginx 413 오류 → client_max_body_size 500M 설정
- EC2 OpenCV 코덱 미지원 → extract_frames를 ffmpeg subprocess로 교체
- EC2 MedSAM2 구버전 PNG 미지원 → misc.py에 .png 확장자 추가
