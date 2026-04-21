# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dockerfile — Sandbox V2 (JupyterLab 기반 자유도 높은 sandbox)
#
# 빌드 컨텍스트: beauty/ 루트 디렉토리에서 실행
#   docker compose -f sandbox_v2/docker-compose.yml up --build
#
# 구조:
#   /image/code/     ← 이미지에 고정된 원본 코드 (읽기 전용 역할)
#   /workspace/      ← 사용자 작업 공간 (재시작 시 /image/code 로 초기화)
#     code/          ← 코드 작업 공간 (수정 가능, 재시작 시 초기화)
#     saved/         ← 영구 저장 공간 (.pt 차단, 나머지 유지)
#     data/          ← 데이터 업로드 공간
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ── 시스템 패키지 설치 ────────────────────────────────────────────────────────
# inotify-tools: .pt 파일 감시 데몬에 필요 (inotifywait)
# ffmpeg: 영상 처리
# libgl1, libglib2.0: opencv 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    inotify-tools \
    && rm -rf /var/lib/apt/lists/*

# ── Python 패키지 설치 ────────────────────────────────────────────────────────
# jupyterlab: 브라우저 기반 Python IDE
# jupyter-server: JupyterLab 서버 엔진
RUN pip install --no-cache-dir \
    jupyterlab==4.3.6 \
    jupyter-server==2.15.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    scipy>=1.11.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    peft>=0.9.0

# ── MedSAM2 패키지 설치 ───────────────────────────────────────────────────────
# sam2 모듈을 Python 패키지로 설치 (setup.py 기반)
# 빌드 컨텍스트: beauty/ 루트
COPY MedSAM2/ /image/code/MedSAM2/
RUN pip install --no-cache-dir -e /image/code/MedSAM2/

# ── 모델 코드 복사 (원본 스냅샷) ─────────────────────────────────────────────
# /image/code/ 는 이미지 빌드 시점의 코드 스냅샷
# entrypoint.sh 에서 컨테이너 시작 시 /workspace/code/ 로 복사됨
COPY medsam_finetune_lora/ /image/code/medsam_finetune_lora/

# ── JupyterLab 설정 복사 ──────────────────────────────────────────────────────
COPY sandbox_v2/jupyter_config/ /etc/jupyter/

# ── entrypoint 스크립트 복사 ──────────────────────────────────────────────────
COPY sandbox_v2/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── 비루트 사용자 설정 ────────────────────────────────────────────────────────
# 컨테이너를 root로 실행하면 보안 위험이 있으므로 별도 사용자 생성
RUN groupadd -r sandbox && useradd -r -g sandbox -u 1000 -m sandboxuser

# 작업 공간 및 체크포인트 출력 디렉토리 생성 및 소유권 설정
RUN mkdir -p /workspace/code /workspace/saved /workspace/data \
    && mkdir -p /checkpoints/output \
    && chown -R sandboxuser:sandbox /workspace \
    && chown -R sandboxuser:sandbox /image/code \
    && chown -R sandboxuser:sandbox /checkpoints

# 체크포인트는 컨테이너 내부에서도 읽기 전용 권한
# (실제 read-only 마운트는 docker-compose에서 처리)
RUN chmod -R 555 /image/code/MedSAM2/checkpoints 2>/dev/null || true

USER sandboxuser

# ── 환경 변수 ────────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace/code/MedSAM2:/workspace/code/medsam_finetune_lora"

WORKDIR /workspace

EXPOSE 8888

ENTRYPOINT ["/entrypoint.sh"]
