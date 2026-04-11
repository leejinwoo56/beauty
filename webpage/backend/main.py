"""
main.py
-------
FastAPI 서버 진입점.

실행 방법:
    cd beauty/webpage/backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API 구조:
    POST /api/upload          → 영상 업로드 + 추론 시작 → job_id 반환
    GET  /api/status/{job_id} → 추론 진행 상태 확인
    GET  /api/result/{job_id} → 완료된 결과 영상 다운로드
    GET  /health              → 서버 상태 확인
"""
#deploy
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference import load_models, process_video

# ── 디렉토리 준비 ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"   # 업로드된 원본 영상 임시 저장
RESULT_DIR = BASE_DIR / "results"   # 추론 완료된 결과 영상 저장
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ── 전역 상태 ──────────────────────────────────────────────────────────────────
# job_id → 상태 문자열: "pending" | "processing" | "done" | "error: ..."
jobs: dict[str, str] = {}

# AI 모델 (서버 시작 시 로드)
_model     = None
_predictor = None
_device    = None

# 추론은 CPU/GPU 집약적이므로 별도 스레드 풀에서 실행
_executor = ThreadPoolExecutor(max_workers=1)


# ── 서버 시작/종료 훅 ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트.
    서버가 시작될 때 AI 모델을 메모리에 로드한다.
    매 요청마다 모델을 로드하면 수십 초씩 걸리므로, 한 번만 로드 후 유지.
    """
    global _model, _predictor, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Startup] 사용 디바이스: {_device}")
    print("[Startup] MedSAM2 LoRA 모델 로딩 중... (처음 시작 시 수십 초 소요)")
    _model, _predictor = load_models(_device)
    print("[Startup] 모델 로딩 완료!")
    yield
    # 서버 종료 시 (현재 추가 정리 없음)
    _executor.shutdown(wait=False)


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedSAM2 SMAS 탐지 API",
    description="초음파 영상에서 SMAS층을 자동 탐지하는 웹 서비스",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정
# 프론트엔드(HTML)와 백엔드가 다른 포트에서 동작할 때 브라우저가 요청을 차단하는 것을 방지.
# 로컬 개발 단계에서는 allow_origins=["*"]로 전체 허용.
# (AWS 배포 후에는 실제 도메인으로 제한할 것)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드 정적 파일 서빙 (HTML/CSS/JS)
# /static/ URL로 접근 가능 → 예: http://localhost:8000/static/index.html
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── 백그라운드 추론 함수 ───────────────────────────────────────────────────────
def _run_inference(job_id: str, input_path: str, output_path: str):
    """
    별도 스레드에서 AI 추론 실행.
    FastAPI는 비동기(async) 기반이므로, CPU/GPU 집약 작업은
    ThreadPoolExecutor에 넘겨야 서버가 다른 요청도 처리 가능.
    """
    jobs[job_id] = "processing"
    try:
        process_video(_model, _predictor, input_path, output_path, _device)
        jobs[job_id] = "done"
        print(f"[Job {job_id[:8]}] 완료")
    except Exception as e:
        jobs[job_id] = f"error: {str(e)}"
        print(f"[Job {job_id[:8]}] 오류: {e}")


# ── API 엔드포인트 ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """서버 + 모델 상태 확인용. 배포 후 모니터링에도 사용."""
    return {
        "status": "ok",
        "device": str(_device),
        "model_loaded": _model is not None,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    영상 파일을 받아 저장하고 추론을 백그라운드에서 시작.

    - job_id: 추후 상태 확인 / 결과 다운로드에 사용하는 고유 ID
    - 추론은 수십 초~수 분 걸릴 수 있으므로 즉시 job_id만 반환
    - 클라이언트는 /api/status/{job_id}를 폴링해서 완료 여부 확인
    """
    # 허용 확장자 검사
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 형식: {suffix}")

    # 고유 ID 생성 및 파일 저장
    job_id = str(uuid.uuid4())
    input_path  = str(UPLOAD_DIR / f"{job_id}{suffix}")
    output_path = str(RESULT_DIR / f"{job_id}_result.mp4")

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # 추론을 스레드 풀에 제출 (논블로킹)
    jobs[job_id] = "pending"
    _executor.submit(_run_inference, job_id, input_path, output_path)

    print(f"[Job {job_id[:8]}] 추론 시작: {file.filename}")
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """
    추론 진행 상태 반환.
    상태값: "pending" → "processing" → "done" | "error: ..."
    클라이언트는 이 엔드포인트를 주기적으로 호출(폴링)해서 완료 여부 확인.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="존재하지 않는 job_id")
    return {"job_id": job_id, "status": jobs[job_id]}


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    """
    완료된 결과 영상 다운로드.
    done 상태일 때만 파일을 반환하고, 그 외에는 오류 반환.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="존재하지 않는 job_id")

    status = jobs[job_id]
    if status != "done":
        raise HTTPException(status_code=400, detail=f"아직 준비되지 않음: {status}")

    result_path = RESULT_DIR / f"{job_id}_result.mp4"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="결과 파일 없음")

    return FileResponse(
        path=str(result_path),
        media_type="video/mp4",
        filename="smas_result.mp4",
    )