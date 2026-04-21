import os

# ── 인증 설정 ─────────────────────────────────────────────────────────────────
# 환경 변수 JUPYTER_TOKEN 으로 접속 토큰 설정
# docker-compose.yml 의 .env 파일에서 값을 주입함
# 토큰 없이 접근 시 로그인 페이지로 리다이렉트됨
c.ServerApp.token = os.environ.get("JUPYTER_TOKEN", "")
c.ServerApp.password = os.environ.get("JUPYTER_PASSWORD", "")

# 외부 접속 허용 (0.0.0.0 = 모든 인터페이스에서 수신)
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False

# ── 루트 디렉토리 설정 ────────────────────────────────────────────────────────
# JupyterLab 파일 브라우저의 최상위 경로
# /workspace/ 바깥의 파일은 파일 브라우저에서 보이지 않음
# → /image/code/MedSAM2/checkpoints/ 는 접근 불가
c.ServerApp.root_dir = "/workspace"

# ── 파일 다운로드 제한 ────────────────────────────────────────────────────────
# /workspace/saved/ 안의 .pt 파일은 entrypoint.sh 의 감시 데몬이 삭제함
# 추가로 서버 레벨에서 .pt 파일 다운로드 요청을 차단
class BlockCheckpointDownload:
    """
    .pt 파일 다운로드 요청을 차단하는 미들웨어.
    /files/ 엔드포인트를 통한 파일 다운로드가 여기서 필터링됨.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            # /files/ 경로로 .pt 파일 다운로드 시도를 차단
            if "/files/" in path and path.endswith(".pt"):
                from starlette.responses import Response
                response = Response("체크포인트 파일 다운로드가 차단되었습니다.", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

# ── 터미널 설정 ──────────────────────────────────────────────────────────────
# 터미널(bash) 사용 허용 → 학습/추론 스크립트 직접 실행 가능
c.ServerApp.terminals_enabled = True

# ── 로깅 ─────────────────────────────────────────────────────────────────────
c.ServerApp.log_level = "INFO"
