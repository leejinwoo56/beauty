/**
 * api.js
 * ------
 * FastAPI 백엔드와 통신하는 함수 모음.
 * URL을 한 곳에서 관리해서, 나중에 AWS 배포 시 주소만 바꾸면 됨.
 */

// Vite 프록시 덕분에 개발 중에는 상대경로(/api/...)로 요청 가능.
// 배포 시에는 환경변수로 실제 서버 주소 주입 예정.
const BASE = import.meta.env.VITE_API_URL ?? '';

/**
 * 영상 파일을 업로드하고 추론을 시작.
 * @param {File} file - 사용자가 선택한 영상 파일
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function uploadVideo(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/api/upload`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `업로드 실패 (${res.status})`);
  }
  return res.json();
}

/**
 * 추론 진행 상태를 확인.
 * @param {string} jobId
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function getStatus(jobId) {
  const res = await fetch(`${BASE}/api/status/${jobId}`);
  if (!res.ok) throw new Error(`상태 확인 실패 (${res.status})`);
  return res.json();
}

/**
 * 완료된 결과 영상의 Blob URL을 생성.
 * video 태그의 src에 직접 넣을 수 있는 임시 URL 반환.
 * @param {string} jobId
 * @returns {Promise<string>} - Blob URL
 */
export async function fetchResultUrl(jobId) {
  const res = await fetch(`${BASE}/api/result/${jobId}`);
  if (!res.ok) throw new Error(`결과 다운로드 실패 (${res.status})`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}