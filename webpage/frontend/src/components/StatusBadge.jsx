/**
 * StatusBadge.jsx
 * ---------------
 * 추론 진행 상태를 표시하는 컴포넌트.
 * 상태에 따라 스피너 또는 완료/오류 메시지를 보여줌.
 *
 * Props:
 *   status - "uploading" | "pending" | "processing" | "done" | "error: ..."
 */

const STATUS_LABEL = {
  uploading:  '서버에 업로드 중...',
  pending:    '대기 중 (곧 시작됩니다)',
  processing: 'SMAS 탐지 중... (영상 길이에 따라 수 분 소요)',
  done:       '완료!',
};

export default function StatusBadge({ status }) {
  if (!status) return null;

  const isError = status.startsWith('error');
  const isDone  = status === 'done';
  const label   = isError
    ? `오류 발생: ${status.replace('error: ', '')}`
    : (STATUS_LABEL[status] ?? status);

  return (
    <div className={`status-badge ${isDone ? 'done' : ''} ${isError ? 'error' : ''}`}>
      {!isDone && !isError && <div className="spinner" />}
      {isDone  && <span className="status-icon">✅</span>}
      {isError && <span className="status-icon">❌</span>}
      <p>{label}</p>
    </div>
  );
}
