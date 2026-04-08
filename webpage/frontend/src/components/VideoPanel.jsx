/**
 * VideoPanel.jsx
 * --------------
 * 원본 영상과 SMAS 탐지 결과 영상을 나란히 보여주는 컴포넌트.
 *
 * Props:
 *   originalUrl  - 원본 영상의 Blob URL (또는 파일 객체 URL)
 *   resultUrl    - 결과 영상의 Blob URL
 *   onDownload() - 다운로드 버튼 클릭 핸들러
 */

export default function VideoPanel({ originalUrl, resultUrl, onDownload }) {
  return (
    <div className="video-panel">
      <div className="video-pair">
        <div className="video-box">
          <h3>원본 영상</h3>
          <video src={originalUrl} controls />
        </div>
        <div className="video-box">
          <h3>SMAS 탐지 결과</h3>
          {resultUrl
            ? <video src={resultUrl} controls />
            : <div className="video-placeholder">결과 로드 중...</div>
          }
        </div>
      </div>
      <div className="result-actions">
        <button className="btn-download" onClick={onDownload} disabled={!resultUrl}>
          결과 영상 다운로드
        </button>
      </div>
    </div>
  );
}
