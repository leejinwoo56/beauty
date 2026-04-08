/**
 * App.jsx
 * -------
 * 메인 컴포넌트. 전체 상태를 관리하고 하위 컴포넌트에 props를 전달.
 *
 * 상태 흐름:
 *   idle → (파일 선택) → ready → (업로드) → uploading
 *   → pending → processing → done | error:...
 */

import { useState, useEffect, useRef } from 'react';
import UploadZone from './components/UploadZone';
import StatusBadge from './components/StatusBadge';
import VideoPanel from './components/VideoPanel';
import { uploadVideo, getStatus, fetchResultUrl } from './api';
import './App.css';

// 폴링 간격 (ms): 2초마다 백엔드에 상태 확인 요청
const POLL_INTERVAL = 2000;

export default function App() {
  const [file, setFile]               = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [status, setStatus]           = useState('idle');
  const [jobId, setJobId]             = useState(null);
  const [resultUrl, setResultUrl]     = useState(null);

  const pollRef = useRef(null);

  // 파일 선택 시
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setStatus('ready');
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    setOriginalUrl(URL.createObjectURL(selectedFile));
  };

  // "분석 시작" 클릭
  const handleStart = async () => {
    if (!file) return;
    setStatus('uploading');
    setResultUrl(null);
    setJobId(null);

    try {
      const { job_id } = await uploadVideo(file);
      setJobId(job_id);
      setStatus('pending');
    } catch (err) {
      setStatus(`error: ${err.message}`);
    }
  };

  // jobId가 생기면 폴링 시작 (2초 간격으로 상태 확인)
  useEffect(() => {
    if (!jobId) return;

    pollRef.current = setInterval(async () => {
      try {
        const { status: s } = await getStatus(jobId);
        setStatus(s);

        if (s === 'done') {
          clearInterval(pollRef.current);
          const url = await fetchResultUrl(jobId);
          setResultUrl(url);
        } else if (s.startsWith('error')) {
          clearInterval(pollRef.current);
        }
      } catch (err) {
        clearInterval(pollRef.current);
        setStatus(`error: ${err.message}`);
      }
    }, POLL_INTERVAL);

    return () => clearInterval(pollRef.current);
  }, [jobId]);

  // 결과 영상 다운로드
  const handleDownload = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = 'smas_result.mp4';
    a.click();
  };

  // 처음 상태로 초기화
  const handleReset = () => {
    clearInterval(pollRef.current);
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (resultUrl)   URL.revokeObjectURL(resultUrl);
    setFile(null);
    setOriginalUrl(null);
    setStatus('idle');
    setJobId(null);
    setResultUrl(null);
  };

  const isProcessing = ['uploading', 'pending', 'processing'].includes(status);
  const isDone       = status === 'done';
  const isError      = status.startsWith('error');
  const showStatus   = isProcessing || isDone || isError;

  return (
    <div className="app">
      <header className="app-header">
        <h1>MedSAM2 SMAS 탐지</h1>
        <p>초음파 영상을 업로드하면 SMAS층이 자동으로 표시된 영상을 반환합니다.</p>
      </header>

      <main className="app-main">
        {/* 업로드 영역 */}
        {!isDone && (
          <section className="upload-section">
            <UploadZone onFileSelect={handleFileSelect} disabled={isProcessing} />

            {file && (
              <div className="file-info">
                <span>선택된 파일: <strong>{file.name}</strong></span>
                <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            )}

            <button
              className="btn-start"
              onClick={handleStart}
              disabled={!file || isProcessing}
            >
              {isProcessing ? '분석 중...' : '분석 시작'}
            </button>
          </section>
        )}

        {/* 상태 표시 */}
        {showStatus && (
          <section className="status-section">
            <StatusBadge status={status} />
            {(isDone || isError) && (
              <button className="btn-reset" onClick={handleReset}>
                다시 시작
              </button>
            )}
          </section>
        )}

        {/* 결과 영상 패널 */}
        {isDone && (
          <VideoPanel
            originalUrl={originalUrl}
            resultUrl={resultUrl}
            onDownload={handleDownload}
          />
        )}
      </main>
    </div>
  );
}
