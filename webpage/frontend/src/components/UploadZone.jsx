/**
 * UploadZone.jsx
 * --------------
 * 파일 드래그&드롭 또는 클릭으로 영상을 선택하는 컴포넌트.
 *
 * Props:
 *   onFileSelect(file) - 파일이 선택되면 부모(App)에 전달
 *   disabled           - 추론 중일 때 비활성화
 */

import { useRef, useState } from 'react';

export default function UploadZone({ onFileSelect, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  // 드래그가 영역 위에 들어왔을 때
  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  };

  // 드래그가 영역을 벗어났을 때
  const handleDragLeave = () => setDragging(false);

  // 파일을 드롭했을 때
  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  };

  // 클릭으로 파일 선택했을 때
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.avi,.mov,.mkv"
        style={{ display: 'none' }}
        onChange={handleChange}
      />
      <span className="upload-icon">🎬</span>
      <p>영상을 드래그하거나 클릭하여 선택</p>
      <p className="upload-sub">지원 형식: mp4, avi, mov, mkv</p>
    </div>
  );
}
