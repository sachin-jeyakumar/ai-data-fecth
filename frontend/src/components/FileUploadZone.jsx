import { useState, useRef } from 'react';

export default function FileUploadZone({ onUpload, uploading, progress }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    const pdf = Array.from(files).find(f =>
      f.type === 'application/pdf' || f.name.match(/\.(pdf|png|jpg|jpeg)$/i)
    );
    if (pdf) onUpload(pdf);
  };

  return (
    <div
      className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        onChange={e => handleFiles(e.target.files)}
        style={{ display: 'none' }}
      />

      {uploading ? (
        <div style={{ pointerEvents: 'none' }}>
          <div className="upload-zone-icon">⚙️</div>
          <h3>Processing document…</h3>
          <p>Extracting text & building embeddings ({progress}%)</p>
          <div style={{
            marginTop: 16,
            height: 4,
            borderRadius: 4,
            background: 'var(--border-subtle)',
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: `${progress}%`,
              background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
              transition: 'width 0.3s ease',
              borderRadius: 4,
            }} />
          </div>
        </div>
      ) : (
        <>
          <div className="upload-zone-icon">📄</div>
          <h3>Drop your PDF or brochure here</h3>
          <p>Supports PDF, PNG, JPG — text &amp; scanned documents<br />Max 50 MB</p>
        </>
      )}
    </div>
  );
}
