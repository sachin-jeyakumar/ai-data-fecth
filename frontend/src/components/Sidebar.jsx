import { useRef } from 'react';

export default function Sidebar({
  documents,
  ollamaStatus,
  onNewChat,
  onDeleteDoc,
  onUpload,
  uploading,
  uploadProgress,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
      e.target.value = ''; // Reset
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <aside className="sidebar">
      {/* Control Deck Header */}
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-text">
            DATA INDEXER
            <span>DOCUMENTS &amp; STATUS</span>
          </div>
        </div>
        
        <button className="new-chat-btn" onClick={onNewChat} id="new-chat-btn">
          Clear Workspace
        </button>
      </div>

      {/* Workspace Explorer */}
      <div className="sidebar-section-header">
        <span className="sidebar-section-title">Workspace</span>
        <button className="upload-add-btn" onClick={handleUploadClick} title="Upload new document">
          ＋ Add
        </button>
        <input 
          ref={fileInputRef}
          type="file" 
          accept=".pdf,.png,.jpg,.jpeg" 
          onChange={handleFileChange} 
          style={{ display: 'none' }}
        />
      </div>

      <div className="sidebar-docs">
        {/* Document list */}
        {documents.length === 0 && !uploading ? (
          <div className="empty-docs-placeholder" onClick={handleUploadClick}>
            <span>No documents mounted</span>
            <p>Click here or drop a PDF to begin</p>
          </div>
        ) : (
          <>
            {documents.map(doc => (
              <div key={doc.doc_id} className="doc-item">
                <div className={`doc-status-badge ${doc.status === 'ready' ? 'ready' : 'error'}`}>
                  PDF
                </div>
                <div className="doc-info">
                  <div className="doc-name truncate" title={doc.filename}>
                    {doc.filename}
                  </div>
                  <div className="doc-meta">
                    {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}
                    {doc.has_ocr ? ' · OCR' : ''}
                    <span className="doc-status-text"> · Ready</span>
                  </div>
                </div>
                <button
                  className="doc-delete"
                  onClick={() => onDeleteDoc(doc.doc_id)}
                  title="Unmount document"
                >
                  ×
                </button>
              </div>
            ))}

            {/* Active upload list item */}
            {uploading && (
              <div className="doc-item uploading">
                <div className="doc-status-badge scanning">
                  SCAN
                </div>
                <div className="doc-info">
                  <div className="doc-name truncate">
                    Uploading document...
                  </div>
                  <div className="doc-meta">
                    Parsing layers · {uploadProgress}%
                  </div>
                  <div className="sidebar-progress-track">
                    <div className="sidebar-progress-fill" style={{ width: `${uploadProgress}%` }} />
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* System Status Panel */}
      <div className="sidebar-footer">
        <div className="system-status-title">SYSTEM STATUS</div>
        
        <div className="system-status-group">
          <div className="system-status-row">
            <span className="status-label">AI CORE:</span>
            <span className="status-value active-text">
              <span className="status-pulse-dot" /> ACTIVE
            </span>
          </div>

          <div className="system-status-row">
            <span className="status-label">EXTRACTION:</span>
            <span className="status-value highlight-cyan">OPTIMIZED</span>
          </div>

          <div className="system-status-row">
            <span className="status-label">LATENCY:</span>
            <span className="status-value">24ms</span>
          </div>

          <div className="system-status-row">
            <span className="status-label">CLOUD NODE:</span>
            <span className={`status-value ${ollamaStatus?.running ? 'active-text' : 'inactive-text'}`}>
              {ollamaStatus?.running ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
        </div>

        <div className="status-pill">
          <div className={`status-dot ${ollamaStatus?.running ? 'online' : ''}`} />
          <span>
            {ollamaStatus?.running ? 'Ollama Embedder Ready' : 'Local Embedder Offline'}
          </span>
        </div>
      </div>
    </aside>
  );
}
