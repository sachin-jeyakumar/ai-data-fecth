export default function Sidebar({
  documents,
  ollamaStatus,
  onNewChat,
  onDeleteDoc,
  onUpload,
  uploading,
  uploadProgress,
}) {
  return (
    <aside className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">🧠</div>
          <div className="logo-text">
            AI Data Fetcher
            <span>Document Intelligence</span>
          </div>
        </div>

        <button className="new-chat-btn" onClick={onNewChat} id="new-chat-btn">
          ✏️ New Session
        </button>
      </div>

      {/* Documents list */}
      <p className="sidebar-section-title">📁 Indexed Documents</p>

      <div className="sidebar-docs">
        {documents.length === 0 ? (
          <div style={{
            padding: '20px 10px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: 12,
            lineHeight: 1.6,
          }}>
            No documents yet.<br />
            Upload a PDF to get started.
          </div>
        ) : (
          documents.map(doc => (
            <div key={doc.doc_id} className="doc-item">
              <div className={`doc-icon ${doc.status === 'ready' ? 'ready' : 'pdf'}`}>
                {doc.status === 'ready' ? '✅' : '📄'}
              </div>
              <div className="doc-info">
                <div className="doc-name truncate" title={doc.filename}>
                  {doc.filename}
                </div>
                <div className="doc-meta">
                  {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}
                  {doc.has_ocr ? ' · OCR' : ''}
                  {doc.chunk_count ? ` · ${doc.chunk_count} chunks` : ''}
                </div>
              </div>
              <button
                className="doc-delete"
                onClick={() => onDeleteDoc(doc.doc_id)}
                title="Remove document"
              >
                🗑️
              </button>
            </div>
          ))
        )}
      </div>

      {/* Status footer */}
      <div className="sidebar-footer">
        <div className="status-pill">
          <div className={`status-dot ${ollamaStatus?.running ? 'online' : ''}`} />
          <span>
            {ollamaStatus?.running
              ? `Ollama · ${ollamaStatus.llm_ready ? 'Model ready' : 'Model loading…'}`
              : 'Ollama offline'}
          </span>
        </div>
        {ollamaStatus?.running && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, paddingLeft: 4 }}>
            🤖 qwen2.5:14b · nomic-embed-text
          </div>
        )}
      </div>
    </aside>
  );
}
