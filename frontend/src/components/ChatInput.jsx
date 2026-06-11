import { useRef, useState } from 'react';

export default function ChatInput({ onSend, onUpload, isLoading, uploading, uploadProgress }) {
  const [text, setText] = useState('');
  const fileRef = useRef(null);
  const textRef = useRef(null);

  const handleSend = () => {
    if (!text.trim() || isLoading) return;
    onSend(text.trim());
    setText('');
    textRef.current?.focus();
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) { onUpload(file); e.target.value = ''; }
  };

  return (
    <div className="input-bar">
      {uploading && (
        <div className="upload-progress" style={{ marginBottom: 12 }}>
          <div className="spinner" />
          <span>Processing document… {uploadProgress}%</span>
          <div style={{
            flex: 1, height: 4,
            background: 'var(--border-subtle)',
            borderRadius: 4, overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: `${uploadProgress}%`,
              background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
              transition: 'width 0.3s ease',
              borderRadius: 4,
            }} />
          </div>
        </div>
      )}

      <div className="input-wrapper">
        <textarea
          ref={textRef}
          value={text}
          onChange={e => {
            setText(e.target.value);
            // Auto-resize
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';
          }}
          onKeyDown={handleKey}
          placeholder="Ask a question or attach a brochure..."
          disabled={isLoading}
          rows={1}
          id="chat-input"
        />

        <div className="input-actions">
          <button
            className="icon-btn"
            title="Upload PDF or brochure"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            id="upload-file-btn"
          >
            ＋
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            style={{ display: 'none' }}
            onChange={handleFile}
          />

          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!text.trim() || isLoading}
            title="Send message (Enter)"
            id="send-message-btn"
          >
            {isLoading
              ? <span className="spinner" style={{ width: 14, height: 14 }} />
              : '→'}
          </button>
        </div>
      </div>

      <p className="input-hint">
        Enter to send · Shift+Enter for new line · Attach PDF/brochure
      </p>
    </div>
  );
}
