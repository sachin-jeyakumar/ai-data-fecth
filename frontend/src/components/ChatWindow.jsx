import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import DataTable from './DataTable';

const QUICK_PROMPTS = [
  '📦 Extract all product details',
  '💰 List all prices and models',
  '📐 Get dimensions and specifications',
  '✨ Summarise what products are in this document',
];

export default function ChatWindow({
  messages,
  isLoading,
  extractedData,
  extractColumns,
  onQuickPrompt,
  onDownloadExcel,
  excelLoading,
  extracting,
  extractionProgress,
  extractedCount,
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, extractedData, extracting]);

  const isEmpty = messages.length === 0 && !extractedData && !extracting;

  return (
    <div className="chat-window">
      {isEmpty ? (
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <h2>AI Data Extractor</h2>
          <p>
            Upload a PDF brochure or document using the <strong>📎 button</strong> below,
            then ask me to extract product data — I'll pull it into a structured table
            you can download as Excel.
          </p>
          <div className="quick-actions">
            {QUICK_PROMPTS.map(p => (
              <button
                key={p}
                className="quick-action-chip"
                onClick={() => onQuickPrompt(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Extracted data table shown inline */}
          {extractedData && (
            <div style={{ maxWidth: '100%' }}>
              <DataTable
                data={extractedData}
                columns={extractColumns}
                onDownload={onDownloadExcel}
                isLoading={excelLoading}
              />
            </div>
          )}

          {/* Progress Card when extracting */}
          {extracting && (
            <div className="extraction-progress-container">
              <div className="extraction-progress-card">
                <div className="progress-card-header">
                  <span className="pulse-icon">⚡</span>
                  <h3>AI Data Extraction in Progress</h3>
                </div>
                <p className="progress-card-sub">
                  Scanning document chunks sequentially to respect rate limits. Do not close this window.
                </p>
                <div className="progress-bar-track">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${extractionProgress}%` }}
                  />
                </div>
                <div className="progress-card-stats">
                  <span className="pct-text">{extractionProgress}% Complete</span>
                  <span className="count-text">{extractedCount} products found</span>
                </div>
              </div>
            </div>
          )}

          {isLoading && messages.at(-1)?.role !== 'assistant' && (
            <div className="message ai">
              <div className="avatar ai">🤖</div>
              <div className="bubble-wrap">
                <div className="bubble ai">
                  <div className="bubble-typing">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
