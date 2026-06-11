import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import DataTable from './DataTable';

const QUICK_PROMPTS = [
  'Extract all product details',
  'List all part numbers and models',
  'Get tool specifications and dimensions',
  'Summarize what products are in the brochure',
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
  }, [messages, extractedData, extracting, isLoading]);

  const isEmpty = messages.length === 0 && !extractedData && !extracting;

  return (
    <div className="chat-window">
      {isEmpty ? (
        <div className="empty-state">
          <h2>Document Intelligence Workspace</h2>
          <p className="empty-state-subtitle">
            Mount a PDF brochure on the left Control Deck, then ask questions about its content or trigger a structured product extraction.
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
        <div className="messages-container">
          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Holographic scan card rendering inline */}
          {extracting && (
            <div className="extraction-progress-container">
              <div className="extraction-progress-card">
                <div className="scanner-line-bar" />
                <div className="progress-card-header">
                  <span className="pulse-icon">
                    <span className="spinner" style={{ width: 10, height: 10 }} />
                  </span>
                  <h3>Scanning Document Layers...</h3>
                </div>
                <p className="progress-card-sub">
                  Running parallel tabular extraction via Cloud LLM matrix.
                </p>
                <div className="progress-bar-track">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${extractionProgress}%` }}
                  />
                </div>
                <div className="progress-card-stats">
                  <span className="pct-text">{extractionProgress}% Complete</span>
                  <span className="count-text">{extractedCount} records found</span>
                </div>
              </div>
            </div>
          )}

          {/* Interactive spreadsheet table rendering inline */}
          {extractedData && (
            <div className="inline-table-wrapper animate-fade-in">
              <DataTable
                data={extractedData}
                columns={extractColumns}
                onDownload={onDownloadExcel}
                isLoading={excelLoading}
              />
            </div>
          )}

          {isLoading && messages.at(-1)?.role !== 'assistant' && (
            <div className="message ai">
              <div className="avatar ai">AI</div>
              <div className="bubble-wrap">
                <div className="bubble ai typing-indicator">
                  <div className="bubble-typing">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
