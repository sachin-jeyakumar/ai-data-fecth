import { useState, useEffect, useCallback } from 'react';
import './index.css';

import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { useDocuments } from './hooks/useDocuments';
import { useChat } from './hooks/useChat';
import { getHealth, extractProducts, exportExcel, getExtractionStatus } from './services/api';

// ── Notification helper ─────────────────────────────────────────
function Notification({ note, onDismiss }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 4000);
    return () => clearTimeout(t);
  }, [onDismiss]);

  return (
    <div className={`notification ${note.type}`} onClick={onDismiss}>
      <span className="notif-icon">{note.type === 'success' ? '✓' : '✕'}</span>
      <span className="notif-text">{note.msg}</span>
    </div>
  );
}

export default function App() {
  const [notification, setNotification] = useState(null);
  const [ollamaStatus, setOllamaStatus] = useState(null);
  const [extractedData, setExtractedData] = useState(null);
  const [extractColumns, setExtractColumns] = useState([]);
  const [extracting, setExtracting] = useState(false);
  const [excelLoading, setExcelLoading] = useState(false);
  const [extractionProgress, setExtractionProgress] = useState(0);
  const [extractedCount, setExtractedCount] = useState(0);

  const notify = useCallback((note) => setNotification(note), []);

  const {
    documents,
    uploading,
    uploadProgress,
    addDocument,
    removeDocument,
    clearAllDocuments,
  } = useDocuments(notify);

  const docIds = documents.map(d => d.doc_id);

  const {
    messages,
    isLoading,
    sendMessage,
    clearMessages,
  } = useChat(docIds);

  // ── Poll Ollama status on mount ───────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const { data } = await getHealth();
        setOllamaStatus(data.ollama);
      } catch {
        setOllamaStatus({ running: false });
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Manual extraction ─────────────────────────────────────────
  const handleExtract = useCallback(async (ids) => {
    let targetIds = Array.isArray(ids) ? ids : [];
    if (targetIds.length === 0 && docIds.length > 0) {
      targetIds = [docIds[docIds.length - 1]];
    }

    if (targetIds.length === 0) {
      notify({ type: 'error', msg: '✕ Please mount a document first.' });
      return;
    }

    setExtracting(true);
    setExtractedData(null);
    setExtractionProgress(0);
    setExtractedCount(0);
    
    try {
      const { data } = await extractProducts(targetIds);
      const jobId = data.job_id;
      
      if (!jobId) {
        throw new Error("No job_id returned from server.");
      }

      // Poll every 2 seconds
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await getExtractionStatus(jobId);
          const job = statusRes.data;
          
          if (job.status === 'processing') {
            setExtractionProgress(job.progress || 0);
            setExtractedCount(job.products?.length || 0);
            if (job.products?.length > 0) {
              setExtractedData(job.products);
              setExtractColumns(job.columns || []);
            }
          } else if (job.status === 'completed') {
            clearInterval(pollInterval);
            setExtractionProgress(100);
            setExtractedData(job.products);
            setExtractColumns(job.columns || []);
            setExtracting(false);
            notify({ type: 'success', msg: `Extracted ${job.products.length} products successfully!` });
          } else if (job.status === 'failed') {
            clearInterval(pollInterval);
            setExtracting(false);
            notify({ type: 'error', msg: `Extraction failed: ${job.error || 'Unknown error'}` });
          }
        } catch (pollErr) {
          console.error("Polling extraction job failed", pollErr);
          clearInterval(pollInterval);
          setExtracting(false);
          notify({ type: 'error', msg: 'Connection lost to server. Please try again.' });
        }
      }, 2000);

    } catch (err) {
      console.error("Triggering extraction failed", err);
      notify({ type: 'error', msg: 'Extraction failed. Please check backend server logs.' });
      setExtracting(false);
    }
  }, [docIds, notify]);

  // ── Auto-extract when a new document is added ─────────────────
  const handleUpload = useCallback(async (file) => {
    const doc = await addDocument(file);
    if (!doc) return;
    setTimeout(() => handleExtract([doc.doc_id]), 500);
  }, [addDocument, handleExtract]);

  // ── Excel download ────────────────────────────────────────────
  const handleDownloadExcel = useCallback(async () => {
    if (!extractedData?.length) return;
    setExcelLoading(true);
    try {
      await exportExcel(extractedData, 'product_data');
      notify({ type: 'success', msg: 'Excel file downloaded successfully!' });
    } catch {
      notify({ type: 'error', msg: 'Excel export failed.' });
    } finally {
      setExcelLoading(false);
    }
  }, [extractedData, notify]);

  // ── New session ───────────────────────────────────────────────
  const handleNewSession = useCallback(async () => {
    clearMessages();
    setExtractedData(null);
    setExtractColumns([]);
    await clearAllDocuments();
  }, [clearMessages, clearAllDocuments]);

  // ── Quick prompt click ────────────────────────────────────────
  const handleQuickPrompt = useCallback((prompt) => {
    if (prompt.includes('Extract all product')) {
      handleExtract();
    } else {
      sendMessage(prompt);
    }
  }, [handleExtract, sendMessage]);

  return (
    <div className="app-shell">
      <Sidebar
        documents={documents}
        ollamaStatus={ollamaStatus}
        onNewChat={handleNewSession}
        onDeleteDoc={removeDocument}
        onUpload={handleUpload}
        uploading={uploading}
        uploadProgress={uploadProgress}
      />

      <div className="main-area">
        {/* Top Bar */}
        <div className="topbar">
          <div className="topbar-left">
            <h1>AI Data Assistant</h1>
            <p>
              {documents.length === 0
                ? 'No documents mounted'
                : `${documents.length} document${documents.length !== 1 ? 's' : ''} mounted · RAG Active`}
            </p>
          </div>

          <div className="topbar-actions">
            {documents.length > 0 && (
              <button
                className="btn btn-primary"
                onClick={() => handleExtract()}
                disabled={extracting}
                id="extract-btn"
              >
                {extracting
                  ? <><span className="spinner btn-spinner" /> Extracting…</>
                  : 'Run Extraction'
                }
              </button>
            )}
            {extractedData?.length > 0 && (
              <button
                className="btn btn-success"
                onClick={handleDownloadExcel}
                disabled={excelLoading}
                id="topbar-download-btn"
              >
                {excelLoading
                  ? <><span className="spinner btn-spinner" /> Exporting…</>
                  : 'Export Excel'
                }
              </button>
            )}
          </div>
        </div>

        {/* Conversational Chat Feed */}
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          extractedData={extractedData}
          extractColumns={extractColumns}
          onQuickPrompt={handleQuickPrompt}
          onDownloadExcel={handleDownloadExcel}
          excelLoading={excelLoading}
          extracting={extracting}
          extractionProgress={extractionProgress}
          extractedCount={extractedCount}
        />

        {/* Floating Chat Input bar */}
        <ChatInput
          onSend={sendMessage}
          onUpload={handleUpload}
          isLoading={isLoading || extracting}
          uploading={uploading}
          uploadProgress={uploadProgress}
        />
      </div>

      {notification && (
        <Notification
          note={notification}
          onDismiss={() => setNotification(null)}
        />
      )}
    </div>
  );
}
