import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: BASE,
  headers: {
    'Bypass-Tunnel-Reminder': 'true'
  }
});

// ── Health ────────────────────────────────────────
export const getHealth = () => api.get('/health');

// ── Documents ─────────────────────────────────────
export const listDocuments = () => api.get('/documents');

export const uploadDocument = (file, onProgress) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round((e.loaded / e.total) * 100)),
  });
};

export const deleteDocument = (docId) => api.delete(`/documents/${docId}`);

export const deleteAllDocuments = () => api.delete('/documents');

// ── Extract ───────────────────────────────────────
export const extractProducts = (documentIds = []) =>
  api.post('/extract/json', { document_ids: documentIds });

export const getExtractionStatus = (jobId) =>
  api.get(`/extract/status/${jobId}`);

// ── Excel export ──────────────────────────────────
export const exportExcel = async (data, filename = 'extracted_data') => {
  const response = await api.post(
    '/export/excel',
    { data, filename },
    { responseType: 'blob' }
  );
  const url = URL.createObjectURL(response.data);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.xlsx`;
  a.click();
  URL.revokeObjectURL(url);
};

// ── Streaming chat (SSE) ──────────────────────────
export const streamChat = (sessionId, message, documentIds, callbacks) => {
  // We use fetch + ReadableStream for SSE (axios doesn't support native streaming well)
  const controller = new AbortController();

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      document_ids: documentIds,
    }),
    signal: controller.signal,
  }).then(async (res) => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // incomplete line stays in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') {
          callbacks.onDone?.();
          return;
        }
        try {
          const data = JSON.parse(payload);
          if (data.token)   callbacks.onToken?.(data.token);
          if (data.sources) callbacks.onSources?.(data.sources);
          if (data.error)   callbacks.onError?.(data.error);
        } catch (_) {
          // ignore parse errors in stream
        }
      }
    }
    callbacks.onDone?.();
  }).catch(err => {
    if (err.name !== 'AbortError') callbacks.onError?.(err.message);
  });

  return () => controller.abort(); // returns cancel fn
};
