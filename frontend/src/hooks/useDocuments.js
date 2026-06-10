import { useState, useCallback } from 'react';
import { uploadDocument, deleteDocument } from '../services/api';

export function useDocuments(onNotify) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const addDocument = useCallback(async (file) => {
    setUploading(true);
    setUploadProgress(0);
    try {
      const { data } = await uploadDocument(file, setUploadProgress);
      setDocuments(prev => [...prev, data]);
      onNotify?.({ type: 'success', msg: `✅ "${file.name}" indexed successfully!` });
      return data;
    } catch (err) {
      onNotify?.({ type: 'error', msg: `❌ Failed to upload: ${err.response?.data?.detail || err.message}` });
      return null;
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [onNotify]);

  const removeDocument = useCallback(async (docId) => {
    try {
      await deleteDocument(docId);
      setDocuments(prev => prev.filter(d => d.doc_id !== docId));
      onNotify?.({ type: 'success', msg: '🗑️ Document removed.' });
    } catch (err) {
      onNotify?.({ type: 'error', msg: 'Failed to delete document.' });
    }
  }, [onNotify]);

  return { documents, uploading, uploadProgress, addDocument, removeDocument };
}
