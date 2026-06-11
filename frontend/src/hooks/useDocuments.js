import { useState, useCallback, useEffect } from 'react';
import { uploadDocument, deleteDocument, deleteAllDocuments, listDocuments } from '../services/api';

export function useDocuments(onNotify) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Fetch documents on mount
  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const { data } = await listDocuments();
        setDocuments(data || []);
      } catch (err) {
        console.error("Failed to load documents on mount:", err);
      }
    };
    fetchDocs();
  }, []);

  const addDocument = useCallback(async (file) => {
    setUploading(true);
    setUploadProgress(0);
    try {
      const { data } = await uploadDocument(file, setUploadProgress);
      setDocuments(prev => [...prev, data]);
      onNotify?.({ type: 'success', msg: `✓ "${file.name}" indexed successfully!` });
      return data;
    } catch (err) {
      onNotify?.({ type: 'error', msg: `✕ Failed to upload: ${err.response?.data?.detail || err.message}` });
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
      onNotify?.({ type: 'success', msg: 'Document unmounted.' });
    } catch (err) {
      onNotify?.({ type: 'error', msg: 'Failed to delete document.' });
    }
  }, [onNotify]);

  const clearAllDocuments = useCallback(async () => {
    try {
      await deleteAllDocuments();
      setDocuments([]);
    } catch (err) {
      console.error(err);
    }
  }, []);

  return { documents, uploading, uploadProgress, addDocument, removeDocument, clearAllDocuments };
}
