import { useState, useRef, useCallback } from 'react';
import { streamChat } from '../services/api';
import { v4 as uuidv4 } from 'uuid';

const SESSION_ID = uuidv4();   // One session per browser tab

export function useChat(documentIds) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const cancelRef = useRef(null);

  const addUserMessage = useCallback((text) => {
    const msg = { id: uuidv4(), role: 'user', content: text, ts: Date.now() };
    setMessages(prev => [...prev, msg]);
    return msg;
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return;

    addUserMessage(text);
    setIsLoading(true);

    // Create placeholder AI message
    const aiId = uuidv4();
    setMessages(prev => [...prev, {
      id: aiId,
      role: 'assistant',
      content: '',
      sources: [],
      typing: true,
      ts: Date.now(),
    }]);

    let fullText = '';

    const cancel = streamChat(SESSION_ID, text, documentIds, {
      onToken: (token) => {
        fullText += token;
        setMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: fullText, typing: false } : m
        ));
      },
      onSources: (sources) => {
        setMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, sources } : m
        ));
      },
      onDone: () => {
        setIsLoading(false);
      },
      onError: (err) => {
        setMessages(prev => prev.map(m =>
          m.id === aiId
            ? { ...m, content: `Error: ${err}`, typing: false, error: true }
            : m
        ));
        setIsLoading(false);
      },
    });

    cancelRef.current = cancel;
  }, [isLoading, documentIds, addUserMessage]);

  const clearMessages = useCallback(() => {
    cancelRef.current?.();
    setMessages([]);
    setIsLoading(false);
  }, []);

  return { messages, isLoading, sendMessage, clearMessages, sessionId: SESSION_ID };
}
