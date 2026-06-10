export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'user' : 'ai'}`}>
          {message.typing && !message.content ? (
            <div className="bubble-typing">
              <span /><span /><span />
            </div>
          ) : (
            <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
          )}
        </div>

        {/* Source citations */}
        {message.sources?.length > 0 && (
          <div className="sources">
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Sources:</span>
            {message.sources.map((s, i) => (
              <span key={i} className="source-tag">{s}</span>
            ))}
          </div>
        )}

        <span style={{ fontSize: 10, color: 'var(--text-muted)', paddingLeft: 4 }}>
          {new Date(message.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}
