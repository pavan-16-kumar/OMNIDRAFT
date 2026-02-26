import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { MessageCircle, Send, Loader2, Bot, User, Sparkles, X, Minimize2, Maximize2 } from 'lucide-react';
import { chatWithNotes, fetchChatSuggestions } from '../api/client';

export default function ChatSidebar({ activeNoteId }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "👋 Hi! I'm your notes assistant. Upload some handwritten notes and ask me anything about them!",
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [isFetchSuggestionsLoading, setIsFetchSuggestionsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
      loadSuggestions();
    }
  }, [isOpen, activeNoteId]);

  const loadSuggestions = async () => {
    setIsFetchSuggestionsLoading(true);
    try {
      const data = await fetchChatSuggestions(activeNoteId);
      setSuggestions(data.suggestions || []);
    } catch (err) {
      console.error('Failed to load suggestions:', err);
    } finally {
      setIsFetchSuggestionsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    // Auto-send if it's a suggestion
    setTimeout(() => {
      handleSendWithContent(suggestion);
    }, 100);
  };

  const handleSend = async () => {
    await handleSendWithContent(input);
  };

  const handleSendWithContent = async (content) => {
    const query = content.trim();
    if (!query || isLoading) return;

    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await chatWithNotes(query, activeNoteId);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '❌ Sorry, I encountered an error. Please make sure you have uploaded notes and the backend is running.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Chat toggle button
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 p-4 bg-linear-to-br from-brand-500 to-brand-700 text-white rounded-2xl shadow-2xl shadow-brand-500/25 hover:shadow-brand-500/40 hover:scale-105 transition-all duration-300 group"
        id="chat-toggle"
      >
        <MessageCircle className="w-6 h-6 group-hover:scale-110 transition-transform" />
        <span className="absolute -top-1 -right-1 w-3 h-3 bg-accent-500 rounded-full animate-pulse" />
      </button>
    );
  }

  return (
    <div
      className={`
        fixed z-50 bg-surface-raised border border-white/5 shadow-2xl shadow-black/40
        flex flex-col animate-slide-right
        ${isExpanded
          ? 'inset-4 rounded-2xl'
          : 'bottom-6 right-6 w-[400px] h-[600px] rounded-2xl'
        }
      `}
      id="chat-sidebar"
    >
      {/* ── Header ───────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-linear-to-br from-brand-500 to-accent-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-dark-100">Chat with Notes</p>
            <p className="text-[10px] text-dark-600">
              {activeNoteId ? 'Searching current note' : 'Searching all notes'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-dark-500 hover:text-dark-200 hover:bg-white/5 rounded-lg transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1.5 text-dark-500 hover:text-dark-200 hover:bg-white/5 rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Messages ─────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
          >
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-lg bg-brand-500/10 flex items-center justify-center shrink-0 mt-0.5">
                <Bot className="w-4 h-4 text-brand-400" />
              </div>
            )}

            <div
              className={`
                max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed
                ${msg.role === 'user'
                  ? 'bg-brand-500 text-white rounded-br-md'
                  : 'bg-surface-overlay text-dark-200 rounded-bl-md border border-white/5'
                }
              `}
            >
              {msg.role === 'assistant' ? (
                <div className="prose-scribe text-sm">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <p>{msg.content}</p>
              )}

              {msg.sources?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-white/5">
                  <p className="text-[10px] text-dark-500 mb-1">Sources:</p>
                  <div className="flex flex-wrap gap-1">
                    {msg.sources.map((src, j) => (
                      <span key={j} className="px-1.5 py-0.5 text-[9px] bg-brand-500/10 text-brand-400 rounded font-mono">
                        {src.slice(0, 8)}...
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {msg.role === 'user' && (
              <div className="w-7 h-7 rounded-lg bg-dark-700 flex items-center justify-center shrink-0 mt-0.5">
                <User className="w-4 h-4 text-dark-400" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-3 animate-fade-in">
            <div className="w-7 h-7 rounded-lg bg-brand-500/10 flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4 text-brand-400" />
            </div>
            <div className="px-3.5 py-2.5 bg-surface-overlay border border-white/5 rounded-2xl rounded-bl-md">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1.5 h-1.5 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Suggestions ────────────────────────── */}
      {!isLoading && suggestions.length > 0 && (
        <div className="px-4 py-2 flex flex-wrap gap-2 animate-fade-in">
          {suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => handleSuggestionClick(s)}
              className="text-[10px] px-2.5 py-1.5 bg-brand-500/5 hover:bg-brand-500/10 border border-brand-500/10 hover:border-brand-500/20 text-brand-300 rounded-full transition-all text-left max-w-full truncate"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* ── Input ────────────────────────────── */}
      <div className="p-3 border-t border-white/5">
        <div className="flex items-end gap-2 bg-surface-overlay border border-white/5 rounded-xl p-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your notes..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-dark-200 resize-none outline-none placeholder:text-dark-600 max-h-24"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="p-2 bg-brand-500 text-white rounded-lg hover:bg-brand-600 disabled:opacity-30 disabled:cursor-not-allowed transition-all shrink-0"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <p className="text-[10px] text-dark-600 mt-1.5 text-center">Press Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
