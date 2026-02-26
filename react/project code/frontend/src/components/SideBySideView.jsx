import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  Image, Type, AlertTriangle, CheckCircle2, Edit3, 
  Save, X, ChevronDown, ZoomIn, ZoomOut, RotateCw 
} from 'lucide-react';
import { updateNoteText } from '../api/client';
import TextToSpeech from './TextToSpeech';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function SideBySideView({ note, onTextUpdate }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const [zoom, setZoom] = useState(1);
  const [activeTab, setActiveTab] = useState('verified'); // 'raw' | 'verified'
  const [showFlags, setShowFlags] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEditText(note?.verified_markdown || note?.raw_markdown || '');
  }, [note]);

  if (!note) return null;

  const imageSrc = note.image_path ? `${API_URL}${note.image_path}` : null;
  const displayText = activeTab === 'verified' 
    ? (note.verified_markdown || note.raw_markdown || '') 
    : (note.raw_markdown || '');
  const confidencePercent = Math.round((note.confidence_score || 0) * 100);

  const handleSave = async () => {
    try {
      await updateNoteText(note.note_id, editText);
      onTextUpdate?.(editText);
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to save:', err);
    }
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.9) return 'from-success-500 to-success-400';
    if (score >= 0.7) return 'from-warning-500 to-warning-400';
    return 'from-danger-500 to-danger-400';
  };

  const getConfidenceLabel = (score) => {
    if (score >= 0.95) return 'Excellent';
    if (score >= 0.9) return 'High';
    if (score >= 0.7) return 'Good';
    if (score >= 0.5) return 'Fair';
    return 'Low';
  };

  return (
    <div className="animate-fade-in space-y-4">
      {/* ── Confidence Bar ─────────────────────────────────── */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <CheckCircle2 className={`w-4 h-4 ${
              note.confidence_score >= 0.9 ? 'text-success-400' :
              note.confidence_score >= 0.7 ? 'text-warning-400' : 'text-danger-400'
            }`} />
            <span className="text-sm font-medium text-dark-200">
              Accuracy: {confidencePercent}% — {getConfidenceLabel(note.confidence_score)}
            </span>
          </div>

          {note.flags?.length > 0 && (
            <button
              onClick={() => setShowFlags(!showFlags)}
              className="flex items-center gap-1 text-xs text-warning-400 hover:text-warning-300 transition-colors"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              {note.flags.length} flag{note.flags.length > 1 ? 's' : ''}
              <ChevronDown className={`w-3 h-3 transition-transform ${showFlags ? 'rotate-180' : ''}`} />
            </button>
          )}
        </div>

        {/* Progress bar */}
        <div className="w-full h-1.5 bg-dark-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-linear-to-r ${getConfidenceColor(note.confidence_score)} transition-all duration-1000`}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>

        {/* Flags dropdown */}
        {showFlags && note.flags?.length > 0 && (
          <div className="mt-3 space-y-2 animate-fade-in">
            {note.flags.map((flag, i) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-warning-500/5 border border-warning-500/10 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-warning-400 mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm text-dark-200">
                    <span className="font-mono text-warning-400">"{flag.word}"</span>
                    {flag.suggestion && (
                      <span className="text-dark-500"> → suggested: <span className="text-success-400 font-mono">"{flag.suggestion}"</span></span>
                    )}
                  </p>
                  {flag.context && (
                    <p className="text-xs text-dark-600 mt-0.5 truncate">...{flag.context}...</p>
                  )}
                  <p className="text-[10px] text-dark-600 mt-0.5">
                    Confidence: {Math.round(flag.confidence * 100)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Text-to-Speech Player ────────────────────────────── */}
      <TextToSpeech text={displayText} />

      {/* ── Side-by-Side Panels ────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Image Panel */}
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
            <div className="flex items-center gap-2">
              <Image className="w-4 h-4 text-brand-400" />
              <span className="text-sm font-medium text-dark-300">Original</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setZoom(z => Math.max(0.25, z - 0.25))}
                className="p-1.5 text-dark-500 hover:text-dark-200 hover:bg-white/5 rounded-lg transition-colors"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] text-dark-500 font-mono w-10 text-center">{Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom(z => Math.min(3, z + 0.25))}
                className="p-1.5 text-dark-500 hover:text-dark-200 hover:bg-white/5 rounded-lg transition-colors"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setZoom(1)}
                className="p-1.5 text-dark-500 hover:text-dark-200 hover:bg-white/5 rounded-lg transition-colors"
              >
                <RotateCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="p-4 overflow-auto" style={{ maxHeight: '600px' }}>
            {imageSrc ? (
              <img
                src={imageSrc}
                alt={note.filename}
                className="rounded-lg transition-transform duration-200"
                style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
              />
            ) : (
              <div className="flex items-center justify-center h-64 text-dark-600">
                <p className="text-sm">No image preview available</p>
              </div>
            )}
          </div>
        </div>

        {/* Text Panel */}
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
            <div className="flex items-center gap-2">
              <Type className="w-4 h-4 text-accent-400" />
              {/* Tab switcher */}
              <div className="flex bg-dark-800/60 rounded-lg p-0.5">
                <button
                  onClick={() => setActiveTab('verified')}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-all ${
                    activeTab === 'verified'
                      ? 'bg-brand-500/20 text-brand-300'
                      : 'text-dark-500 hover:text-dark-300'
                  }`}
                >
                  Verified
                </button>
                <button
                  onClick={() => setActiveTab('raw')}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-all ${
                    activeTab === 'raw'
                      ? 'bg-brand-500/20 text-brand-300'
                      : 'text-dark-500 hover:text-dark-300'
                  }`}
                >
                  Raw OCR
                </button>
              </div>
            </div>

            <div className="flex items-center gap-1">
              {isEditing ? (
                <>
                  <button
                    onClick={handleSave}
                    className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-success-400 hover:bg-success-500/10 rounded-lg transition-colors"
                  >
                    <Save className="w-3.5 h-3.5" /> Save
                  </button>
                  <button
                    onClick={() => { setIsEditing(false); setEditText(displayText); }}
                    className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-dark-500 hover:bg-white/5 rounded-lg transition-colors"
                  >
                    <X className="w-3.5 h-3.5" /> Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={() => { setIsEditing(true); setEditText(displayText); }}
                  className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-dark-400 hover:text-brand-400 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <Edit3 className="w-3.5 h-3.5" /> Edit
                </button>
              )}
            </div>
          </div>

          <div className="p-4 overflow-auto" style={{ maxHeight: '600px' }}>
            {isEditing ? (
              <textarea
                ref={textareaRef}
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                className="w-full h-[500px] bg-transparent text-dark-200 font-mono text-sm leading-relaxed resize-none outline-none placeholder:text-dark-600"
                placeholder="Edit transcription..."
                spellCheck={false}
              />
            ) : (
              <div className="prose-scribe">
                <ReactMarkdown>{displayText || '*No transcription available*'}</ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
