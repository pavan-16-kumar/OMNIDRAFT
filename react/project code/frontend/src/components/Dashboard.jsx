import { useState, useEffect, useCallback } from 'react';
import { 
  PenTool, Upload, BookOpen, Settings, RefreshCw, 
  Wifi, WifiOff, ChevronLeft, ChevronRight, User
} from 'lucide-react';
import FileUpload from './FileUpload';
import SideBySideView from './SideBySideView';
import NotesList from './NotesList';
import ExportPanel from './ExportPanel';
import ChatSidebar from './ChatSidebar';
import { fetchNotes, fetchNote, deleteNote, checkHealth } from '../api/client';

export default function Dashboard() {
  const [notes, setNotes] = useState([]);
  const [activeNote, setActiveNote] = useState(null);
  const [activeView, setActiveView] = useState('upload'); // 'upload' | 'viewer'
  const [isUploading, setIsUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Check backend health
  useEffect(() => {
    checkHealth()
      .then((data) => setBackendStatus(data.status === 'ok' ? 'online' : 'offline'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  // Load notes on mount
  const loadNotes = useCallback(async () => {
    try {
      const data = await fetchNotes();
      setNotes(data);
    } catch (err) {
      console.error('Failed to load notes:', err);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadNotes();
  }, [loadNotes]);

  const handleUploadComplete = async (result) => {
    await loadNotes();
    setActiveNote(result);
    setActiveView('viewer');
  };

  const handleSelectNote = async (noteId) => {
    try {
      const note = await fetchNote(noteId);
      setActiveNote(note);
      setActiveView('viewer');
    } catch (err) {
      console.error('Failed to fetch note:', err);
    }
  };

  const handleDeleteNote = async (noteId) => {
    try {
      await deleteNote(noteId);
      if (activeNote?.note_id === noteId) {
        setActiveNote(null);
        setActiveView('upload');
      }
      await loadNotes();
    } catch (err) {
      console.error('Failed to delete note:', err);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadNotes();
    setTimeout(() => setIsRefreshing(false), 600);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* ══════════════════════════════════════════════════════════
          TOP NAV BAR
         ══════════════════════════════════════════════════════════ */}
      <header className="glass sticky top-0 z-40 border-b border-white/5">
        <div className="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-lg shadow-brand-500/20">
                <PenTool className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-surface border-2 border-surface rounded-full">
                <div className={`w-full h-full rounded-full ${
                  backendStatus === 'online' ? 'bg-success-500 animate-pulse' :
                  backendStatus === 'offline' ? 'bg-danger-500' : 'bg-warning-500 animate-pulse'
                }`} />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-bold text-gradient tracking-tight">OmniDraft</h1>
              <p className="text-[10px] text-dark-600 -mt-0.5 tracking-wide">Universal Document Architecture</p>
            </div>
          </div>

          {/* Status line */}
          <div className="hidden sm:flex items-center gap-4">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg ${
              backendStatus === 'online' ? 'bg-success-500/10' : 'bg-danger-500/10'
            }`}>
              {backendStatus === 'online' ? (
                <Wifi className="w-3.5 h-3.5 text-success-400" />
              ) : (
                <WifiOff className="w-3.5 h-3.5 text-danger-400" />
              )}
              <span className={`text-[11px] font-medium ${
                backendStatus === 'online' ? 'text-success-400' : 'text-danger-400'
              }`}>
                {backendStatus === 'online' ? 'Backend Online' : 'Backend Offline'}
              </span>
            </div>

            <span className="text-[10px] text-dark-700 font-mono">v1.0.0</span>
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════
          MAIN CONTENT
         ══════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex overflow-hidden">
        {/* ── LEFT SIDEBAR ──────────────────────────── */}
        <aside
          className={`
            ${sidebarOpen ? 'w-72 lg:w-80' : 'w-0'}
            transition-all duration-300 border-r border-white/5
            bg-surface-raised flex flex-col shrink-0 overflow-hidden
          `}
        >
          <div className="p-4 border-b border-white/5">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-brand-400" />
                <h2 className="text-sm font-semibold text-dark-200">My Notes</h2>
                <span className="px-1.5 py-0.5 text-[10px] font-mono text-dark-500 bg-dark-700/60 rounded">
                  {notes.length}
                </span>
              </div>

              <button
                onClick={handleRefresh}
                className="p-1.5 text-dark-500 hover:text-brand-400 hover:bg-white/5 rounded-lg transition-all"
                title="Refresh notes"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {/* New upload button */}
            <button
              onClick={() => { setActiveView('upload'); setActiveNote(null); }}
              className={`
                w-full flex items-center gap-2 px-3 py-2.5 rounded-xl
                text-sm font-medium transition-all duration-200
                ${activeView === 'upload'
                  ? 'bg-brand-500/10 text-brand-300 border border-brand-500/20'
                  : 'text-dark-400 hover:bg-white/[0.03] hover:text-dark-200 border border-transparent'
                }
              `}
            >
              <Upload className="w-4 h-4" />
              New Upload
            </button>
          </div>

          {/* Notes list */}
          <div className="flex-1 overflow-y-auto p-3">
            <NotesList
              notes={notes}
              activeNoteId={activeNote?.note_id}
              onSelectNote={handleSelectNote}
              onDeleteNote={handleDeleteNote}
            />
          </div>

        </aside>

        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="relative z-10 -ml-3 mt-6 w-6 h-12 bg-surface-raised border border-white/5 rounded-r-lg flex items-center justify-center text-dark-600 hover:text-dark-300 hover:bg-surface-overlay transition-all"
        >
          {sidebarOpen ? <ChevronLeft className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        </button>

        {/* ── MAIN AREA ─────────────────────────────── */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {activeView === 'upload' && (
              <div className="space-y-8 animate-fade-in">
                {/* Hero section */}
                <div className="text-center space-y-3 mb-12">
                  <div className="inline-flex items-center gap-2 px-3 py-1 bg-brand-500/10 border border-brand-500/20 rounded-full mb-4">
                    <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-pulse" />
                    <span className="text-xs text-brand-300 font-medium">Multi-Agent AI Pipeline</span>
                  </div>
                  <h2 className="text-3xl sm:text-4xl font-bold text-dark-50">
                    Raw Inputs to Documents,{' '}
                    <span className="text-gradient">Architected.</span>
                  </h2>
                  <p className="text-dark-500 max-w-lg mx-auto text-sm leading-relaxed">
                    Upload any handwritten note, scribble, or layout and our AI agents will transcribe, verify, 
                    and format it into a perfect 2026-grade document.
                  </p>
                </div>

                {/* Upload zone */}
                <FileUpload
                  onUploadComplete={handleUploadComplete}
                  isUploading={isUploading}
                  setIsUploading={setIsUploading}
                />

                {/* Feature cards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8">
                  {[
                    {
                      icon: '🔍',
                      title: 'Vision OCR',
                      desc: 'Gemini/GPT-4o reads your handwriting with high accuracy',
                    },
                    {
                      icon: '✅',
                      title: 'AI Verification',
                      desc: 'A second agent cross-checks every word for 100% accuracy',
                    },
                    {
                      icon: '💬',
                      title: 'Chat with Notes',
                      desc: 'RAG-powered search lets you ask questions about your notes',
                    },
                  ].map((feature, i) => (
                    <div
                      key={i}
                      className="glass-card p-5 text-center animate-fade-in"
                      style={{ animationDelay: `${(i + 1) * 100}ms` }}
                    >
                      <div className="text-2xl mb-3">{feature.icon}</div>
                      <h3 className="text-sm font-semibold text-dark-200 mb-1">{feature.title}</h3>
                      <p className="text-xs text-dark-500 leading-relaxed">{feature.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeView === 'viewer' && activeNote && (
              <div className="space-y-6 animate-fade-in">
                {/* Note header */}
                <div className="flex items-center justify-between">
                  <div>
                    <button
                      onClick={() => { setActiveView('upload'); setActiveNote(null); }}
                      className="text-xs text-dark-600 hover:text-brand-400 transition-colors mb-1 flex items-center gap-1"
                    >
                      <ChevronLeft className="w-3 h-3" /> Back to upload
                    </button>
                    <h2 className="text-xl font-bold text-dark-100">{activeNote.filename}</h2>
                  </div>
                </div>

                {/* Export panel */}
                <ExportPanel noteId={activeNote.note_id} filename={activeNote.filename} />

                {/* Side-by-side view */}
                <SideBySideView
                  note={activeNote}
                  onTextUpdate={(text) => {
                    setActiveNote((prev) => ({
                      ...prev,
                      verified_markdown: text,
                      confidence_score: 1.0,
                      flags: [],
                    }));
                    loadNotes();
                  }}
                />
              </div>
            )}
          </div>
        </main>
      </div>

      {/* ── CHAT SIDEBAR ──────────────────────────── */}
      <ChatSidebar activeNoteId={activeNote?.note_id} />
    </div>
  );
}
