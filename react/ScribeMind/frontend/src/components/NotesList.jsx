import { FileText, Trash2, Clock, CheckCircle2, AlertTriangle, Loader2, ChevronRight } from 'lucide-react';

const statusConfig = {
  completed: { icon: CheckCircle2, color: 'text-success-400', bg: 'bg-success-500/10', label: 'Completed' },
  processing: { icon: Loader2, color: 'text-brand-400', bg: 'bg-brand-500/10', label: 'Processing', spin: true },
  verifying: { icon: Loader2, color: 'text-accent-400', bg: 'bg-accent-500/10', label: 'Verifying', spin: true },
  pending: { icon: Clock, color: 'text-dark-500', bg: 'bg-dark-700', label: 'Pending' },
  failed: { icon: AlertTriangle, color: 'text-danger-400', bg: 'bg-danger-500/10', label: 'Failed' },
};

export default function NotesList({ notes, activeNoteId, onSelectNote, onDeleteNote }) {
  if (!notes?.length) {
    return (
      <div className="text-center py-12 animate-fade-in">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-dark-700/50 flex items-center justify-center">
          <FileText className="w-8 h-8 text-dark-600" />
        </div>
        <p className="text-sm text-dark-500">No notes yet</p>
        <p className="text-xs text-dark-600 mt-1">Upload a handwritten note to get started</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 animate-fade-in">
      {notes.map((note, i) => {
        const status = statusConfig[note.status] || statusConfig.pending;
        const StatusIcon = status.icon;
        const isActive = note.note_id === activeNoteId;
        const confidencePercent = Math.round((note.confidence_score || 0) * 100);

        return (
          <div
            key={note.note_id}
            onClick={() => onSelectNote(note.note_id)}
            className={`
              group relative cursor-pointer rounded-xl p-3.5 
              transition-all duration-200 animate-fade-in
              ${isActive
                ? 'bg-brand-500/10 border border-brand-500/20'
                : 'hover:bg-white/[0.03] border border-transparent hover:border-white/5'
              }
            `}
            style={{ animationDelay: `${i * 50}ms` }}
          >
            <div className="flex items-start gap-3">
              {/* File icon */}
              <div className={`w-9 h-9 rounded-lg ${isActive ? 'bg-brand-500/20' : 'bg-dark-700/50'} flex items-center justify-center shrink-0 transition-colors`}>
                <FileText className={`w-4 h-4 ${isActive ? 'text-brand-400' : 'text-dark-500'}`} />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className={`text-sm font-medium truncate ${isActive ? 'text-brand-200' : 'text-dark-200'}`}>
                    {note.filename}
                  </p>
                  <ChevronRight className={`w-3.5 h-3.5 shrink-0 transition-all ${
                    isActive ? 'text-brand-400 opacity-100' : 'text-dark-600 opacity-0 group-hover:opacity-100'
                  }`} />
                </div>

                {/* Preview */}
                {note.preview && (
                  <p className="text-xs text-dark-600 mt-0.5 line-clamp-2">{note.preview}</p>
                )}

                {/* Status + Confidence */}
                <div className="flex items-center gap-3 mt-2">
                  <div className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md ${status.bg}`}>
                    <StatusIcon className={`w-3 h-3 ${status.color} ${status.spin ? 'animate-spin' : ''}`} />
                    <span className={`text-[10px] font-medium ${status.color}`}>{status.label}</span>
                  </div>

                  {note.status === 'completed' && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-16 h-1 bg-dark-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            note.confidence_score >= 0.9 ? 'bg-success-500' :
                            note.confidence_score >= 0.7 ? 'bg-warning-500' : 'bg-danger-500'
                          }`}
                          style={{ width: `${confidencePercent}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-dark-600 font-mono">{confidencePercent}%</span>
                    </div>
                  )}

                  <span className="text-[10px] text-dark-700 ml-auto">
                    {new Date(note.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>

              {/* Delete button */}
              <button
                onClick={(e) => { e.stopPropagation(); onDeleteNote(note.note_id); }}
                className="p-1.5 text-dark-700 hover:text-danger-400 hover:bg-danger-500/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                title="Delete note"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
