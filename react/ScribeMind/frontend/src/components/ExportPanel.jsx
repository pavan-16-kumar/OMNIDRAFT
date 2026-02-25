import { useState } from 'react';
import { Download, FileText, FileType, FileCode, Loader2, Check } from 'lucide-react';
import { downloadNote } from '../api/client';

const formats = [
  { id: 'pdf', label: 'PDF', desc: 'Portable Document', icon: FileText, color: 'text-danger-400', bg: 'bg-danger-500/10' },
  { id: 'docx', label: 'DOCX', desc: 'Microsoft Word', icon: FileType, color: 'text-brand-400', bg: 'bg-brand-500/10' },
  { id: 'md', label: 'Markdown', desc: 'Plain Markdown', icon: FileCode, color: 'text-accent-400', bg: 'bg-accent-500/10' },
  { id: 'txt', label: 'TXT', desc: 'Plain Text', icon: FileCode, color: 'text-success-400', bg: 'bg-success-500/10' },
];

export default function ExportPanel({ noteId }) {
  const [downloading, setDownloading] = useState(null);
  const [downloaded, setDownloaded] = useState(null);

  if (!noteId) return null;

  const handleDownload = async (format) => {
    if (downloading) return;
    setDownloading(format);
    setDownloaded(null);

    try {
      await downloadNote(noteId, format);
      setDownloaded(format);
      setTimeout(() => setDownloaded(null), 2000);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="glass-card p-4 animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <Download className="w-4 h-4 text-brand-400" />
        <h3 className="text-sm font-semibold text-dark-200">Export</h3>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {formats.map((fmt) => {
          const Icon = fmt.icon;
          const isDownloading = downloading === fmt.id;
          const isDownloaded = downloaded === fmt.id;

          return (
            <button
              key={fmt.id}
              onClick={() => handleDownload(fmt.id)}
              disabled={!!downloading}
              className={`
                flex items-center gap-2.5 p-3 rounded-xl border border-white/5
                transition-all duration-200 group
                ${isDownloaded
                  ? 'bg-success-500/10 border-success-500/20'
                  : 'hover:bg-white/[0.03] hover:border-white/10'
                }
                disabled:opacity-50 disabled:cursor-not-allowed
              `}
            >
              <div className={`w-8 h-8 rounded-lg ${fmt.bg} flex items-center justify-center shrink-0 transition-transform group-hover:scale-105`}>
                {isDownloading ? (
                  <Loader2 className={`w-4 h-4 ${fmt.color} animate-spin`} />
                ) : isDownloaded ? (
                  <Check className="w-4 h-4 text-success-400" />
                ) : (
                  <Icon className={`w-4 h-4 ${fmt.color}`} />
                )}
              </div>
              <div className="text-left">
                <p className="text-xs font-medium text-dark-200">{fmt.label}</p>
                <p className="text-[10px] text-dark-600">{fmt.desc}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
