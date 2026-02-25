import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage, FileText, Loader2, CheckCircle2, AlertTriangle, X } from 'lucide-react';
import { uploadFile } from '../api/client';

const ACCEPTED_TYPES = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/heic': ['.heic'],
  'image/heif': ['.heif'],
  'application/pdf': ['.pdf'],
};

export default function FileUpload({ onUploadComplete, setIsUploading }) {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null); // null | 'uploading' | 'processing' | 'success' | 'error'
  const [errorMsg, setErrorMsg] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    setSelectedFile(file);
    setUploadStatus('uploading');
    setUploadProgress(0);
    setErrorMsg('');
    setIsUploading(true);

    try {
      
      setUploadStatus('uploading');
      const result = await uploadFile(file, (progress) => {
        setUploadProgress(progress);
        if (progress >= 100) {
          setUploadStatus('processing');
        }
      });

      setUploadStatus('success');
      setTimeout(() => {
        onUploadComplete(result);
        setUploadStatus(null);
        setSelectedFile(null);
        setUploadProgress(0);
      }, 1500);
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadStatus('error');
      setErrorMsg(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [onUploadComplete, setIsUploading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    disabled: uploadStatus === 'uploading' || uploadStatus === 'processing',
    maxSize: 20 * 1024 * 1024,
  });


  const resetUpload = () => {
    setUploadStatus(null);
    setSelectedFile(null);
    setUploadProgress(0);
    setErrorMsg('');
  };

  return (
    <div className="animate-fade-in">
      <div
        {...getRootProps()}
        className={`
          relative group cursor-pointer
          rounded-2xl border-2 border-dashed
          transition-all duration-300 ease-out
          ${isDragActive
            ? 'dropzone-active border-brand-500'
            : uploadStatus === 'error'
              ? 'border-danger-500/40 bg-danger-500/5'
              : uploadStatus === 'success'
                ? 'border-success-500/40 bg-success-500/5'
                : 'border-dark-600/40 hover:border-brand-500/50 hover:bg-brand-500/[0.03]'
          }
          p-8 md:p-12
        `}
      >
        <input {...getInputProps()} />

        {/* Background pattern */}
        <div className="absolute inset-0 opacity-[0.02] pointer-events-none rounded-2xl overflow-hidden">
          <div className="w-full h-full" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, white 1px, transparent 0)`,
            backgroundSize: '24px 24px',
          }} />
        </div>

        <div className="relative flex flex-col items-center gap-4 text-center">
          {/* Status-dependent icon */}
          {uploadStatus === 'uploading' || uploadStatus === 'processing' ? (
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-brand-500/10 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-brand-400 animate-spin" />
              </div>
              {/* Spinning ring */}
              <svg className="absolute -inset-2 w-20 h-20 animate-[spin_3s_linear_infinite]" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="36" fill="none" stroke="url(#grad)" strokeWidth="2" strokeDasharray="40 180" strokeLinecap="round" />
                <defs><linearGradient id="grad"><stop offset="0%" stopColor="#5c7cfa" /><stop offset="100%" stopColor="#3bc9db" /></linearGradient></defs>
              </svg>
            </div>
          ) : uploadStatus === 'success' ? (
            <div className="w-16 h-16 rounded-2xl bg-success-500/10 flex items-center justify-center animate-scale-in">
              <CheckCircle2 className="w-8 h-8 text-success-400" />
            </div>
          ) : uploadStatus === 'error' ? (
            <div className="w-16 h-16 rounded-2xl bg-danger-500/10 flex items-center justify-center animate-scale-in">
              <AlertTriangle className="w-8 h-8 text-danger-400" />
            </div>
          ) : (
            <div className={`
              w-16 h-16 rounded-2xl flex items-center justify-center
              transition-all duration-300 
              ${isDragActive 
                ? 'bg-brand-500/20 scale-110' 
                : 'bg-dark-700/50 group-hover:bg-brand-500/10 group-hover:scale-105'
              }
            `}>
              <Upload className={`w-8 h-8 transition-colors ${isDragActive ? 'text-brand-400' : 'text-dark-400 group-hover:text-brand-400'}`} />
            </div>
          )}

          {/* Status text */}
          {uploadStatus === 'uploading' && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-brand-300">Uploading {selectedFile?.name}...</p>
              <div className="w-64 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-brand-500 to-accent-500 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-dark-500">{uploadProgress}%</p>
            </div>
          )}

          {uploadStatus === 'processing' && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-brand-300">Processing with AI...</p>
              <p className="text-xs text-dark-500">Transcribing handwriting → Verifying accuracy</p>
              <div className="w-64 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-brand-500 to-accent-500 rounded-full animate-shimmer" style={{ width: '100%' }} />
              </div>
            </div>
          )}

          {uploadStatus === 'success' && (
            <div>
              <p className="text-sm font-semibold text-success-400">Successfully processed!</p>
              <p className="text-xs text-dark-500 mt-1">Opening transcription view...</p>
            </div>
          )}

          {uploadStatus === 'error' && (
            <div className="space-y-2">
              <p className="text-sm font-semibold text-danger-400">Upload Failed</p>
              <p className="text-xs text-dark-500 max-w-xs">{errorMsg}</p>
              <button
                onClick={(e) => { e.stopPropagation(); resetUpload(); }}
                className="inline-flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300 transition-colors"
              >
                <X className="w-3 h-3" /> Try Again
              </button>
            </div>
          )}

          {!uploadStatus && (
            <>
              <div>
                <p className="text-base font-medium text-dark-200">
                  {isDragActive ? 'Drop your file here' : 'Drag & drop your handwritten notes'}
                </p>
                <p className="text-sm text-dark-500 mt-1">
                  or <span className="text-brand-400 font-medium underline underline-offset-2">browse files</span>
                </p>
              </div>
              <div className="flex items-center gap-2 flex-wrap justify-center">
                {['PNG', 'JPG', 'HEIC', 'PDF'].map((fmt) => (
                  <span key={fmt} className="px-2.5 py-1 text-[10px] font-mono font-medium bg-dark-700/60 text-dark-400 rounded-md uppercase tracking-wider">
                    {fmt}
                  </span>
                ))}
              </div>
              <p className="text-[11px] text-dark-600">Max 20 MB</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
