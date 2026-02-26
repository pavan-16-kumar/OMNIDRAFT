import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Volume2, Play, Pause, Square,
  ChevronDown, Mic2, Gauge, Globe2, Loader2, AlertCircle
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * TextToSpeech — Microsoft Edge Neural TTS via backend /tts endpoint
 * ✅ Zero cost  ✅ Unlimited  ✅ Crystal-clear neural voices
 * ✅ Perfect for Telugu, Tamil, Hindi, Kannada, Malayalam and 40+ languages
 */

// ── Script detection (Unicode ranges → BCP-47 lang code) ────────────────────
const SCRIPTS = [
  { name: 'Telugu',    lang: 'te-IN', voice: 'te-IN-ShrutiNeural',   range: [0x0C00, 0x0C7F], emoji: '🇮🇳' },
  { name: 'Tamil',     lang: 'ta-IN', voice: 'ta-IN-PallaviNeural',  range: [0x0B80, 0x0BFF], emoji: '🇮🇳' },
  { name: 'Hindi',     lang: 'hi-IN', voice: 'hi-IN-SwaraNeural',    range: [0x0900, 0x097F], emoji: '🇮🇳' },
  { name: 'Kannada',   lang: 'kn-IN', voice: 'kn-IN-SapnaNeural',    range: [0x0C80, 0x0CFF], emoji: '🇮🇳' },
  { name: 'Malayalam', lang: 'ml-IN', voice: 'ml-IN-SobhanaNeural',  range: [0x0D00, 0x0D7F], emoji: '🇮🇳' },
  { name: 'Bengali',   lang: 'bn-IN', voice: 'bn-IN-TanishaaNeural', range: [0x0980, 0x09FF], emoji: '🇮🇳' },
  { name: 'Gujarati',  lang: 'gu-IN', voice: 'gu-IN-DhwaniNeural',   range: [0x0A80, 0x0AFF], emoji: '🇮🇳' },
  { name: 'Punjabi',   lang: 'pa-IN', voice: 'pa-IN-OjasNeural',     range: [0x0A00, 0x0A7F], emoji: '🇮🇳' },
  { name: 'Odia',      lang: 'or-IN', voice: 'or-IN-SubhasiniNeural',range: [0x0B00, 0x0B7F], emoji: '🇮🇳' },
  { name: 'Sinhala',   lang: 'si-LK', voice: 'si-LK-ThiliniNeural', range: [0x0D80, 0x0DFF], emoji: '🇱🇰' },
  { name: 'Arabic',    lang: 'ar-SA', voice: 'ar-SA-ZariyahNeural',  range: [0x0600, 0x06FF], emoji: '🇸🇦' },
  { name: 'Hebrew',    lang: 'he-IL', voice: 'he-IL-HilaNeural',     range: [0x0590, 0x05FF], emoji: '🇮🇱' },
  { name: 'Thai',      lang: 'th-TH', voice: 'th-TH-PremwadeeNeural',range: [0x0E00, 0x0E7F], emoji: '🇹🇭' },
  { name: 'Chinese',   lang: 'zh-CN', voice: 'zh-CN-XiaoxiaoNeural', range: [0x4E00, 0x9FFF], emoji: '🇨🇳' },
  { name: 'Japanese',  lang: 'ja-JP', voice: 'ja-JP-NanamiNeural',   range: [0x3040, 0x30FF], emoji: '🇯🇵' },
  { name: 'Korean',    lang: 'ko-KR', voice: 'ko-KR-SunHiNeural',    range: [0xAC00, 0xD7AF], emoji: '🇰🇷' },
  { name: 'Cyrillic',  lang: 'ru-RU', voice: 'ru-RU-SvetlanaNeural', range: [0x0400, 0x04FF], emoji: '🇷🇺' },
  { name: 'Greek',     lang: 'el-GR', voice: 'el-GR-AthinaNeural',   range: [0x0370, 0x03FF], emoji: '🇬🇷' },
  { name: 'Georgian',  lang: 'ka-GE', voice: 'ka-GE-EkaNeural',      range: [0x10A0, 0x10FF], emoji: '🇬🇪' },
];

const ENGLISH_SCRIPT = { name: 'English', lang: 'en-US', voice: 'en-US-AriaNeural', emoji: '🇺🇸' };

function detectScript(text) {
  if (!text) return null;
  const clean = text.replace(/\s/g, '');
  if (!clean.length) return null;

  let best = null, bestCount = 0;
  for (const s of SCRIPTS) {
    let count = 0;
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (cp >= s.range[0] && cp <= s.range[1]) count++;
    }
    if (count > bestCount) { bestCount = count; best = s; }
  }
  // Need at least 5% non-Latin characters to count as that script
  return (bestCount / clean.length) >= 0.05 ? best : null;
}

// ── Preset voices shown in UI ────────────────────────────────────────────────
const PRESET_VOICES = [
  { label: 'en-US-AriaNeural',       display: '🇺🇸 Aria (English US)',      lang: 'en-US' },
  { label: 'en-IN-NeerjaNeural',     display: '🇮🇳 Neerja (Indian English)',lang: 'en-IN' },
  { label: 'te-IN-ShrutiNeural',     display: '🇮🇳 Shruti (Telugu)',        lang: 'te-IN' },
  { label: 'ta-IN-PallaviNeural',    display: '🇮🇳 Pallavi (Tamil)',        lang: 'ta-IN' },
  { label: 'hi-IN-SwaraNeural',      display: '🇮🇳 Swara (Hindi)',          lang: 'hi-IN' },
  { label: 'kn-IN-SapnaNeural',      display: '🇮🇳 Sapna (Kannada)',        lang: 'kn-IN' },
  { label: 'ml-IN-SobhanaNeural',    display: '🇮🇳 Sobhana (Malayalam)',    lang: 'ml-IN' },
  { label: 'bn-IN-TanishaaNeural',   display: '🇮🇳 Tanishaa (Bengali)',     lang: 'bn-IN' },
  { label: 'ar-SA-ZariyahNeural',    display: '🇸🇦 Zariyah (Arabic)',       lang: 'ar-SA' },
  { label: 'zh-CN-XiaoxiaoNeural',   display: '🇨🇳 Xiaoxiao (Chinese)',     lang: 'zh-CN' },
  { label: 'ja-JP-NanamiNeural',     display: '🇯🇵 Nanami (Japanese)',      lang: 'ja-JP' },
  { label: 'fr-FR-DeniseNeural',     display: '🇫🇷 Denise (French)',        lang: 'fr-FR' },
  { label: 'de-DE-KatjaNeural',      display: '🇩🇪 Katja (German)',         lang: 'de-DE' },
  { label: 'es-ES-ElviraNeural',     display: '🇪🇸 Elvira (Spanish)',       lang: 'es-ES' },
  { label: 'ru-RU-SvetlanaNeural',   display: '🇷🇺 Svetlana (Russian)',     lang: 'ru-RU' },
];

// ── Component ────────────────────────────────────────────────────────────────
export default function TextToSpeech({ text }) {
  const [isSpeaking,  setIsSpeaking]  = useState(false);
  const [isPaused,    setIsPaused]    = useState(false);
  const [isLoading,   setIsLoading]   = useState(false);
  const [error,       setError]       = useState('');
  const [progress,    setProgress]    = useState(0);
  const [showPanel,   setShowPanel]   = useState(false);
  const [rate,        setRate]        = useState(0.92);
  // 'auto' = detect language from text | or a fixed voice label
  const [voiceMode,   setVoiceMode]   = useState('auto');
  const [detectedScript, setDetectedScript] = useState(null);

  const audioRef   = useRef(null);   // HTMLAudioElement
  const abortRef   = useRef(null);   // AbortController for fetch
  const timerRef   = useRef(null);

  // ── Detect script whenever text changes ─────────────────────────────
  useEffect(() => {
    setDetectedScript(detectScript(text));
    stopAudio();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  // ── Cleanup ──────────────────────────────────────────────────────────
  useEffect(() => {
    return () => { stopAudio(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopAudio = useCallback(() => {
    abortRef.current?.abort();
    clearInterval(timerRef.current);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
    setIsSpeaking(false);
    setIsPaused(false);
    setIsLoading(false);
    setProgress(0);
    setError('');
  }, []);

  // ── Resolve which lang + voice to send ──────────────────────────────
  const resolveVoice = useCallback(() => {
    if (voiceMode !== 'auto') {
      const preset = PRESET_VOICES.find(p => p.label === voiceMode);
      return { lang: preset?.lang ?? 'en-US', voice: voiceMode };
    }
    // Auto: use detected script or fallback to English
    const s = detectedScript ?? ENGLISH_SCRIPT;
    return { lang: s.lang, voice: s.voice ?? ENGLISH_SCRIPT.voice };
  }, [voiceMode, detectedScript]);

  // ── Speak ──────────────────────────────────────────────────────────
  const speak = useCallback(async () => {
    if (!text?.trim()) return;

    stopAudio();
    setError('');
    setIsLoading(true);

    const { lang, voice } = resolveVoice();
    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${API_URL}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang, voice, rate }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'TTS request failed' }));
        throw new Error(err.detail || 'TTS request failed');
      }

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audioRef.current = audio;

      // Progress tracking
      audio.addEventListener('loadedmetadata', () => {
        timerRef.current = setInterval(() => {
          if (!audio.duration) return;
          setProgress(Math.min((audio.currentTime / audio.duration) * 100, 99));
        }, 200);
      });

      audio.addEventListener('ended', () => {
        clearInterval(timerRef.current);
        setIsSpeaking(false);
        setIsPaused(false);
        setProgress(100);
        URL.revokeObjectURL(url);
        setTimeout(() => setProgress(0), 700);
      });

      audio.addEventListener('error', () => {
        clearInterval(timerRef.current);
        setIsSpeaking(false); setIsPaused(false); setProgress(0);
        setError('Audio playback error');
        URL.revokeObjectURL(url);
      });

      setIsLoading(false);
      setIsSpeaking(true);
      await audio.play();

    } catch (e) {
      if (e.name === 'AbortError') return; // user cancelled
      setIsLoading(false);
      setIsSpeaking(false);
      setError(e.message || 'TTS failed');
    }
  }, [text, rate, resolveVoice, stopAudio]);

  // ── Pause / Resume ──────────────────────────────────────────────────
  const togglePause = useCallback(() => {
    if (!audioRef.current) return;
    if (isPaused) { audioRef.current.play(); setIsPaused(false); }
    else          { audioRef.current.pause(); setIsPaused(true); }
  }, [isPaused]);

  const hasText       = Boolean(text?.trim());
  const activePreset  = PRESET_VOICES.find(p => p.label === voiceMode);
  const autoScript    = detectedScript ?? ENGLISH_SCRIPT;
  const displayVoice  = voiceMode === 'auto'
    ? `${autoScript.emoji} ${autoScript.name} — ${autoScript.voice ?? ENGLISH_SCRIPT.voice}`
    : activePreset?.display ?? voiceMode;

  return (
    <div className="tts-root">

      {/* ── Main Bar ─────────────────────────────────────────────────── */}
      <div
        className="glass-card px-4 py-3 flex items-center gap-3"
        style={{
          borderColor: isSpeaking
            ? 'rgba(92,124,250,0.45)'
            : error ? 'rgba(255,107,107,0.3)' : undefined,
          transition: 'border-color 0.3s ease',
        }}
      >
        {/* Icon badge */}
        <div
          className="shrink-0 w-9 h-9 rounded-xl flex items-center justify-center"
          style={{
            background: isLoading ? 'rgba(92,124,250,0.18)'
              : isSpeaking ? 'linear-gradient(135deg,#5c7cfa,#3bc9db)'
              : error ? 'rgba(255,107,107,0.18)' : 'rgba(92,124,250,0.12)',
            boxShadow: isSpeaking ? '0 0 20px rgba(92,124,250,0.5)' : 'none',
            transition: 'all 0.3s ease',
          }}
        >
          {isLoading
            ? <Loader2  className="w-4 h-4 text-brand-400" style={{ animation: 'spin 1s linear infinite' }} />
            : isSpeaking
            ? <Volume2  className="w-4 h-4 text-white" style={{ animation: 'tts-wave 0.7s ease-in-out infinite' }} />
            : error
            ? <AlertCircle className="w-4 h-4 text-danger-400" />
            : <Mic2     className="w-4 h-4 text-brand-400" />
          }
        </div>

        {/* Centre: labels + progress */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-[11px] font-semibold text-dark-300 tracking-wide uppercase">
              {isLoading ? 'Generating audio…'
                : isSpeaking ? (isPaused ? '⏸ Paused' : '🔊 Speaking…')
                : error ? 'TTS Error'
                : 'Read Aloud'}
            </span>

            {/* Detected language chip */}
            {detectedScript && !isSpeaking && !isLoading && !error && (
              <span
                className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium"
                style={{
                  background: 'rgba(59,201,219,0.12)',
                  color: '#66d9e8',
                  border: '1px solid rgba(59,201,219,0.25)',
                }}
              >
                <Globe2 className="w-2.5 h-2.5" />
                {detectedScript.emoji} {detectedScript.name} detected
              </span>
            )}

            {isSpeaking && (
              <span className="text-[10px] text-brand-400 font-mono ml-auto">
                {Math.round(progress)}%
              </span>
            )}
          </div>

          {/* Progress / voice label */}
          {(isSpeaking || isLoading) ? (
            <div className="w-full h-1 bg-dark-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: isLoading ? '100%' : `${progress}%`,
                  background: 'linear-gradient(90deg,#5c7cfa,#3bc9db)',
                  animation: isLoading ? 'tts-indeterminate 1.4s ease-in-out infinite' : 'none',
                }}
              />
            </div>
          ) : error ? (
            <p className="text-[10px] text-danger-400 truncate">{error}</p>
          ) : (
            <p className="text-[10px] text-dark-600 truncate" title={displayVoice}>
              {displayVoice}
            </p>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-1 shrink-0">
          {!isSpeaking && !isLoading ? (
            <button
              id="tts-play-btn"
              onClick={speak}
              disabled={!hasText}
              title="Read aloud"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
              style={{
                background:  hasText ? 'linear-gradient(135deg,#5c7cfa,#4263eb)' : 'rgba(255,255,255,0.04)',
                color:       hasText ? '#fff' : '#495057',
                cursor:      hasText ? 'pointer' : 'not-allowed',
                boxShadow:   hasText ? '0 2px 14px rgba(92,124,250,0.4)' : 'none',
              }}
            >
              <Play className="w-3.5 h-3.5" /> Play
            </button>
          ) : isLoading ? (
            <button
              onClick={stopAudio}
              title="Cancel"
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-dark-400 hover:text-danger-400 hover:bg-danger-500/10 transition-all"
            >
              <Square className="w-3.5 h-3.5" /> Cancel
            </button>
          ) : (
            <>
              <button
                id="tts-pause-btn"
                onClick={togglePause}
                title={isPaused ? 'Resume' : 'Pause'}
                className="p-2 rounded-lg text-dark-400 hover:text-brand-300 hover:bg-white/5 transition-all"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              </button>
              <button
                id="tts-stop-btn"
                onClick={stopAudio}
                title="Stop"
                className="p-2 rounded-lg text-dark-400 hover:text-danger-400 hover:bg-danger-500/10 transition-all"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}

          <button
            id="tts-settings-btn"
            onClick={() => setShowPanel(p => !p)}
            title="Voice settings"
            className="p-2 rounded-lg text-dark-500 hover:text-dark-300 hover:bg-white/5 transition-all"
          >
            <ChevronDown
              className="w-4 h-4 transition-transform duration-200"
              style={{ transform: showPanel ? 'rotate(180deg)' : 'rotate(0deg)' }}
            />
          </button>
        </div>
      </div>

      {/* ── Settings Panel ────────────────────────────────────────────── */}
      {showPanel && (
        <div
          className="glass-card mt-2 p-4 space-y-4 animate-fade-in"
          style={{ borderColor: 'rgba(92,124,250,0.15)' }}
        >

          {/* Voice selector */}
          <div>
            <label className="flex items-center gap-1.5 text-[11px] font-medium text-dark-500 uppercase tracking-wider mb-2">
              <Globe2 className="w-3.5 h-3.5" /> Voice
            </label>

            {/* Auto option */}
            <button
              onClick={() => setVoiceMode('auto')}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl mb-2 text-left transition-all"
              style={{
                background: voiceMode === 'auto' ? 'rgba(92,124,250,0.15)' : 'rgba(255,255,255,0.03)',
                border: voiceMode === 'auto' ? '1px solid rgba(92,124,250,0.3)' : '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <span className="text-lg">🤖</span>
              <div>
                <p className="text-xs font-semibold text-dark-200">Auto-detect language</p>
                <p className="text-[10px] text-dark-500 mt-0.5">
                  {detectedScript
                    ? `Will use ${detectedScript.emoji} ${detectedScript.name} voice`
                    : 'Detects Telugu, Tamil, Hindi, Kannada … from the text'}
                </p>
              </div>
              {voiceMode === 'auto' && (
                <span className="ml-auto text-[10px] text-brand-400 font-medium">Active</span>
              )}
            </button>

            {/* Preset voice list */}
            <div className="grid grid-cols-1 gap-1 max-h-52 overflow-y-auto pr-1">
              {PRESET_VOICES.map(p => (
                <button
                  key={p.label}
                  onClick={() => setVoiceMode(p.label)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-all"
                  style={{
                    background: voiceMode === p.label ? 'rgba(92,124,250,0.15)' : 'rgba(255,255,255,0.02)',
                    border: voiceMode === p.label ? '1px solid rgba(92,124,250,0.25)' : '1px solid rgba(255,255,255,0.04)',
                  }}
                >
                  <span className="text-sm">{p.display.split(' ')[0]}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-[11px] font-medium text-dark-300 truncate">
                      {p.display.slice(p.display.indexOf(' ') + 1)}
                    </p>
                    <p className="text-[9px] text-dark-600 font-mono">{p.label}</p>
                  </div>
                  {voiceMode === p.label && (
                    <span className="text-[10px] text-brand-400 font-medium shrink-0">✓</span>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Speed */}
          <div>
            <label className="flex items-center justify-between text-[11px] font-medium text-dark-500 uppercase tracking-wider mb-2">
              <span className="flex items-center gap-1.5">
                <Gauge className="w-3.5 h-3.5" /> Speed
              </span>
              <span className="text-brand-400 font-mono normal-case">{rate.toFixed(2)}×</span>
            </label>
            <input
              id="tts-rate-slider"
              type="range" min="0.5" max="1.8" step="0.05"
              value={rate}
              onChange={e => setRate(Number(e.target.value))}
              className="tts-slider w-full"
            />
            <div className="flex justify-between text-[10px] text-dark-700 mt-1">
              <span>0.5× Slow</span>
              <span className="text-brand-400">0.92× Clearest</span>
              <span>1.8× Fast</span>
            </div>
          </div>

          {/* Quick presets */}
          <div>
            <p className="text-[11px] font-medium text-dark-500 uppercase tracking-wider mb-2">Speed Presets</p>
            <div className="flex gap-2 flex-wrap">
              {[
                { label: '⭐ Clearest', rate: 0.92 },
                { label: '📖 Calm',     rate: 0.80 },
                { label: '▶️ Normal',    rate: 1.00 },
                { label: '⚡ Fast',      rate: 1.25 },
              ].map(({ label, rate: r }) => {
                const active = Math.abs(rate - r) < 0.01;
                return (
                  <button
                    key={label}
                    onClick={() => setRate(r)}
                    className="px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all"
                    style={{
                      background: active ? 'rgba(92,124,250,0.2)' : 'rgba(255,255,255,0.04)',
                      color:      active ? '#91a7ff' : '#868e96',
                      border:     active ? '1px solid rgba(92,124,250,0.3)' : '1px solid rgba(255,255,255,0.06)',
                    }}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Info footer */}
          <div className="p-3 rounded-xl text-[11px] leading-relaxed"
            style={{ background: 'rgba(92,124,250,0.06)', border: '1px solid rgba(92,124,250,0.12)' }}
          >
            <p className="font-semibold text-dark-300 mb-1">🎙️ Microsoft Edge Neural Voices</p>
            <p className="text-dark-500">
              Using <strong className="text-dark-400">Neural TTS</strong> — same technology as Azure Cognitive Services.
              Crystal-clear for Telugu, Tamil, Hindi, Kannada, Malayalam and 40+ more languages.
              Free &amp; unlimited. Requires internet connection.
            </p>
          </div>
        </div>
      )}

      {/* ── Inline styles ─────────────────────────────────────────────── */}
      <style>{`
        .tts-slider {
          -webkit-appearance: none;
          appearance: none;
          height: 4px;
          border-radius: 999px;
          background: #343a40;
          outline: none;
          cursor: pointer;
          width: 100%;
        }
        .tts-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 15px; height: 15px;
          border-radius: 50%;
          background: linear-gradient(135deg, #5c7cfa, #3bc9db);
          box-shadow: 0 0 8px rgba(92,124,250,0.55);
          cursor: pointer;
          transition: transform 0.15s;
        }
        .tts-slider::-webkit-slider-thumb:hover { transform: scale(1.25); }
        @keyframes tts-wave {
          0%, 100% { transform: scale(1); opacity: 1; }
          50%       { transform: scale(1.2); opacity: 0.85; }
        }
        @keyframes tts-indeterminate {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(250%); }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
