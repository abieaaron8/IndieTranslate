import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

interface SegmentItem {
  id: number;
  sourceText: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
  timestamp: string;
  isEarlyCommit?: boolean;
}

interface StabilityInfo {
  stable_prefix: string;
  unstable_suffix: string;
  stability_score: number;
  stable_token_count: number;
  total_token_count: number;
}

interface EvidenceGateInfo {
  decision: 'WAIT' | 'PREVIEW' | 'COMMIT' | 'REVISE';
  commit_score: number;
  stability_score: number;
  completeness_score: number;
  completeness_state: string;
  completeness_reason: string;
  committed_source: string;
  committed_translation: string;
  uncommitted_source: string;
}

interface LanguagePair {
  id: string;
  sourceCode: string;
  targetCode: string;
  label: string;
  sourceLabel: string;
  targetLabel: string;
}

const LANGUAGE_PAIRS: LanguagePair[] = [
  {
    id: 'ta-en',
    sourceCode: 'ta-IN',
    targetCode: 'en-IN',
    label: 'Tamil → English',
    sourceLabel: 'தமிழ் (Tamil)',
    targetLabel: 'English',
  },
  {
    id: 'hi-en',
    sourceCode: 'hi-IN',
    targetCode: 'en-IN',
    label: 'Hindi → English',
    sourceLabel: 'हिन्दी (Hindi)',
    targetLabel: 'English',
  },
  {
    id: 'en-ta',
    sourceCode: 'en-IN',
    targetCode: 'ta-IN',
    label: 'English → Tamil',
    sourceLabel: 'English',
    targetLabel: 'தமிழ் (Tamil)',
  },
  {
    id: 'en-hi',
    sourceCode: 'en-IN',
    targetCode: 'hi-IN',
    label: 'English → Hindi',
    sourceLabel: 'English',
    targetLabel: 'हिन्दी (Hindi)',
  },
];

export default function App() {
  const [selectedPairId, setSelectedPairId] = useState<string>('ta-en');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Ready to stream');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Partial transcript currently being spoken (SOURCE)
  const [partialTranscript, setPartialTranscript] = useState<string>('');
  // Stability metadata for current partial transcript
  const [stabilityInfo, setStabilityInfo] = useState<StabilityInfo | null>(null);
  // Evidence gate research metadata
  const [evidenceGate, setEvidenceGate] = useState<EvidenceGateInfo | null>(null);
  // Live early committed translation during ongoing utterance
  const [earlyCommitted, setEarlyCommitted] = useState<{ source: string; translation: string } | null>(null);

  // Completed / finalized segments history
  const [history, setHistory] = useState<SegmentItem[]>([]);
  // Audio volume level for visualizer
  const [audioLevel, setAudioLevel] = useState<number>(0);
  // Live TTS speaking status
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // FIFO TTS speech queue and deduplication set
  const speechQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef<boolean>(false);
  const spokenSetRef = useRef<Set<string>>(new Set());

  const activePair = LANGUAGE_PAIRS.find((p) => p.id === selectedPairId) || LANGUAGE_PAIRS[0];

  // Process next item in FIFO speech queue
  const processSpeechQueue = useCallback(() => {
    if (!('speechSynthesis' in window)) return;
    if (isSpeakingRef.current) return;
    if (speechQueueRef.current.length === 0) {
      setIsSpeaking(false);
      return;
    }

    const nextItem = speechQueueRef.current.shift();
    if (!nextItem || !nextItem.text.trim()) {
      processSpeechQueue();
      return;
    }

    isSpeakingRef.current = true;
    setIsSpeaking(true);

    const utterance = new SpeechSynthesisUtterance(nextItem.text.trim());
    utterance.lang = nextItem.lang; // 'ta-IN', 'hi-IN', or 'en-IN'

    // Look for matching voice, fallback to default if unavailable
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find((v) => v.lang === nextItem.lang || v.lang.startsWith(nextItem.lang.slice(0, 2)));
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onend = () => {
      isSpeakingRef.current = false;
      processSpeechQueue();
    };

    utterance.onerror = (e) => {
      console.warn('Speech synthesis error or cancelled:', e);
      isSpeakingRef.current = false;
      processSpeechQueue();
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  // Enqueue a committed translation for sequential speech output
  const enqueueSpeech = useCallback((text: string, targetLang: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    // Deduplication check: do not speak the same committed segment twice
    if (spokenSetRef.current.has(trimmed)) {
      return;
    }
    spokenSetRef.current.add(trimmed);

    speechQueueRef.current.push({ text: trimmed, lang: targetLang });
    processSpeechQueue();
  }, [processSpeechQueue]);

  // Auto scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, partialTranscript, earlyCommitted]);

  const stopStreaming = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    // Cancel any active speech output
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    speechQueueRef.current = [];
    isSpeakingRef.current = false;
    setIsSpeaking(false);

    setIsStreaming(false);
    setAudioLevel(0);
    setStatusMessage('Streaming stopped');
  }, []);

  const startStreaming = async () => {
    setErrorMessage(null);
    setStatusMessage('Requesting microphone access...');

    // Initialize/unlock browser TTS audio context on user gesture
    if ('speechSynthesis' in window) {
      window.speechSynthesis.resume();
    }

    try {
      // 1. Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // 2. Set up Web Audio API Context
      const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // 3. Connect to backend WebSocket
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws/transcribe`;
      setStatusMessage('Connecting to backend...');

      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setStatusMessage(`Connected. Initializing Sarvam STT (${activePair.sourceCode})...`);
        // Send initial source and target language configuration
        ws.send(
          JSON.stringify({
            source_language_code: activePair.sourceCode,
            target_language_code: activePair.targetCode,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'ready') {
            setStatusMessage(`Live Streaming active (${activePair.label})`);
            setIsStreaming(true);
          } else if (data.type === 'translation.commit') {
            // Dedicated committed translation event: send to sequential TTS queue
            if (data.translation) {
              enqueueSpeech(data.translation, data.target_language || activePair.targetCode);
            }
          } else if (data.type === 'transcript.partial') {
            setPartialTranscript(data.text);
            if (data.stability) {
              setStabilityInfo(data.stability);
            }
            if (data.evidence_gate) {
              setEvidenceGate(data.evidence_gate);
              if (data.evidence_gate.committed_source && data.evidence_gate.committed_translation) {
                setEarlyCommitted({
                  source: data.evidence_gate.committed_source,
                  translation: data.evidence_gate.committed_translation,
                });
              } else if (data.evidence_gate.decision === 'REVISE') {
                setEarlyCommitted(null);
              }
            }
          } else if (data.type === 'transcript.final') {
            if (data.text && data.text.trim()) {
              setHistory((prev) => [
                ...prev,
                {
                  id: Date.now() + Math.random(),
                  sourceText: data.text.trim(),
                  translatedText: data.translated_text || '',
                  sourceLang: data.source_language || activePair.sourceCode,
                  targetLang: data.target_language || activePair.targetCode,
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                },
              ]);
            }
            setPartialTranscript('');
            setStabilityInfo(null);
            setEvidenceGate(null);
            setEarlyCommitted(null);
          } else if (data.type === 'error') {
            setErrorMessage(data.message);
            stopStreaming();
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setErrorMessage('WebSocket connection error. Is the FastAPI backend running on port 8000?');
        stopStreaming();
      };

      ws.onclose = () => {
        if (isStreaming) {
          stopStreaming();
        }
      };

      // 4. Set up Web Audio processing & 16kHz resampling
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const bufferSize = 4096;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      const inputSampleRate = audioContext.sampleRate;
      const targetSampleRate = 16000;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputChannelData = e.inputBuffer.getChannelData(0);

        // Compute volume level
        let sum = 0;
        for (let i = 0; i < inputChannelData.length; i++) {
          sum += inputChannelData[i] * inputChannelData[i];
        }
        const rms = Math.sqrt(sum / inputChannelData.length);
        setAudioLevel(Math.min(100, Math.round(rms * 400)));

        // Downsample input audio to 16000Hz PCM 16-bit mono
        const downsampledLength = Math.round((inputChannelData.length * targetSampleRate) / inputSampleRate);
        const pcm16Buffer = new ArrayBuffer(downsampledLength * 2);
        const pcm16View = new DataView(pcm16Buffer);

        for (let i = 0; i < downsampledLength; i++) {
          const originalIndex = (i * inputSampleRate) / targetSampleRate;
          const leftIndex = Math.floor(originalIndex);
          const rightIndex = Math.min(leftIndex + 1, inputChannelData.length - 1);
          const fraction = originalIndex - leftIndex;

          const sample = (1 - fraction) * inputChannelData[leftIndex] + fraction * inputChannelData[rightIndex];
          const clamped = Math.max(-1, Math.min(1, sample));
          const pcmSample = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
          pcm16View.setInt16(i * 2, pcmSample, true);
        }

        wsRef.current.send(pcm16Buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

    } catch (err: unknown) {
      console.error('Failed to start streaming:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Microphone / Setup error: ${msg}`);
      stopStreaming();
    }
  };

  const handleToggleListening = () => {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  };

  const handleClear = () => {
    setHistory([]);
    setPartialTranscript('');
    setStabilityInfo(null);
    setEvidenceGate(null);
    setEarlyCommitted(null);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="badge">HNX26EPS03</div>
          <h1>Live Translation for Indic Languages</h1>
        </div>
        <p className="subtitle">Evidence-Gated Early Indic Translation & Continuous Streaming</p>
      </header>

      {/* Control Bar */}
      <section className="controls-card">
        <div className="language-selector">
          <label>Translation Direction:</label>
          <div className="pair-buttons">
            {LANGUAGE_PAIRS.map((pair) => (
              <button
                key={pair.id}
                type="button"
                className={`pair-btn ${selectedPairId === pair.id ? 'active' : ''}`}
                disabled={isStreaming}
                onClick={() => setSelectedPairId(pair.id)}
              >
                <span className="pair-title">{pair.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="action-buttons">
          <button
            type="button"
            className={`stream-btn ${isStreaming ? 'listening' : ''}`}
            onClick={handleToggleListening}
          >
            <span className={`pulse-dot ${isStreaming ? 'active' : ''}`}></span>
            {isStreaming ? 'Stop Listening' : 'Start Listening'}
          </button>

          <button
            type="button"
            className="clear-btn"
            disabled={history.length === 0 && !partialTranscript && !earlyCommitted}
            onClick={handleClear}
          >
            Clear
          </button>
        </div>
      </section>

      {/* Audio & Connection Status */}
      <section className="status-bar">
        <div className="status-indicator">
          <span className={`status-pill ${isStreaming ? 'connected' : 'idle'}`}></span>
          <span className="status-text">{statusMessage}</span>
        </div>

        {isStreaming && (
          <div className="audio-meter">
            <span className="meter-label">Mic Activity:</span>
            <div className="meter-track">
              <div
                className="meter-fill"
                style={{ width: `${Math.max(5, audioLevel)}%` }}
              ></div>
            </div>
          </div>
        )}
      </section>

      {/* Live Research Indicators Bar */}
      {isStreaming && (
        <section className="research-status-bar">
          <div className="research-item">
            <span className="res-label">Stability:</span>
            <span className="res-val">
              {stabilityInfo ? `${Math.round(stabilityInfo.stability_score * 100)}%` : '0%'}
            </span>
          </div>
          <div className="research-item">
            <span className="res-label">Completeness:</span>
            <span className="res-val">
              {evidenceGate ? `${Math.round(evidenceGate.completeness_score * 100)}% (${evidenceGate.completeness_state})` : '0%'}
            </span>
          </div>
          <div className="research-item">
            <span className="res-label">Decision:</span>
            <span className={`res-badge ${evidenceGate?.decision ? evidenceGate.decision.toLowerCase() : 'wait'}`}>
              {evidenceGate?.decision || 'WAIT'}
            </span>
          </div>
          <div className="research-item">
            <span className={`audio-badge ${isSpeaking ? 'speaking' : 'ready'}`}>
              {isSpeaking ? '🔊 Speaking...' : '🎧 Earphone Audio Ready'}
            </span>
          </div>
        </section>
      )}

      {errorMessage && (
        <div className="error-banner">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {/* Main Translation Container with SOURCE and TRANSLATION sections */}
      <main className="translation-workspace">
        {/* Left: SOURCE Transcript Area */}
        <section className="feed-panel source-panel">
          <div className="panel-header">
            <div className="panel-title-group">
              <span className="panel-tag source-tag">SOURCE</span>
              <h2>{activePair.sourceLabel}</h2>
            </div>
            <span className="live-indicator">
              {isStreaming ? '● LISTENING' : 'IDLE'}
            </span>
          </div>

          <div className="panel-body">
            {history.length === 0 && !partialTranscript && !earlyCommitted ? (
              <div className="empty-feed">
                <p>Spoken audio and live partial transcript will appear here.</p>
              </div>
            ) : (
              <div className="feed-list">
                {history.map((item) => (
                  <div key={item.id} className="feed-card source-card">
                    <span className="feed-time">{item.timestamp}</span>
                    <p className="feed-text">{item.sourceText}</p>
                  </div>
                ))}

                {/* Active Utterance (Committed source + Uncommitted tail) */}
                {partialTranscript && (
                  <div className="feed-card partial-card">
                    <div className="partial-meta">
                      <span className="badge-partial">Live Utterance</span>
                      {evidenceGate && (
                        <span className={`badge-decision ${evidenceGate.decision.toLowerCase()}`}>
                          {evidenceGate.decision}
                        </span>
                      )}
                    </div>

                    <p className="feed-text partial-text">
                      {earlyCommitted ? (
                        <>
                          <span className="committed-source-part">{earlyCommitted.source}</span>
                          <span className="uncommitted-source-part">
                            {partialTranscript.slice(earlyCommitted.source.length)}
                          </span>
                        </>
                      ) : (
                        partialTranscript
                      )}
                      <span className="cursor-blink">|</span>
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Right: TRANSLATION Output Area */}
        <section className="feed-panel translation-panel">
          <div className="panel-header">
            <div className="panel-title-group">
              <span className="panel-tag translation-tag">TRANSLATION</span>
              <h2>{activePair.targetLabel}</h2>
            </div>
            <span className="final-indicator">COMMITTED / FINAL</span>
          </div>

          <div className="panel-body">
            {history.length === 0 && !earlyCommitted ? (
              <div className="empty-feed">
                <p>Translated committed segments will appear here in real time.</p>
              </div>
            ) : (
              <div className="feed-list">
                {history.map((item) => (
                  <div key={item.id} className="feed-card translation-card">
                    <span className="feed-time">{item.timestamp}</span>
                    <p className="feed-text translated-text">
                      {item.translatedText || '—'}
                    </p>
                  </div>
                ))}

                {/* Live Early Committed Translation for ongoing utterance */}
                {earlyCommitted && (
                  <div className="feed-card early-committed-card">
                    <div className="partial-meta">
                      <span className="badge-early-commit">Early Committed</span>
                    </div>
                    <p className="feed-text translated-text">
                      {earlyCommitted.translation}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
        <div ref={transcriptEndRef} />
      </main>
    </div>
  );
}
