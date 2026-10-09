import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

export interface LanguageOption {
  code: string;
  name: string;
  native: string;
  label: string;
  shortLabel: string;
}

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: 'en-IN', name: 'English', native: 'English', label: 'English — English', shortLabel: 'English (English)' },
  { code: 'ta-IN', name: 'Tamil', native: 'தமிழ்', label: 'Tamil — தமிழ்', shortLabel: 'Tamil (தமிழ்)' },
  { code: 'hi-IN', name: 'Hindi', native: 'हिन्दी', label: 'Hindi — हिन्दी', shortLabel: 'Hindi (हिन्दी)' },
  { code: 'te-IN', name: 'Telugu', native: 'తెలుగు', label: 'Telugu — తెలుగు', shortLabel: 'Telugu (తెలుగు)' },
  { code: 'kn-IN', name: 'Kannada', native: 'ಕನ್ನಡ', label: 'Kannada — ಕನ್ನಡ', shortLabel: 'Kannada (ಕನ್ನಡ)' },
  { code: 'ml-IN', name: 'Malayalam', native: 'മലയാളം', label: 'Malayalam — മലയാളം', shortLabel: 'Malayalam (മലയാളം)' },
];

export interface ChatMessage {
  id: string; // Deterministic, persistent utterance identifier (e.g. "msg_you_utt_0")
  utteranceIndex: number;
  sender: 'you' | 'peer';
  originalText?: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
  timestamp: string;
}

export interface ActiveUtterance {
  utteranceIndex: number;
  sender: 'you' | 'peer';
  sourceText: string;
  previewTranslation: string;
  sourceLang: string;
  targetLang: string;
  status: 'listening' | 'processing' | 'committed';
}

export default function App() {
  // Navigation: 'lobby' | 'room'
  const [view, setView] = useState<'lobby' | 'room'>('lobby');

  // Language state
  const [myLanguage, setMyLanguage] = useState<string>('en-IN');
  const [peerLanguage, setPeerLanguage] = useState<string | null>(null);
  const [peerConnected, setPeerConnected] = useState<boolean>(false);

  // Role & Session
  const [role, setRole] = useState<'Host (Participant A)' | 'Guest (Participant B)'>('Host (Participant A)');
  const [roomId, setRoomId] = useState<string>('');
  const [inputRoomCode, setInputRoomCode] = useState<string>('');
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  // Streaming / Mic
  const [isSpeakingMic, setIsSpeakingMic] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Dedicated Active Utterance (in-progress live preview without flickering)
  const [activeSpeakerUtterance, setActiveSpeakerUtterance] = useState<ActiveUtterance | null>(null);

  // Finalized Conversation History
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(true);

  // Refs for fresh state access in async & audio callbacks
  const isSpeakingMicRef = useRef<boolean>(false);
  const myLanguageRef = useRef<string>(myLanguage);
  const peerLanguageRef = useRef<string | null>(peerLanguage);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  // TTS Queue & deduplication
  const speechQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isPlayingTtsRef = useRef<boolean>(false);
  const spokenSetRef = useRef<Set<string>>(new Set());
  const spokenCommitIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    myLanguageRef.current = myLanguage;
  }, [myLanguage]);

  useEffect(() => {
    peerLanguageRef.current = peerLanguage;
  }, [peerLanguage]);

  const currentLangObj = SUPPORTED_LANGUAGES.find((l) => l.code === myLanguage) || SUPPORTED_LANGUAGES[0];
  const peerLangObj = SUPPORTED_LANGUAGES.find((l) => l.code === peerLanguage);

  // URL query parameter check
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionParam = params.get('session');
    if (sessionParam) {
      setInputRoomCode(sessionParam.trim().toUpperCase());
    }
  }, []);

  // Smooth auto-scroll when message count changes or active utterance starts/stops
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [messages.length, Boolean(activeSpeakerUtterance)]);

  // FIFO TTS queue
  const processSpeechQueue = useCallback(() => {
    if (!ttsEnabled || !('speechSynthesis' in window)) return;
    if (isPlayingTtsRef.current) return;
    if (speechQueueRef.current.length === 0) return;

    const nextItem = speechQueueRef.current.shift();
    if (!nextItem || !nextItem.text.trim()) {
      processSpeechQueue();
      return;
    }

    isPlayingTtsRef.current = true;
    const utterance = new SpeechSynthesisUtterance(nextItem.text.trim());
    utterance.lang = nextItem.lang;

    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(
      (v) => v.lang === nextItem.lang || v.lang.startsWith(nextItem.lang.slice(0, 2))
    );
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onend = () => {
      isPlayingTtsRef.current = false;
      processSpeechQueue();
    };

    utterance.onerror = (e) => {
      console.warn('TTS playback error:', e);
      isPlayingTtsRef.current = false;
      processSpeechQueue();
    };

    window.speechSynthesis.speak(utterance);
  }, [ttsEnabled]);

  const speakText = useCallback(
    (text: string, langCode: string) => {
      if (!ttsEnabled) return;
      const trimmed = text.trim();
      if (!trimmed) return;

      if (spokenSetRef.current.has(trimmed)) return;
      spokenSetRef.current.add(trimmed);

      speechQueueRef.current.push({ text: trimmed, lang: langCode });
      processSpeechQueue();
    },
    [processSpeechQueue, ttsEnabled]
  );

  const replayMessageTts = (text: string, langCode: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = langCode;
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(
      (v) => v.lang === langCode || v.lang.startsWith(langCode.slice(0, 2))
    );
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }
    window.speechSynthesis.speak(utterance);
  };

  // Teardown
  const stopAudioCapture = useCallback(() => {
    isSpeakingMicRef.current = false;
    setIsSpeakingMic(false);
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch {}
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {}
      sourceRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const leaveRoom = useCallback(() => {
    stopAudioCapture();
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'leave_session' }));
        wsRef.current.close();
      }
      wsRef.current = null;
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    speechQueueRef.current = [];
    spokenSetRef.current.clear();
    spokenCommitIdsRef.current.clear();
    setRoomId('');
    setPeerConnected(false);
    setPeerLanguage(null);
    setMessages([]);
    setActiveSpeakerUtterance(null);
    setView('lobby');
  }, [stopAudioCapture]);

  // Connect & Join
  const connectAndJoin = (action: 'create_session' | 'join_session', targetCode?: string) => {
    setErrorMessage(null);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/session`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      const payload: Record<string, string> = {
        action,
        language: myLanguageRef.current,
      };
      if (action === 'join_session' && targetCode) {
        payload.session_id = targetCode.trim().toUpperCase();
      }
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'session.created':
            setRoomId(data.session_id);
            setRole('Host (Participant A)');
            setView('room');
            break;

          case 'session.joined':
            setRoomId(data.session_id);
            setRole('Guest (Participant B)');
            setPeerConnected(Boolean(data.peer_connected));
            if (data.other_language) {
              setPeerLanguage(data.other_language);
            }
            setView('room');
            break;

          case 'peer.connected':
            setPeerConnected(true);
            setPeerLanguage(data.language);
            break;

          case 'peer.disconnected':
            setPeerConnected(false);
            break;

          case 'peer.language_changed':
            setPeerLanguage(data.language);
            break;

          case 'listening.started':
            isSpeakingMicRef.current = true;
            setIsSpeakingMic(true);
            setActiveSpeakerUtterance({
              utteranceIndex: 0,
              sender: 'you',
              sourceText: '',
              previewTranslation: '',
              sourceLang: myLanguageRef.current,
              targetLang: peerLanguageRef.current || (myLanguageRef.current.startsWith('en') ? 'ta-IN' : 'en-IN'),
              status: 'listening',
            });
            break;

          case 'listening.stopped':
            isSpeakingMicRef.current = false;
            setIsSpeakingMic(false);
            setActiveSpeakerUtterance((prev) => (prev ? { ...prev, status: 'processing' } : null));
            break;

          case 'transcript.partial': {
            const currentText = data.text || '';
            const uttIdx = data.utterance_idx ?? 0;
            const decision = data.evidence_gate?.decision || 'WAIT';
            const committedTranslation = data.evidence_gate?.committed_translation || '';

            // Update the persistent active utterance in place (no new bubble spawned!)
            setActiveSpeakerUtterance({
              utteranceIndex: uttIdx,
              sender: 'you',
              sourceText: currentText,
              previewTranslation: committedTranslation,
              sourceLang: data.language || myLanguageRef.current,
              targetLang: data.target_language || peerLanguageRef.current || (myLanguageRef.current.startsWith('en') ? 'ta-IN' : 'en-IN'),
              status: decision === 'COMMIT' ? 'committed' : (isSpeakingMicRef.current ? 'listening' : 'processing'),
            });
            break;
          }

          case 'transcript.final': {
            const cleanText = data.text?.trim() || '';
            const transText = data.translated_text?.trim() || cleanText;
            const uttIdx = data.utterance_idx ?? 0;
            const stableId = `msg_you_utt_${uttIdx}`;

            if (cleanText) {
              setMessages((prev) => {
                const existingIndex = prev.findIndex((m) => m.id === stableId);
                const newMsg: ChatMessage = {
                  id: stableId,
                  utteranceIndex: uttIdx,
                  sender: 'you',
                  originalText: cleanText,
                  translatedText: transText,
                  sourceLang: data.source_language || myLanguageRef.current,
                  targetLang: data.target_language || peerLanguageRef.current || (myLanguageRef.current.startsWith('en') ? 'ta-IN' : 'en-IN'),
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                };
                if (existingIndex !== -1) {
                  const next = [...prev];
                  next[existingIndex] = newMsg;
                  return next;
                }
                return [...prev, newMsg];
              });
            }

            // Smoothly finalize active utterance without flash
            setActiveSpeakerUtterance(null);
            break;
          }

          case 'translation.commit': {
            const transText = data.translation?.trim() || '';
            const origText = data.original?.trim() || '';
            const uttIdx = data.utterance_index ?? 0;
            const stableId = `msg_peer_utt_${uttIdx}`;

            if (transText || origText) {
              setMessages((prev) => {
                const existingIndex = prev.findIndex((m) => m.id === stableId);
                const newMsg: ChatMessage = {
                  id: stableId,
                  utteranceIndex: uttIdx,
                  sender: 'peer',
                  originalText: origText,
                  translatedText: transText || origText,
                  sourceLang: data.source_language || peerLanguageRef.current || '',
                  targetLang: data.target_language || myLanguageRef.current,
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                };
                if (existingIndex !== -1) {
                  const next = [...prev];
                  next[existingIndex] = newMsg;
                  return next;
                }
                return [...prev, newMsg];
              });

              // Trigger TTS once per unique utterance commit (never duplicate TTS)
              const ttsKey = `${stableId}_${transText}`;
              if (!spokenCommitIdsRef.current.has(ttsKey)) {
                spokenCommitIdsRef.current.add(ttsKey);
                speakText(transText, data.target_language || myLanguageRef.current);
              }
            }
            break;
          }

          case 'error':
            setErrorMessage(data.message);
            stopAudioCapture();
            break;
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket connection error:', err);
      setErrorMessage('Failed to connect to backend server. Make sure the backend is running on port 8000.');
      stopAudioCapture();
    };

    ws.onclose = () => {
      stopAudioCapture();
    };
  };

  const handleCreateRoom = () => {
    connectAndJoin('create_session');
  };

  const handleJoinRoom = () => {
    if (!inputRoomCode.trim()) {
      setErrorMessage('Please enter a conversation code.');
      return;
    }
    connectAndJoin('join_session', inputRoomCode.trim().toUpperCase());
  };

  const handleChangeLanguage = (newLangCode: string) => {
    setMyLanguage(newLangCode);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && view === 'room') {
      wsRef.current.send(
        JSON.stringify({
          action: 'change_language',
          language: newLangCode,
        })
      );
    }
  };

  const handleToggleSpeaking = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErrorMessage('Not connected to a conversation.');
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.resume();
    }

    if (isSpeakingMicRef.current) {
      isSpeakingMicRef.current = false;
      setIsSpeakingMic(false);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'stop_listening' }));
      }
      stopAudioCapture();
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: { ideal: 16000 },
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });
        streamRef.current = stream;

        let audioContext: AudioContext;
        const AudioContextClass =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        try {
          audioContext = new AudioContextClass({ sampleRate: 16000 });
        } catch {
          audioContext = new AudioContextClass();
        }
        audioContextRef.current = audioContext;

        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        const bufferSize = 4096;
        const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        processorRef.current = processor;

        const inputSampleRate = audioContext.sampleRate;
        const targetSampleRate = 16000;

        processor.onaudioprocess = (e) => {
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
          if (!isSpeakingMicRef.current) return;

          const inputChannelData = e.inputBuffer.getChannelData(0);

          if (inputSampleRate === targetSampleRate) {
            const pcm16Buffer = new ArrayBuffer(inputChannelData.length * 2);
            const pcm16View = new DataView(pcm16Buffer);
            for (let i = 0; i < inputChannelData.length; i++) {
              const clamped = Math.max(-1, Math.min(1, inputChannelData[i]));
              const pcmSample = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
              pcm16View.setInt16(i * 2, pcmSample, true);
            }
            wsRef.current.send(pcm16Buffer);
          } else {
            const downsampledLength = Math.round(
              (inputChannelData.length * targetSampleRate) / inputSampleRate
            );
            const pcm16Buffer = new ArrayBuffer(downsampledLength * 2);
            const pcm16View = new DataView(pcm16Buffer);

            for (let i = 0; i < downsampledLength; i++) {
              const originalIndex = (i * inputSampleRate) / targetSampleRate;
              const leftIndex = Math.floor(originalIndex);
              const rightIndex = Math.min(leftIndex + 1, inputChannelData.length - 1);
              const fraction = originalIndex - leftIndex;

              const sample =
                (1 - fraction) * inputChannelData[leftIndex] +
                fraction * inputChannelData[rightIndex];
              const clamped = Math.max(-1, Math.min(1, sample));
              const pcmSample = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
              pcm16View.setInt16(i * 2, pcmSample, true);
            }
            wsRef.current.send(pcm16Buffer);
          }
        };

        // Notify backend to start listening before connecting audio graph
        wsRef.current.send(JSON.stringify({ action: 'start_listening' }));
        isSpeakingMicRef.current = true;
        setIsSpeakingMic(true);

        source.connect(processor);
        processor.connect(audioContext.destination);
      } catch (err: unknown) {
        console.error('Microphone access failed:', err);
        const msg = err instanceof Error ? err.message : String(err);
        setErrorMessage(`Microphone access error: ${msg}`);
        stopAudioCapture();
      }
    }
  };

  const copyRoomCode = () => {
    if (!roomId) return;
    navigator.clipboard.writeText(roomId).then(() => {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    });
  };

  const shareRoomLink = () => {
    if (!roomId) return;
    const url = `${window.location.protocol}//${window.location.host}/?session=${roomId}`;
    navigator.clipboard.writeText(url).then(() => {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    });
  };

  const handleClearMessages = () => {
    setMessages([]);
    setActiveSpeakerUtterance(null);
    spokenCommitIdsRef.current.clear();
  };

  // ── RENDER LOBBY (MATCHING SCREENSHOT 1) ──────────────────────────────────
  if (view === 'lobby') {
    return (
      <div className="app-viewport">
        {/* Top Header */}
        <header className="main-header">
          <h1 className="app-brand-title">IndieTranslate</h1>
          <p className="app-brand-sub">Two-Device Bidirectional Live Translation for Indic Languages</p>
        </header>

        {/* Main Central Card */}
        <main className="lobby-card-wrapper">
          <div className="lobby-main-card">
            <div className="card-top-action">
              <button type="button" className="btn-back-modes">
                ← Back to Modes
              </button>
            </div>

            <h2 className="card-title">Start a Live Conversation</h2>
            <p className="card-desc">
              Connect two devices for real-time bilingual translation. Speak in your language, and the other person hears and reads in theirs.
            </p>

            {/* Language Selection Dropdown */}
            <div className="field-group">
              <label className="field-label">MY LANGUAGE</label>
              <div className="select-container">
                <select
                  className="custom-dropdown"
                  value={myLanguage}
                  onChange={(e) => setMyLanguage(e.target.value)}
                >
                  {SUPPORTED_LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label}
                    </option>
                  ))}
                </select>
                <span className="dropdown-arrow">▼</span>
              </div>
            </div>

            {/* Choose Action Divider */}
            <div className="section-divider">
              <span className="divider-text">CHOOSE ACTION</span>
            </div>

            {/* Two Action Columns */}
            <div className="action-cards-grid">
              {/* Card 1: Start New Session */}
              <div className="sub-action-card">
                <h3 className="sub-card-title">Start a New Session</h3>
                <p className="sub-card-desc">
                  Create a conversation code and share it with the other person.
                </p>
                <button
                  type="button"
                  className="btn-create-conversation"
                  onClick={handleCreateRoom}
                >
                  Create Conversation
                </button>
              </div>

              {/* Card 2: Join Existing Session */}
              <div className="sub-action-card">
                <h3 className="sub-card-title">Join Existing Session</h3>
                <p className="sub-card-desc">
                  Enter the 6-character code shared by the other person.
                </p>
                <div className="join-form">
                  <input
                    type="text"
                    maxLength={6}
                    placeholder="CONVERSATION CODE (E."
                    value={inputRoomCode}
                    onChange={(e) => setInputRoomCode(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleJoinRoom()}
                    className="code-input-box"
                  />
                  <button
                    type="button"
                    className="btn-join-conversation"
                    onClick={handleJoinRoom}
                  >
                    Join Conversation
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>

        {errorMessage && (
          <div className="error-toast" onClick={() => setErrorMessage(null)}>
            <span>⚠️ {errorMessage}</span>
          </div>
        )}
      </div>
    );
  }

  // ── RENDER LIVE CONVERSATION ROOM (MATCHING SCREENSHOT 2) ────────────────
  return (
    <div className="app-viewport room-viewport">
      {/* Top Header */}
      <header className="main-header room-main-header">
        <h1 className="app-brand-title">IndieTranslate</h1>
        <p className="app-brand-sub">Two-Device Bidirectional Live Translation for Indic Languages</p>
      </header>

      <div className="room-layout-container">
        {/* Top Control Panel Card */}
        <section className="room-control-card">
          {/* Metadata Grid */}
          <div className="control-grid">
            {/* Conversation Code */}
            <div className="grid-cell conversation-cell">
              <span className="cell-label">CONVERSATION</span>
              <div className="code-badge-pill">{roomId}</div>
              <button type="button" className="btn-copy-sub" onClick={copyRoomCode}>
                {copiedCode ? 'Copied!' : 'Copy Code'}
              </button>
            </div>

            {/* My Language Dropdown */}
            <div className="grid-cell language-cell">
              <span className="cell-label">MY LANGUAGE</span>
              <div className="select-container compact-select">
                <select
                  className="custom-dropdown compact-dropdown"
                  value={myLanguage}
                  onChange={(e) => handleChangeLanguage(e.target.value)}
                  disabled={isSpeakingMic}
                >
                  {SUPPORTED_LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.shortLabel}
                    </option>
                  ))}
                </select>
                <span className="dropdown-arrow">▼</span>
              </div>
            </div>

            {/* Role */}
            <div className="grid-cell role-cell">
              <span className="cell-label">ROLE</span>
              <div className="role-badge-box">
                <span className="role-text">{role}</span>
              </div>
            </div>

            {/* Other Person */}
            <div className="grid-cell other-person-cell">
              <span className="cell-label">OTHER PERSON</span>
              <div className="other-person-status">
                {peerConnected && peerLangObj ? (
                  <span className="peer-connected-text">{peerLangObj.shortLabel}</span>
                ) : (
                  <span className="peer-waiting-text">Waiting to join...</span>
                )}
              </div>
            </div>

            {/* Status */}
            <div className="grid-cell status-cell">
              <span className="cell-label">STATUS</span>
              <div className="live-status-row">
                <span className={`status-bulb ${peerConnected ? 'bulb-green' : 'bulb-orange'}`}></span>
                <span className="status-caption">
                  {peerConnected ? 'Connected with peer' : 'Waiting for other person to join...'}
                </span>
              </div>
            </div>
          </div>

          {/* Microphone Status Row */}
          <div className="mic-status-row">
            <span className="cell-label">MICROPHONE</span>
            <div className="mic-indicator-text">
              <span className={`mic-dot ${isSpeakingMic ? 'mic-dot-active' : 'mic-dot-idle'}`}></span>
              <span>{isSpeakingMic ? 'Listening...' : 'Mic Ready'}</span>
            </div>
          </div>

          {/* Action Buttons Row */}
          <div className="room-actions-bar">
            <button
              type="button"
              className={`btn-speak-main ${isSpeakingMic ? 'speaking-active' : ''}`}
              onClick={handleToggleSpeaking}
            >
              <span className="speak-dot">●</span>
              <span>{isSpeakingMic ? 'Stop Speaking' : 'Speak'}</span>
            </button>

            <button
              type="button"
              className="btn-ghost"
              onClick={handleClearMessages}
              disabled={messages.length === 0 && !activeSpeakerUtterance}
            >
              Clear
            </button>

            <button type="button" className="btn-ghost" onClick={leaveRoom}>
              Leave Conversation
            </button>

            <button type="button" className="btn-ghost btn-share-room" onClick={shareRoomLink}>
              🔗 Share Link
            </button>

            <button
              type="button"
              className={`btn-ghost btn-tts-icon ${ttsEnabled ? 'tts-on' : 'tts-off'}`}
              onClick={() => setTtsEnabled(!ttsEnabled)}
              title={ttsEnabled ? 'TTS Audio Enabled' : 'TTS Audio Muted'}
            >
              {ttsEnabled ? '🔊 Audio' : '🔇 Muted'}
            </button>
          </div>
        </section>

        {/* Live Conversation Event / Feed Box */}
        <section className="live-feed-card">
          <div className="feed-card-header">
            <h3 className="feed-title">Live Event</h3>
            <div className="feed-lang-badge">
              {currentLangObj.name} {peerConnected && peerLangObj ? `→ ${peerLangObj.name}` : '(Waiting for peer)'}
            </div>
          </div>

          <div className="feed-card-body">
            {messages.length === 0 && !activeSpeakerUtterance ? (
              <div className="empty-feed-placeholder">
                <p className="no-messages-title">
                  {isSpeakingMic ? 'Listening... Speak now.' : 'No messages yet.'}
                </p>
                <p className="no-messages-sub">
                  {peerConnected
                    ? 'Click "Speak" to start live bilingual translation.'
                    : `Share code "${roomId}" with the second participant to begin.`}
                </p>
              </div>
            ) : (
              <div className="messages-scroll-area">
                {/* Completed Messages History with Stable React Keys */}
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`message-row ${msg.sender === 'you' ? 'msg-sent' : 'msg-received'}`}
                  >
                    <div className="message-bubble">
                      <div className="msg-meta">
                        <span className="msg-sender">
                          {msg.sender === 'you' ? 'You' : `Peer (${msg.sourceLang})`}
                        </span>
                        <span className="msg-timestamp">{msg.timestamp}</span>
                      </div>

                      <p className="msg-primary-text">{msg.translatedText || msg.originalText}</p>

                      {msg.originalText && msg.translatedText && msg.originalText !== msg.translatedText && (
                        <p className="msg-secondary-text">
                          <span className="orig-tag">Original:</span> {msg.originalText}
                        </p>
                      )}

                      {msg.sender === 'peer' && (
                        <button
                          type="button"
                          className="btn-replay-tts"
                          onClick={() => replayMessageTts(msg.translatedText, msg.targetLang)}
                        >
                          🔊 Replay Audio
                        </button>
                      )}
                    </div>
                  </div>
                ))}

                {/* Persistent In-Place Active Utterance Preview (Speaker) */}
                {activeSpeakerUtterance && (activeSpeakerUtterance.sourceText || isSpeakingMic) && (
                  <div className="message-row msg-sent">
                    <div className="message-bubble active-utterance-bubble">
                      <div className="msg-meta">
                        <span className="msg-sender">
                          You ({activeSpeakerUtterance.status === 'listening' ? 'Speaking...' : activeSpeakerUtterance.status === 'committed' ? 'Committed' : 'Processing...'})
                        </span>
                        <span className={`status-pill ${activeSpeakerUtterance.status}`}>
                          {activeSpeakerUtterance.status}
                        </span>
                      </div>

                      <p className="msg-primary-text active-primary-text">
                        {activeSpeakerUtterance.previewTranslation || activeSpeakerUtterance.sourceText || (
                          <span className="typing-placeholder">Start speaking...</span>
                        )}
                        {activeSpeakerUtterance.status === 'listening' && (
                          <span className="cursor-indicator">|</span>
                        )}
                      </p>

                      {activeSpeakerUtterance.sourceText && activeSpeakerUtterance.previewTranslation && activeSpeakerUtterance.sourceText !== activeSpeakerUtterance.previewTranslation && (
                        <p className="msg-secondary-text">
                          <span className="orig-tag">Original:</span> {activeSpeakerUtterance.sourceText}
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div ref={chatBottomRef} />
              </div>
            )}
          </div>
        </section>
      </div>

      {errorMessage && (
        <div className="error-toast" onClick={() => setErrorMessage(null)}>
          <span>⚠️ {errorMessage}</span>
        </div>
      )}
    </div>
  );
}
