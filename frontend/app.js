/* eslint-disable no-console */

const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/audio`;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const englishEl = document.getElementById('english');
const hindiEl = document.getElementById('hindi');
const audioEl = document.getElementById('audio');
const errorsEl = document.getElementById('errors');

const latAsr = document.getElementById('lat-asr');
const latTr = document.getElementById('lat-tr');
const latTts = document.getElementById('lat-tts');
const latTotal = document.getElementById('lat-total');

let ws = null;
let audioCtx = null;
let micStream = null;
let workletNode = null;
let micSource = null;

// PCM batching: AudioWorklet posts ~128-sample blocks; we coalesce up to
// ~40 ms of audio per WebSocket frame to reduce per-message overhead.
let pcmBatch = [];
let pcmBatchSamples = 0;
const TARGET_SAMPLES_PER_FRAME = 640; // 40 ms at 16 kHz

// MediaSource state.
let mediaSource = null;
let sourceBuffer = null;
let appendQueue = [];
let mediaReady = false;

function setStatus(text, cls) {
  statusText.textContent = text;
  statusEl.className = 'status' + (cls ? ' ' + cls : '');
}

function showError(msg) {
  errorsEl.textContent = msg;
  if (msg) setTimeout(() => { if (errorsEl.textContent === msg) errorsEl.textContent = ''; }, 5000);
}

function resetMediaSource() {
  appendQueue = [];
  mediaReady = false;
  if (sourceBuffer) {
    try { sourceBuffer.abort(); } catch (_) {}
  }
  sourceBuffer = null;
  if (mediaSource && mediaSource.readyState === 'open') {
    try { mediaSource.endOfStream(); } catch (_) {}
  }
  mediaSource = new MediaSource();
  audioEl.src = URL.createObjectURL(mediaSource);
  mediaSource.addEventListener('sourceopen', () => {
    try {
      sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
      sourceBuffer.mode = 'sequence';
      sourceBuffer.addEventListener('updateend', flushAppendQueue);
      mediaReady = true;
      flushAppendQueue();
    } catch (e) {
      showError('MediaSource setup failed: ' + e.message);
    }
  }, { once: true });
}

function enqueueAudio(buf) {
  appendQueue.push(buf);
  flushAppendQueue();
}

function flushAppendQueue() {
  if (!mediaReady || !sourceBuffer || sourceBuffer.updating) return;
  if (appendQueue.length === 0) return;
  const next = appendQueue.shift();
  try {
    sourceBuffer.appendBuffer(next);
  } catch (e) {
    if (e.name === 'QuotaExceededError') {
      try { sourceBuffer.remove(0, audioEl.currentTime - 1); } catch (_) {}
      appendQueue.unshift(next);
    } else {
      console.warn('appendBuffer failed', e);
    }
  }
  if (audioEl.paused) audioEl.play().catch(() => {});
}

function sendPcmBatchNow() {
  if (pcmBatchSamples === 0 || !ws || ws.readyState !== WebSocket.OPEN) {
    pcmBatch = [];
    pcmBatchSamples = 0;
    return;
  }
  const merged = new Int16Array(pcmBatchSamples);
  let offset = 0;
  for (const chunk of pcmBatch) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  ws.send(merged.buffer);
  pcmBatch = [];
  pcmBatchSamples = 0;
}

async function start() {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  englishEl.textContent = '';
  hindiEl.textContent = '';
  latAsr.textContent = latTr.textContent = latTts.textContent = latTotal.textContent = '-';
  errorsEl.textContent = '';
  setStatus('connecting...', '');

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    showError('Mic access denied: ' + e.message);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus('idle', '');
    return;
  }

  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  } catch (_) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.sampleRate !== 16000) {
    console.warn(`AudioContext sample rate is ${audioCtx.sampleRate}; server expects 16000.`);
    showError(`Your browser captures at ${audioCtx.sampleRate}Hz; VAD may misbehave. Try Chrome.`);
  }

  await audioCtx.audioWorklet.addModule('/static/pcm-worklet.js');
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-worklet');
  workletNode.port.onmessage = (e) => {
    const int16 = new Int16Array(e.data);
    pcmBatch.push(int16);
    pcmBatchSamples += int16.length;
    if (pcmBatchSamples >= TARGET_SAMPLES_PER_FRAME) sendPcmBatchNow();
  };
  micSource = audioCtx.createMediaStreamSource(micStream);
  micSource.connect(workletNode);

  resetMediaSource();

  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setStatus('recording', 'recording');
    ws.send(JSON.stringify({ type: 'start' }));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      let payload;
      try { payload = JSON.parse(ev.data); } catch { return; }
      handleEvent(payload);
    } else {
      enqueueAudio(new Uint8Array(ev.data));
    }
  };
  ws.onerror = () => showError('WebSocket error');
  ws.onclose = () => {
    setStatus('disconnected', '');
    teardownMic();
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };
}

function handleEvent(payload) {
  switch (payload.type) {
    case 'ready':
      console.log('Server ready', payload);
      break;
    case 'partial_transcript':
      englishEl.textContent = payload.text;
      hindiEl.textContent = '...';
      break;
    case 'translation':
      englishEl.textContent = payload.english;
      hindiEl.textContent = payload.hindi || '(filler/skipped)';
      setStatus('playing', 'playing');
      break;
    case 'stop_playback':
      console.log('stop_playback', payload.reason);
      resetMediaSource();
      setStatus('recording', 'recording');
      break;
    case 'latency':
      latAsr.textContent = payload.asr_ms;
      latTr.textContent = payload.translate_ms;
      latTts.textContent = payload.tts_first_chunk_ms;
      latTotal.textContent = payload.total_ms;
      break;
    case 'error':
      showError(`[${payload.stage || 'server'}] ${payload.message}`);
      break;
    default:
      console.debug('unknown event', payload);
  }
}

function teardownMic() {
  try { workletNode && workletNode.disconnect(); } catch (_) {}
  try { micSource && micSource.disconnect(); } catch (_) {}
  try { audioCtx && audioCtx.close(); } catch (_) {}
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  workletNode = null;
  micSource = null;
  audioCtx = null;
  micStream = null;
}

function stop() {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('idle', '');
  sendPcmBatchNow();
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: 'stop' })); } catch (_) {}
    try { ws.close(); } catch (_) {}
  }
  teardownMic();
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
audioEl.addEventListener('ended', () => {
  if (stopBtn.disabled) setStatus('idle', '');
  else setStatus('recording', 'recording');
});
