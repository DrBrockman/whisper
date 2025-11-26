import React, { useState, useRef, useEffect } from "react";

// Initialize Web Worker for transcription
let transcriptionWorker = null;
let messageCounter = 0;

const getTranscriptionWorker = () => {
  if (transcriptionWorker === null) {
    transcriptionWorker = new Worker(
      new URL("./transcriptionWorker.js", import.meta.url),
      { type: "module" }
    );
  }
  return transcriptionWorker;
};

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState("ready");
  const [transcription, setTranscription] = useState("");

  // --- Refs ---
  const mediaRecorderRef = useRef(null);
  const pointerActiveRef = useRef(false);
  const pendingTranscriptionRef = useRef(new Map());

  // --- Worker Message Handler ---
  useEffect(() => {
    const worker = getTranscriptionWorker();

    const handleWorkerMessage = (event) => {
      const msg = event.data || {};
      // Log entire worker message for easier debugging
      console.log("[App] Worker message:", msg);

      // Support both `type` (older) and `status` (worker) fields
      const status = msg.status || msg.type || null;
      const messageId = msg.messageId || msg.id || null;

      switch (status) {
        case "initiate":
          // model file loading started
          console.log(
            "[App] Worker initiating model load:",
            msg.file || msg.model
          );
          break;
        case "progress":
          // progress updates forwarded from the worker/pipeline
          console.log("[App] Worker progress:", msg);
          break;
        case "ready":
          console.log("[App] Worker ready for transcription", msg.model || "");
          break;
        case "update":
          // partial transcript update
          try {
            const data = msg.data;
            if (data) {
              // data may be [text, { chunks }]
              const text = Array.isArray(data) ? data[0] : data.text || "";
              setTranscription(text);
            }
          } catch (e) {
            console.warn("[App] Failed to handle update message", e);
          }
          break;
        case "complete":
          // final result
          console.log("[App] Transcription complete from worker", msg);
          try {
            const out = msg.data || msg;
            const text = out?.text || out?.data?.text || msg.text || "";
            if (text && String(text).trim().length > 0) {
              setTranscription(String(text));
            } else {
              setTranscription("No speech detected.");
            }
          } catch (e) {
            console.error("[App] Error parsing complete message", e);
            setTranscription("Transcription complete (no text)");
          }
          pendingTranscriptionRef.current.delete(messageId);
          setStatus("ready");
          break;
        case "error":
          console.error("[App] Worker error:", msg.data || msg.error || msg);
          setTranscription(
            "Transcription failed: " +
              (msg?.data?.message || msg.error || "Unknown error")
          );
          pendingTranscriptionRef.current.delete(messageId);
          setStatus("ready");
          break;
        default:
          // Unknown or auxiliary message; just log
          // e.g., pipeline progress objects
          // console.log('[App] Worker message (unhandled):', msg);
          break;
      }
    };

    worker.addEventListener("message", handleWorkerMessage);

    // Ping the worker to ensure it's alive and responsive
    const pingId = "ping-" + Date.now();
    const pingTimeout = setTimeout(() => {
      console.warn(
        "[App] Worker did not reply to ping within 5s. Worker may be busy or failed to initialize."
      );
    }, 5000);

    // Store timeout ref so handler can clear it on pong
    const onPong = (evt) => {
      const m = evt.data || {};
      if (m.status === "pong") {
        console.log("[App] Worker pong received", m);
        clearTimeout(pingTimeout);
        worker.removeEventListener("message", onPong);
      }
    };

    worker.addEventListener("message", onPong);
    try {
      worker.postMessage({ cmd: "ping", messageId: pingId });
    } catch (e) {
      console.error("[App] Failed to send ping to worker:", e);
    }

    return () => {
      worker.removeEventListener("message", handleWorkerMessage);
      worker.removeEventListener("message", onPong);
    };
  }, []);

  // --- Helper: resample audio to target sample rate (linear interpolation) ---
  function resampleAudio(audioBuffer, targetSampleRate = 16000) {
    const sourceSampleRate = audioBuffer.sampleRate;
    const sourceData = audioBuffer.getChannelData(0);

    if (sourceSampleRate === targetSampleRate) {
      return new Float32Array(sourceData);
    }

    const sampleRateRatio = sourceSampleRate / targetSampleRate;
    const newLength = Math.round(sourceData.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const sourceIndex = i * sampleRateRatio;
      const index = Math.floor(sourceIndex);
      const fraction = sourceIndex - index;

      if (index + 1 < sourceData.length) {
        result[i] =
          sourceData[index] * (1 - fraction) + sourceData[index + 1] * fraction;
      } else {
        result[i] = sourceData[index];
      }
    }

    return result;
  }

  // --- Helper: encode Float32Array (-1..1) to WAV PCM16 ArrayBuffer ---
  function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF identifier */ writeString(view, 0, "RIFF");
    /* file length */ view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */ writeString(view, 8, "WAVE");
    /* format chunk identifier */ writeString(view, 12, "fmt ");
    /* format chunk length */ view.setUint32(16, 16, true);
    /* sample format (raw) */ view.setUint16(20, 1, true);
    /* channel count */ view.setUint16(22, 1, true);
    /* sample rate */ view.setUint32(24, sampleRate, true);
    /* byte rate (sampleRate * blockAlign) */ view.setUint32(
      28,
      sampleRate * 2,
      true
    );
    /* block align (channelCount * bytesPerSample) */ view.setUint16(
      32,
      2,
      true
    );
    /* bits per sample */ view.setUint16(34, 16, true);
    /* data chunk identifier */ writeString(view, 36, "data");
    /* data chunk length */ view.setUint32(40, samples.length * 2, true);

    // PCM16 conversion
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return buffer;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // --- Helper: convert audio blob to Float32Array resampled to 16k using OfflineAudioContext ---
  async function convertAudioToFloat32(audioBlob) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // We need to resample to 16000Hz for Whisper
    const targetSampleRate = 16000;
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * targetSampleRate), targetSampleRate);

    // Create a buffer source for the offline context
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();

    // Render the audio at 16k
    const resampledBuffer = await offlineCtx.startRendering();

    // Return the Float32Array data from the first channel
    return resampledBuffer.getChannelData(0);
  }

  // --- 2. Recording Logic ---
  const startRecording = async () => {
    console.log("[Recording] Starting recording...");
    setStatus("recording");
    setTranscription("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("[Recording] Microphone access granted");

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      console.log(
        "[Recording] MediaRecorder created, MIME type:",
        recorder.mimeType
      );

      const audioChunks = [];

      recorder.ondataavailable = (e) => {
        console.log(
          "[Recording] Data available, chunk size:",
          e.data.size,
          "bytes"
        );
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        console.log(
          "[Recording] Stop event fired. Total audio chunks:",
          audioChunks.length
        );
        stream.getTracks().forEach((track) => track.stop());

        if (audioChunks.length === 0) {
          console.warn("[Recording] No audio data recorded.");
          setStatus("ready");
          return;
        }

        setStatus("processing");
        console.log(
          "[Recording] Status set to processing. Calling transcribeAudio..."
        );

        const recordedType = audioChunks[0].type || "audio/webm";
        console.log("[Recording] Blob MIME type:", recordedType);
        const blob = new Blob(audioChunks, { type: recordedType });
        console.log("[Recording] Blob created, size:", blob.size, "bytes");

        await transcribeAudio(blob);
        setStatus("ready");
      };

      recorder.start();
      console.log("[Recording] Recording started.");
    } catch (err) {
      console.error("[Recording] Microphone access error:", err);
      alert("Microphone access denied or not supported.");
      setStatus("ready");
    }
  };

  const stopRecording = () => {
    console.log(
      "[Recording] Stop requested. Recorder state:",
      mediaRecorderRef.current?.state
    );
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      console.log("[Recording] Calling recorder.stop()");
      mediaRecorderRef.current.stop();
    } else {
      console.warn(
        "[Recording] Stop called but recorder is not in recording state"
      );
    }
  };

  // --- 3. Transcription Logic (Web Worker) ---
  const transcribeAudio = async (audioBlob) => {
    console.log("[App] Starting transcription with Web Worker...");
    setTranscription("Transcribing...");
    setStatus("processing");

    try {
      // Convert blob -> Float32Array resampled to 16kHz using OfflineAudioContext
      const float32 = await convertAudioToFloat32(audioBlob);
      const sampleRate = 16000;
      console.log("[App] Converted audio to Float32 length:", float32.length);

      const messageId = ++messageCounter;
      const worker = getTranscriptionWorker();

      // Send raw Float32 ArrayBuffer as transferable along with sampleRate
      const audioBuffer = float32.buffer;
      worker.postMessage(
        {
          audio: audioBuffer,
          sampleRate,
          messageId,
          model: "Xenova/whisper-tiny.en",
          quantized: false,
          multilingual: false,
          subtask: "transcribe",
          language: "english",
          chunk_length_s: 30,
          stride_length_s: 5,
        },
        [audioBuffer]
      );

      // Track pending so we can cleanup if needed
      pendingTranscriptionRef.current.set(messageId, { sent: true });
      console.log(
        "[App] Message sent to worker (wav transferable). messageId:",
        messageId
      );
    } catch (error) {
      console.error("[App] Transcription setup error:", error);
      setTranscription("Error: " + error.message);
      setStatus("ready");
    }
  };

  // --- 4. UI Handlers ---
  const copyText = () => {
    navigator.clipboard.writeText(transcription);
  };

  const handlePointerDown = (e) => {
    if (e.button !== 0 || status !== "ready") return;
    e.preventDefault();
    pointerActiveRef.current = true;
    startRecording();
  };

  const handlePointerUp = () => {
    if (pointerActiveRef.current && status === "recording") {
      pointerActiveRef.current = false;
      stopRecording();
    }
  };

  useEffect(() => {
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);

    return () => {
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [status]);

  return (
    <div className="app-container">
      <div className="card">
        <header className="header">
          <div className="header-content">
            <h1>Audio Transcriber</h1>
          </div>
          <div className="header-controls">
            <div className={`status-badge ${status}`}>
              <span className="dot"></span>
              {status === "processing"
                ? "Processing..."
                : status === "recording"
                ? "Recording..."
                : "Ready"}
            </div>
          </div>
        </header>

        <div className="transcript-area">
          {transcription ? (
            <p className="transcript-text">{transcription}</p>
          ) : (
            <div className="placeholder">
              {status === "processing"
                ? "Transcribing audio..."
                : status === "error"
                ? "Error transcribing. Please try again."
                : "Press and hold the button to start speaking."}
            </div>
          )}
          {status === "recording" && (
            <div className="recording-indicator">
              <span className="wave"></span>
              <span className="wave"></span>
              <span className="wave"></span>
              Listening...
            </div>
          )}
        </div>

        <div className="controls">
          <button
            className="btn record-btn"
            onPointerDown={handlePointerDown}
            disabled={status !== "ready"}
            aria-pressed={status === "recording"}
            title="Press and hold to record (release to stop)"
          >
            <MicIcon />
            {status === "recording"
              ? "Recording..."
              : status === "loading"
              ? "Loading..."
              : "Hold to Record"}
          </button>
          <div className="secondary-actions">
            <button
              className="btn icon-btn"
              onClick={copyText}
              disabled={!transcription}
              title="Copy to clipboard"
            >
              <CopyIcon />
            </button>
            <button
              className="btn icon-btn danger"
              onClick={() => setTranscription("")}
              disabled={!transcription}
              title="Clear text"
            >
              <TrashIcon />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// SVG Icon Components
const MicIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
    <line x1="12" y1="19" x2="12" y2="23"></line>
    <line x1="8" y1="23" x2="16" y2="23"></line>
  </svg>
);

const CopyIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
  </svg>
);

const TrashIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="3 6 5 6 21 6"></polyline>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
  </svg>
);
