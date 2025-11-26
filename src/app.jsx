import React, { useState, useRef, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";

// Configuration
env.allowLocalModels = false;
env.useBrowserCache = true;
const PT_VOCABULARY = `
  theraband, kettle bell, cable, machine, active assistive, AAROM, 
  flexion, extension, abduction, adduction, internal rotation, external rotation, 
  dorsiflexion, plantarflexion, supine, prone, side-lying, reps, sets, 
  tibial, femoral, patella, quadriceps, hamstrings, gluteus maximus, 
  distal, proximal, medial, lateral.
`;

const MODELS = [
  {
    id: "whisper-tiny",
    name: "Whisper Tiny (Fast)",
    modelId: "Xenova/whisper-tiny.en",
  },
  {
    id: "medical-v1",
    name: "Medical v1 (Specialized)",
    modelId: "Crystalcareai/Whisper-Medicalv1",
  },
  {
    id: "lite-large",
    name: "Lite Whisper Large v3 (Accurate)",
    modelId: "onnx-community/lite-whisper-large-v3-turbo-fast-ONNX",
  },
];

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState(null); // 'loading', 'ready', 'recording', 'processing'
  const [transcription, setTranscription] = useState("");
  const [progress, setProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState("whisper-tiny");

  // --- Refs ---
  const transcriberRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]); // Stores the raw audio data
  const isBusyRef = useRef(false); // Prevents overlapping processing
  const intervalRef = useRef(null); // Timer for periodic updates
  const streamRef = useRef(null); // Keep track of the stream to close it properly
  const pointerActiveRef = useRef(false); // track press-and-hold state
  const loadTokenRef = useRef(0); // prevent race when switching models

  // --- 1. Load Model ---
  useEffect(() => {
    const loadModel = async () => {
      // increment token to mark this load; other pending loads will be ignored
      const myToken = ++loadTokenRef.current;
      setStatus("loading");
      setProgress(0);
      // Clear the current transcriber while loading the new one so realtime
      // processing won't accidentally call an old model.
      transcriberRef.current = null;

      try {
        // Find the selected model
        const model = MODELS.find((m) => m.id === selectedModel);
        const modelId = model ? model.modelId : "Xenova/whisper-tiny.en";

        // Provide a progress callback that ignores updates from stale loads
        const progress_callback = (data) => {
          if (loadTokenRef.current !== myToken) return;
          if (data.status === "progress") {
            setProgress(Math.round(data.progress));
          }
        };

        const pipelineInstance = await pipeline(
          "automatic-speech-recognition",
          modelId,
          { progress_callback }
        );

        // Only apply if this load is still the latest requested
        if (loadTokenRef.current === myToken) {
          transcriberRef.current = pipelineInstance;
          setStatus("ready");
        } else {
          // stale load: dispose if possible (best-effort)
          try {
            if (pipelineInstance && pipelineInstance.destroy)
              pipelineInstance.destroy();
          } catch (e) {
            // ignore
          }
        }
      } catch (err) {
        // Only set error if this is the active load
        if (loadTokenRef.current === myToken) {
          console.error(err);
          setStatus("error");
        }
      }
    };
    loadModel();

    return () => {
      // Cleanup on unmount
      clearInterval(intervalRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [selectedModel]);

  // --- 2. Recording Logic ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);

      // Reset state for new session
      audioChunksRef.current = [];
      setTranscription("");

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      // Start with 1s timeslice so ondataavailable fires regularly for realtime updates
      mediaRecorderRef.current.start(1000);
      setStatus("recording");

      // REAL-TIME LOGIC:
      // Instead of waiting for `onstop`, we periodically grab the *current*
      // buffer and transcribe it. This allows the model to "correct" itself
      // as it gets more context.
      intervalRef.current = setInterval(processRealtimeAudio, 2000);
    } catch (err) {
      console.error(err);
      alert("Microphone access denied or not supported.");
    }
  };

  const stopRecording = async () => {
    // Proceed to stop if recorder exists and is not already inactive.
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      clearInterval(intervalRef.current); // Stop the periodic updates
      try {
        mediaRecorderRef.current.stop();
      } catch (err) {
        // Ignore if already stopped
      }
      setStatus("processing");

      // Stop the microphone
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      // Do one final full-quality pass
      setTimeout(async () => {
        await processRealtimeAudio(true);
        setStatus("ready");
      }, 500);
    }
  };

  // --- 3. Transcription Logic ---
  const processRealtimeAudio = async (isFinal = false) => {
    if (!transcriberRef.current || (isBusyRef.current && !isFinal)) return;

    // Guard: Don't process if we have no audio yet
    if (audioChunksRef.current.length === 0) return;

    isBusyRef.current = true;

    try {
      // Create a single blob from all chunks so far. Prefer the recorded
      // chunk MIME type when available; don't force a type that may be
      // incorrect for the browser/codec combination.
      const recordedType =
        (audioChunksRef.current[0] && audioChunksRef.current[0].type) ||
        (mediaRecorderRef.current && mediaRecorderRef.current.mimeType) ||
        "audio/webm";
      const blob = new Blob(audioChunksRef.current, { type: recordedType });

      // Try passing the Blob directly to the pipeline. Some browsers
      // produce blobs that the pipeline can consume directly. If that
      // fails with an encoding/decoding error, fall back to passing an
      // ArrayBuffer obtained from the blob.
      let output;
      try {
        output = await transcriberRef.current(blob, {
        chunk_length_s: 30, // Whisper works best with 30s chunks
        stride_length_s: 5,
        language: "english",
        task: "transcribe",
        return_timestamps: false, // Set true if you want word timings
        initial_prompt: `The following is a physical therapy documentation session. Terminology: ${PT_VOCABULARY}`,
      });
      } catch (err) {
        // If this looks like an audio decode/encoding problem, try an
        // ArrayBuffer fallback which some backends accept better.
        const isEncodingError =
          err && (err.name === "EncodingError" || /decode/i.test(err.message));
        if (isEncodingError) {
          try {
            const arrayBuffer = await blob.arrayBuffer();
            output = await transcriberRef.current(arrayBuffer, {
              chunk_length_s: 30,
              stride_length_s: 5,
              language: "english",
              task: "transcribe",
              return_timestamps: false,
              initial_prompt: `The following is a physical therapy documentation session. Terminology: ${PT_VOCABULARY}`,
            });
          } catch (err2) {
            // rethrow the original error if fallback also fails
            throw err;
          }
        } else {
          throw err;
        }
      }
      // Update the text with the latest result.
      // Note: We replace the whole text because Whisper may have auto-corrected
      // previous words based on new context.
      if (output && output.text) {
        setTranscription(output.text);
      }
      
    } catch (error) {
      console.error("Transcription error", error);
    } finally {
      isBusyRef.current = false;
    }
  };

  // --- 4. UI Helpers ---
  const copyText = () => {
    navigator.clipboard.writeText(transcription);
    // Optional: Add a toast notification here
  };

  // Press-and-hold handlers: start on pointerdown, stop on pointerup.
  const handlePointerDown = (e) => {
    // Ignore secondary buttons and when model is loading/processing
    if (e.button && e.button !== 0) return;
    if (status === "loading" || status === "processing") return;
    e.preventDefault();
    pointerActiveRef.current = true;
    // Start recording
    startRecording();
    // Ensure we stop if pointer is released anywhere
    window.addEventListener("pointerup", handlePointerUp);
  };

  const handlePointerUp = (e) => {
    if (!pointerActiveRef.current) return;
    pointerActiveRef.current = false;
    stopRecording();
    window.removeEventListener("pointerup", handlePointerUp);
  };

  // Keyboard support: Space or Enter to hold-record (keydown -> start, keyup -> stop)
  const handleKeyDown = (e) => {
    if (e.key === " " || e.key === "Spacebar" || e.key === "Enter") {
      if (pointerActiveRef.current) return;
      e.preventDefault();
      pointerActiveRef.current = true;
      startRecording();
      window.addEventListener("keyup", handleKeyUp);
    }
  };

  const handleKeyUp = (e) => {
    if (e.key === " " || e.key === "Spacebar" || e.key === "Enter") {
      if (!pointerActiveRef.current) return;
      pointerActiveRef.current = false;
      stopRecording();
      window.removeEventListener("keyup", handleKeyUp);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        {/* Header */}
        <header className="header">
          <div className="header-content"></div>

          <div className="header-controls">
            <div className="model-selector">
              <label htmlFor="model-select">Model:</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={status === "recording" || status === "processing"}
                className="select-input"
              >
                {MODELS.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div className={`status-badge ${status}`}>
              <span className="dot"></span>
              {status === "loading" ? `Loading Model (${progress}%)` : status}
            </div>
          </div>
        </header>

        {/* Main Transcript Area */}
        <div className="transcript-area">
          {transcription ? (
            <p className="transcript-text">{transcription}</p>
          ) : (
            <div className="placeholder">
              {status === "loading"
                ? "Initializing AI model..."
                : "Ready. Click Record to start speaking."}
            </div>
          )}

          {/* Visualizer / Active State Overlay */}
          {status === "recording" && (
            <div className="recording-indicator">
              <span className="wave"></span>
              <span className="wave"></span>
              <span className="wave"></span>
              Listening...
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="controls">
          <button
            className="btn record-btn"
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onKeyDown={handleKeyDown}
            onKeyUp={handleKeyUp}
            disabled={status === "loading" || status === "processing"}
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

// Simple Icons Components for a cleaner JSX
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
const StopIcon = () => (
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
    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
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
