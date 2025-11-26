import React, { useState, useRef, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";

// Configuration
env.allowLocalModels = false;
env.useBrowserCache = true;

// Define the models available for selection
const MODELS = [
  {
    id: "whisper-tiny",
    name: "Whisper Tiny (Fast)",
    modelId: "Xenova/whisper-tiny.en",
  },
];

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState(null); // 'loading', 'ready', 'recording', 'processing', 'error'
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
      // Clear the current transcriber while loading the new one
      transcriberRef.current = null;

      try {
        const model = MODELS.find((m) => m.id === selectedModel);
        const modelId = model ? model.modelId : "Xenova/whisper-tiny.en";

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

        if (loadTokenRef.current === myToken) {
          transcriberRef.current = pipelineInstance;
          setStatus("ready");
        }
      } catch (err) {
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

      audioChunksRef.current = [];
      setTranscription("");

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      // Start recording without timeslice (simpler, avoids partial audio issues)
      mediaRecorderRef.current.start();
      setStatus("recording");
    } catch (err) {
      console.error(err);
      alert("Microphone access denied or not supported.");
    }
  };

  const stopRecording = async () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      setStatus("processing");

      mediaRecorderRef.current.onstop = async () => {
        // Stop the microphone stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }

        // Transcribe the complete recording
        await transcribeAudio();
        setStatus("ready");

        // Clean up the event listener
        if (mediaRecorderRef.current) {
          mediaRecorderRef.current.onstop = null;
        }
      };

      // Stop the recorder, which will trigger the 'onstop' event
      mediaRecorderRef.current.stop();
    }
  };

  // --- 3. Transcription Logic ---
  const transcribeAudio = async () => {
    if (!transcriberRef.current) return;
    if (audioChunksRef.current.length === 0) return;

    isBusyRef.current = true;

    try {
      const recordedType =
        (audioChunksRef.current[0] && audioChunksRef.current[0].type) ||
        (mediaRecorderRef.current && mediaRecorderRef.current.mimeType) ||
        "audio/webm";
      const blob = new Blob(audioChunksRef.current, { type: recordedType });

      const prompt = `Capture physical therapy exercises, sets, and reps. For example: theraband external rotation four sets twelve reps. kettle bell squats three sets ten reps. active assistive extension three sets fifteen reps.`;

      const commonOptions = {
        chunk_length_s: 30,
        stride_length_s: 5,
        language: "english",
        task: "transcribe",
        return_timestamps: false,
        initial_prompt: prompt,
      };

      let output;
      try {
        output = await transcriberRef.current(blob, commonOptions);
      } catch (err) {
        const isEncodingError =
          err && (err.name === "EncodingError" || /decode/i.test(err.message));
        if (isEncodingError) {
          try {
            const arrayBuffer = await blob.arrayBuffer();
            output = await transcriberRef.current(arrayBuffer, commonOptions);
          } catch (err2) {
            throw err; // rethrow original error if fallback fails
          }
        } else {
          throw err;
        }
      }

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
  };

  const handlePointerDown = (e) => {
    if (e.button && e.button !== 0) return;
    if (status === "loading" || status === "processing" || status === "error")
      return;
    e.preventDefault();
    pointerActiveRef.current = true;
    startRecording();
    window.addEventListener("pointerup", handlePointerUp, { once: true });
    window.addEventListener("pointercancel", handlePointerUp, { once: true });
  };

  const handlePointerUp = () => {
    if (pointerActiveRef.current) {
      pointerActiveRef.current = false;
      stopRecording();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === " " || e.key === "Enter") {
      if (pointerActiveRef.current) return;
      e.preventDefault();
      pointerActiveRef.current = true;
      startRecording();
      window.addEventListener("keyup", handleKeyUp, { once: true });
    }
  };

  const handleKeyUp = (e) => {
    if (e.key === " " || e.key === "Enter") {
      if (pointerActiveRef.current) {
        pointerActiveRef.current = false;
        stopRecording();
      }
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
              {status === "loading" ? `Loading (${progress}%)` : status}
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

        {/* Controls */}
        <div className="controls">
          <button
            className="btn record-btn"
            onPointerDown={handlePointerDown}
            onKeyDown={handleKeyDown}
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
    {" "}
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>{" "}
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>{" "}
    <line x1="12" y1="19" x2="12" y2="23"></line>{" "}
    <line x1="8" y1="23" x2="16" y2="23"></line>{" "}
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
    {" "}
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>{" "}
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>{" "}
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
    {" "}
    <polyline points="3 6 5 6 21 6"></polyline>{" "}
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>{" "}
  </svg>
);
