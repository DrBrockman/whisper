import React, { useState, useRef, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";

// Configuration
env.allowLocalModels = false;
env.useBrowserCache = true;

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState(null); // 'loading', 'ready', 'recording', 'processing'
  const [transcription, setTranscription] = useState("");
  const [progress, setProgress] = useState(0);

  // --- Refs ---
  const transcriberRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]); // Stores the raw audio data
  const isBusyRef = useRef(false); // Prevents overlapping processing
  const intervalRef = useRef(null); // Timer for periodic updates
  const streamRef = useRef(null); // Keep track of the stream to close it properly

  // --- 1. Load Model ---
  useEffect(() => {
    const loadModel = async () => {
      setStatus("loading");
      try {
        // "whisper-tiny.en" is fast and decent for English.
        // For multi-lingual, use "Xenova/whisper-tiny"
        transcriberRef.current = await pipeline(
          "automatic-speech-recognition",
          "Xenova/whisper-tiny.en",
          {
            progress_callback: (data) => {
              if (data.status === "progress") {
                setProgress(Math.round(data.progress));
              }
            },
          }
        );
        console.log("here it"); // Debug: print the pipeline object and its call signature / docstring
        console.log("ASR task:", transcriberRef.current?.task);
        console.log(
          "Pipeline ownProps:",
          Object.getOwnPropertyNames(transcriberRef.current)
        );
        console.log(
          "Model keys:",
          Object.keys(transcriberRef.current?.model || {})
        );
        console.log("Model config:", transcriberRef.current?.model?.config);
        console.log(
          "Processor keys:",
          Object.getOwnPropertyNames(transcriberRef.current?.processor || {})
        );
        console.log(
          "Tokenizer keys:",
          Object.getOwnPropertyNames(transcriberRef.current?.tokenizer || {})
        );
        console.log(
          "Processor feature_extractor:",
          transcriberRef.current?.processor?.feature_extractor
        );
        console.log("aint!!!");
        setStatus("ready");
      } catch (err) {
        console.error(err);
        setStatus("error");
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
  }, []);

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

      mediaRecorderRef.current.start();
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
    if (mediaRecorderRef.current && status === "recording") {
      clearInterval(intervalRef.current); // Stop the periodic updates
      mediaRecorderRef.current.stop();
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
      // Create a single blob from all chunks so far
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const url = URL.createObjectURL(blob);

      // Run the model
      const output = await transcriberRef.current(url, {
        chunk_length_s: 30, // Whisper works best with 30s chunks
        stride_length_s: 5,
        language: "english",
        task: "transcribe",
        return_timestamps: false, // Set true if you want word timings
      });

      // Update the text with the latest result.
      // Note: We replace the whole text because Whisper may have auto-corrected
      // previous words based on new context.
      if (output && output.text) {
        setTranscription(output.text);
      }

      URL.revokeObjectURL(url);
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

  return (
    <div className="app-container">
      <div className="card">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <h1>Whisper Live</h1>
            <p>In-browser, real-time speech recognition</p>
          </div>
          <div className={`status-badge ${status}`}>
            <span className="dot"></span>
            {status === "loading" ? `Loading Model (${progress}%)` : status}
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
          {status === "recording" ? (
            <button className="btn stop-btn" onClick={stopRecording}>
              <StopIcon /> Stop
            </button>
          ) : (
            <button
              className="btn record-btn"
              onClick={startRecording}
              disabled={status === "loading" || status === "processing"}
            >
              <MicIcon /> {status === "loading" ? "Loading..." : "Record"}
            </button>
          )}

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
