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

// Resample audio to 16kHz (required by Whisper)
function resampleAudio(audioBuffer, targetSampleRate = 16000) {
  const sourceSampleRate = audioBuffer.sampleRate;
  const sourceData = audioBuffer.getChannelData(0);

  if (sourceSampleRate === targetSampleRate) {
    return sourceData;
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

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState("loading");
  const [transcription, setTranscription] = useState("");
  const [progress, setProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState("whisper-tiny");

  // --- Refs ---
  const transcriberRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const pointerActiveRef = useRef(false);
  const loadTokenRef = useRef(0);

  // --- 1. Load Model ---
  useEffect(() => {
    const loadModel = async () => {
      const myToken = ++loadTokenRef.current;
      console.log("[Model] Loading model, token:", myToken);
      setStatus("loading");
      setProgress(0);
      transcriberRef.current = null;

      try {
        const modelId =
          MODELS.find((m) => m.id === selectedModel)?.modelId ||
          "Xenova/whisper-tiny.en";
        console.log("[Model] Model ID:", modelId);

        const progress_callback = (data) => {
          if (loadTokenRef.current !== myToken) return;
          if (data.status === "progress") {
            console.log(
              "[Model] Loading progress:",
              Math.round(data.progress) + "%"
            );
            setProgress(Math.round(data.progress));
          }
        };

        console.log("[Model] Creating pipeline...");
        const pipelineInstance = await pipeline(
          "automatic-speech-recognition",
          modelId,
          { progress_callback }
        );

        if (loadTokenRef.current === myToken) {
          console.log("[Model] Pipeline loaded successfully, token:", myToken);
          transcriberRef.current = pipelineInstance;
          setStatus("ready");
        } else {
          console.log(
            "[Model] Pipeline loaded but token mismatch (stale load)"
          );
        }
      } catch (err) {
        if (loadTokenRef.current === myToken) {
          console.error("[Model] Model loading error:", err);
          setStatus("error");
        }
      }
    };
    loadModel();
  }, [selectedModel]);

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

  // --- 3. Transcription Logic (FIXED) ---
  const transcribeAudio = async (blob) => {
    console.log(
      "[Transcription] Starting transcription. Transcriberref exists:",
      !!transcriberRef.current
    );
    if (!transcriberRef.current) {
      console.error("[Transcription] No transcriberRef.current available!");
      setTranscription("(Model not loaded)");
      return;
    }

    try {
      console.log(
        "[Transcription] Converting blob to audio via AudioContext..."
      );
      const arrayBuffer = await blob.arrayBuffer();
      console.log(
        "[Transcription] ArrayBuffer size:",
        arrayBuffer.byteLength,
        "bytes"
      );

      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      let decodedAudio;

      try {
        decodedAudio = await audioContext.decodeAudioData(arrayBuffer);
        console.log(
          "[Transcription] Audio decoded. Duration:",
          decodedAudio.duration,
          "seconds. Sample rate:",
          decodedAudio.sampleRate
        );
      } catch (decodeError) {
        console.error("[Transcription] Audio decode error:", decodeError);
        setTranscription("(Audio format not supported)");
        return;
      }

      // Check if audio is too short
      if (decodedAudio.duration < 0.1) {
        console.warn(
          "[Transcription] Audio too short:",
          decodedAudio.duration,
          "seconds"
        );
        setTranscription("(Recording too short - hold longer)");
        return;
      }

      // Resample to 16kHz (required by Whisper)
      const resampledAudio = resampleAudio(decodedAudio, 16000);
      console.log(
        "[Transcription] Audio resampled to 16kHz, length:",
        resampledAudio.length
      );

      // Calculate RMS to detect silence
      let sum = 0;
      for (let i = 0; i < resampledAudio.length; i++) {
        sum += resampledAudio[i] * resampledAudio[i];
      }
      const rms = Math.sqrt(sum / resampledAudio.length);
      console.log(
        "[Transcription] Audio RMS (energy level):",
        rms.toFixed(6),
        "(>0.01 indicates speech)"
      );

      if (rms < 0.001) {
        console.warn(
          "[Transcription] Audio appears to be silent (RMS too low)."
        );
        setTranscription("(No audio detected - check microphone)");
        return;
      }

      const prompt = `Capture physical therapy exercises, sets, and reps. For example: theraband external rotation four sets twelve reps. kettle bell squats three sets ten reps. active assistive extension three sets fifteen reps.`;

      console.log("[Transcription] Calling pipeline with resampled audio...");
      console.log("[Transcription] Audio data type:", typeof resampledAudio);
      console.log(
        "[Transcription] Audio is Float32Array:",
        resampledAudio instanceof Float32Array
      );
      console.log(
        "[Transcription] First 10 samples:",
        Array.from(resampledAudio.slice(0, 10))
      );

      let output;
      try {
        output = await transcriberRef.current(resampledAudio, {
          chunk_length_s: 30,
          stride_length_s: 5,
          language: "english",
          task: "transcribe",
          return_timestamps: false,
          initial_prompt: prompt,
        });
      } catch (pipelineError) {
        console.error("[Transcription] Pipeline error:", pipelineError);
        console.error("[Transcription] Error stack:", pipelineError.stack);
        setTranscription("(Pipeline error: " + pipelineError.message + ")");
        return;
      }

      console.log(
        "[Transcription] Pipeline call succeeded. Full output:",
        JSON.stringify(output, null, 2)
      );

      if (output && typeof output.text !== "undefined") {
        const text = output.text.trim();
        console.log("[Transcription] Transcription text received:", text);

        if (text.length > 0) {
          setTranscription(text);
        } else {
          console.warn("[Transcription] Empty transcription text");
          setTranscription("(No speech detected)");
        }
      } else {
        console.warn(
          "[Transcription] Pipeline output missing text field. Output:",
          output
        );
        setTranscription("(Unexpected output format)");
      }
    } catch (error) {
      console.error("[Transcription] Fatal error:", error);
      console.error("[Transcription] Error stack:", error.stack);
      setTranscription("(Error: " + error.message + ")");
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
          <div className="header-content"></div>
          <div className="header-controls">
            <div className={`status-badge ${status}`}>
              <span className="dot"></span>
              {status === "loading" ? `Loading (${progress}%)` : status}
            </div>
          </div>
        </header>

        <div className="transcript-area">
          {transcription ? (
            <p className="transcript-text">{transcription}</p>
          ) : (
            <div className="placeholder">
              {status === "loading"
                ? "Initializing AI model..."
                : status === "error"
                ? "Error loading model. Please refresh."
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
