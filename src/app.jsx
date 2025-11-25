import React, { useState, useRef, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";

// Configuration: Force local execution and setup paths
env.allowLocalModels = false;
env.useBrowserCache = true;

export default function AudioTranscriber() {
  // --- State ---
  const [status, setStatus] = useState(null); // 'loading', 'ready', 'recording', 'processing'
  const [transcription, setTranscription] = useState("");
  const [partial, setPartial] = useState(""); // live partial transcript while recording
  const [progress, setProgress] = useState(0);

  // --- Refs ---
  const transcriberRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const inFlightRef = useRef(false); // avoid overlapping model calls
  const transcriptElRef = useRef(null);

  // --- 1. Load the Model on Mount ---
  useEffect(() => {
    const loadModel = async () => {
      setStatus("loading");
      try {
        // We use 'whisper-tiny.en' because it is small (~40MB) and fast on mobiles
        transcriberRef.current = await pipeline(
          "automatic-speech-recognition",
          "Xenova/whisper-tiny.en",
          {
            // Progress callback to show loading bar
            progress_callback: (data) => {
              if (data.status === "progress") {
                setProgress(Math.round(data.progress));
              }
            },
          }
        );
        setStatus("ready");
      } catch (err) {
        console.error(err);
        setTranscription(
          "Error loading model. Does your browser support WebGPU?"
        );
      }
    };
    loadModel();
  }, []);

  // Auto-scroll transcript when it updates
  useEffect(() => {
    try {
      const el = transcriptElRef.current;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    } catch (err) {
      // ignore
    }
  }, [transcription, partial]);

  // --- 2. Recording Logic ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Create a MediaRecorder that emits short chunks (timeslice) so we can
      // process them incrementally and update the UI in near-real-time.
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0) return;

        // Keep a rolling buffer (in case we want to stitch final audio)
        chunksRef.current.push(e.data);

        // Process the chunk immediately for near-real-time transcription.
        // Avoid overlapping requests to the local model.
        if (transcriberRef.current && !inFlightRef.current) {
          inFlightRef.current = true;
          try {
            const blob = e.data;
            const url = URL.createObjectURL(blob);
            // Call the model with the chunk URL. Many local ASR pipelines accept a
            // single-file URL and will return text for that chunk.
            const result = await transcriberRef.current(url);
            if (result && result.text) {
              // Append the chunk result to the main transcription and
              // clear any partial state.
              setTranscription(
                (prev) => prev + (prev ? " " : "") + result.text
              );
              setPartial("");
            }
            URL.revokeObjectURL(url);
          } catch (err) {
            console.error("Incremental transcription error:", err);
          } finally {
            inFlightRef.current = false;
          }
        }
      };

      mediaRecorderRef.current.onstop = processAudio;

      // Start with a 1 second timeslice so we get chunks frequently.
      mediaRecorderRef.current.start(1000);
      setStatus("recording");
    } catch (err) {
      alert("Microphone access denied. Please check settings.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && status === "recording") {
      mediaRecorderRef.current.stop();
      // Stop all tracks to release microphone
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
    }
  };

  // --- 3. Processing (The AI Part) ---
  const processAudio = async () => {
    setStatus("processing");
    // Create a blob from all recorded chunks (final pass) and run once more
    // to capture any trailing audio that wasn't processed incrementally.
    try {
      if (chunksRef.current.length > 0 && transcriberRef.current) {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        const result = await transcriberRef.current(url);
        if (result && result.text) {
          setTranscription((prev) => prev + (prev ? " " : "") + result.text);
        }
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error(err);
      setTranscription((prev) => prev + " [Error transcribing final audio]");
    } finally {
      setStatus("ready");
      // reset chunks for the next recording
      chunksRef.current = [];
    }
  };

  // --- 4. Helper Functions ---
  const copyToClipboard = () => {
    navigator.clipboard.writeText(transcription);
    alert("Copied to clipboard!");
  };

  const clearText = () => {
    setTranscription("");
  };

  // --- 5. The UI ---
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex flex-col items-center py-10 font-sans">
      <div className="w-full max-w-3xl px-6">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-extrabold text-gray-900">
              Whisper — Live Transcription
            </h1>
            <p className="text-sm text-gray-500">
              Local, privacy-first speech to text
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xs text-gray-500">Model status</div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-3 w-3 rounded-full ${
                    status === "ready"
                      ? "bg-green-400"
                      : status === "loading"
                      ? "bg-yellow-400"
                      : "bg-gray-300"
                  }`}
                ></span>
                <span className="text-sm text-gray-700 capitalize">
                  {status || "idle"}
                </span>
              </div>
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Transcript column */}
          <div className="md:col-span-2 bg-white rounded-2xl shadow p-6 min-h-[320px] flex flex-col">
            <div
              className="flex-1 overflow-auto"
              ref={transcriptElRef}
              aria-live="polite"
            >
              {transcription ? (
                <div className="prose max-w-none text-gray-800 leading-relaxed">
                  {transcription}
                </div>
              ) : (
                <div className="text-gray-400 italic">
                  Your live transcription will appear here.
                </div>
              )}

              {partial && (
                <div className="mt-3 text-sm text-gray-500">
                  Partial: <span className="text-gray-700">{partial}</span>
                </div>
              )}
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-gray-500">
                {status === "processing"
                  ? "Finalizing..."
                  : status === "recording"
                  ? "Listening"
                  : "Ready"}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={copyToClipboard}
                  className="px-4 py-2 bg-white border rounded-lg text-sm text-gray-700"
                >
                  Copy
                </button>
                <button
                  onClick={clearText}
                  className="px-4 py-2 bg-white border rounded-lg text-sm text-red-500"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>

          {/* Controls column */}
          <aside className="bg-white rounded-2xl shadow p-6 flex flex-col items-center">
            <div className="mb-4">
              <div className="text-sm text-gray-500">Recording</div>
              <div className="mt-2">
                {status === "recording" ? (
                  <button
                    onClick={stopRecording}
                    className="w-24 h-24 bg-red-500 rounded-full shadow-lg flex items-center justify-center text-white"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={startRecording}
                    disabled={status === "loading" || status === "processing"}
                    className="w-24 h-24 bg-blue-600 hover:bg-blue-700 rounded-full shadow-lg flex items-center justify-center text-white"
                  >
                    Record
                  </button>
                )}
              </div>
            </div>

            <div className="w-full mt-3">
              {status === "loading" && (
                <div className="text-sm text-yellow-600">
                  Downloading model... {progress}%
                </div>
              )}
              {status === "processing" && (
                <div className="text-sm text-blue-600">
                  Processing final audio...
                </div>
              )}
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
