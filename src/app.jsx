import React, { useState, useRef, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";

// Configuration: Force local execution and setup paths
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
  const chunksRef = useRef([]);

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

  // --- 2. Recording Logic ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = processAudio;

      mediaRecorderRef.current.start();
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

    // Create a blob from the recorded audio chunks
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    const url = URL.createObjectURL(blob);

    try {
      // Run the Whisper model locally
      const result = await transcriberRef.current(url);
      setTranscription((prev) => prev + (prev ? " " : "") + result.text);
    } catch (err) {
      console.error(err);
      setTranscription("Error transcribing audio.");
    } finally {
      setStatus("ready");
      URL.revokeObjectURL(url); // Cleanup memory
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
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6 font-sans">
      {/* Header / Loader */}
      <h1 className="text-2xl font-bold mb-6 text-gray-800">
        Local Whisper on iOS
      </h1>

      {status === "loading" && (
        <div className="mb-4 text-blue-600 font-medium">
          Downloading AI Model... {progress}%
        </div>
      )}

      {/* Text Display Area */}
      <div className="w-full max-w-md bg-white rounded-xl shadow-md p-4 min-h-[200px] mb-6 border border-gray-200">
        {transcription ? (
          <p className="text-gray-800 leading-relaxed">{transcription}</p>
        ) : (
          <p className="text-gray-400 italic">
            Transcription will appear here...
          </p>
        )}

        {status === "processing" && (
          <div className="mt-4 text-sm text-blue-500 animate-pulse">
            Transcribing audio...
          </div>
        )}
      </div>

      {/* Controls Container */}
      <div className="w-full max-w-md flex flex-col gap-4">
        {/* Record Button */}
        <div className="flex justify-center">
          {status === "recording" ? (
            <button
              onClick={stopRecording}
              className="w-20 h-20 bg-red-500 rounded-full shadow-lg flex items-center justify-center animate-pulse border-4 border-red-200"
            >
              <div className="w-8 h-8 bg-white rounded-sm"></div>
            </button>
          ) : (
            <button
              onClick={startRecording}
              disabled={status === "loading" || status === "processing"}
              className={`w-20 h-20 rounded-full shadow-lg flex items-center justify-center border-4 border-gray-100
                ${
                  status === "loading"
                    ? "bg-gray-400"
                    : "bg-blue-600 hover:bg-blue-700"
                }`}
            >
              {/* Microphone Icon */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-10 w-10 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
            </button>
          )}
        </div>
        <div className="text-center text-gray-500 text-sm mb-2">
          {status === "recording" ? "Tap to Stop" : "Tap to Record"}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={copyToClipboard}
            className="flex-1 bg-white border border-gray-300 text-gray-700 py-3 rounded-lg font-medium active:bg-gray-50 transition-colors"
          >
            Copy Text
          </button>
          <button
            onClick={clearText}
            className="flex-1 bg-white border border-gray-300 text-red-500 py-3 rounded-lg font-medium active:bg-gray-50 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
