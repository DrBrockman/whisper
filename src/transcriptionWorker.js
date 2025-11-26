/* eslint-disable camelcase */
import { pipeline, env } from "@xenova/transformers";

/*
  Worker for running the automatic-speech-recognition pipeline.
  Supports receiving either a blob URL (string) or a transferred ArrayBuffer/Float32Array.
  Sends progress/update/complete/error messages back to the main thread.
*/

// Disable local models (use CDN/cache behavior controlled in main app)
env.allowLocalModels = false;

// Single cached pipeline instance and currently selected model params
let transcriber = null;
let currentModel = null;
let currentQuantized = null;

async function createOrGetPipeline(
  modelId = "Xenova/whisper-tiny.en",
  progress_callback = null,
  quantized = false
) {
  // If model/quantized changed, dispose old instance
  if (
    transcriber &&
    (currentModel !== modelId || currentQuantized !== quantized)
  ) {
    try {
      transcriber.dispose();
    } catch (e) {
      console.warn("[Worker] Error disposing previous pipeline:", e);
    }
    transcriber = null;
  }

  if (!transcriber) {
    currentModel = modelId;
    currentQuantized = quantized;
    self.postMessage({ status: "initiate", file: modelId });
    try {
      transcriber = await pipeline("automatic-speech-recognition", modelId, {
        quantized,
        progress_callback: (data) => {
          // Forward progress messages from the pipeline to main thread
          if (data && data.status === "progress") {
            self.postMessage({ status: "progress", ...data });
          } else if (data && data.status === "initiate") {
            self.postMessage({ status: "initiate", ...data });
          } else if (data && data.status === "done") {
            self.postMessage({ status: "done", ...data });
          } else {
            // Generic forward
            self.postMessage(data);
          }
        },
      });
      self.postMessage({ status: "ready", model: modelId });
    } catch (e) {
      console.error("[Worker] Pipeline initialization error:", e);
      self.postMessage({
        status: "error",
        data: { message: "Pipeline init failed: " + e.message },
      });
      throw e;
    }
  }

  return transcriber;
}

// Helper: build and manage chunks merging similar to the sample code
function createChunkManager(transcriber) {
  const time_precision =
    (transcriber.processor?.feature_extractor?.config?.chunk_length || 30) /
    (transcriber.model?.config?.max_source_positions || 1500);

  const chunks_to_process = [
    {
      tokens: [],
      finalised: false,
    },
  ];

  function chunk_callback(chunk) {
    let last = chunks_to_process[chunks_to_process.length - 1];
    Object.assign(last, chunk);
    last.finalised = true;

    if (!chunk.is_last) {
      chunks_to_process.push({ tokens: [], finalised: false });
    }
  }

  async function callback_function(item) {
    // `item` is expected to be the generation output sequence(s)
    // Update the token list of the last chunk if possible
    try {
      let last = chunks_to_process[chunks_to_process.length - 1];
      last.tokens = [...(item[0]?.output_token_ids || last.tokens)];

      // Try to decode progressive tokens to text using tokenizer if available
      if (
        transcriber &&
        transcriber.tokenizer &&
        typeof transcriber.tokenizer._decode_asr === "function"
      ) {
        const data = transcriber.tokenizer._decode_asr(chunks_to_process, {
          time_precision,
          return_timestamps: true,
          force_full_sequences: false,
        });

        self.postMessage({ status: "update", data });
        return;
      }

      // Fallback: try to extract any text from `item`
      const fallbackText = item[0]?.text || item[0]?.generated_text || null;
      if (fallbackText) {
        self.postMessage({
          status: "update",
          data: [
            fallbackText,
            { chunks: [{ text: fallbackText, timestamp: [null, null] }] },
          ],
        });
      }
    } catch (e) {
      console.warn("[Worker] callback_function error:", e);
    }
  }

  return { chunk_callback, callback_function };
}

// Message handler
self.addEventListener("message", async (event) => {
  const msg = event.data || {};
  const messageId = msg.messageId || null;

  try {
    const modelId = msg.model || "Xenova/whisper-tiny.en";
    const quantized = !!msg.quantized;

    // Ensure pipeline exists
    const pipelineInstance = await createOrGetPipeline(
      modelId,
      null,
      quantized
    );

    // Prepare chunk manager for incremental updates
    const { chunk_callback, callback_function } =
      createChunkManager(pipelineInstance);

    // Handle simple ping/pong healthcheck
    if (msg.cmd === "ping") {
      self.postMessage({ status: "pong", messageId });
      return;
    }

    // Determine audio input form
    let audioInput = null;
    if (msg.audioUrl && typeof msg.audioUrl === "string") {
      audioInput = msg.audioUrl;
    } else if (msg.audio) {
      // Could be transferred ArrayBuffer (WAV) or a typed array
      const payload = msg.audio;
      if (payload instanceof ArrayBuffer) {
        // Detect WAV by 'RIFF' header and parse PCM16 -> Float32 if present
        try {
          const headerBytes = new Uint8Array(payload, 0, 12);
          const headerStr = String.fromCharCode.apply(null, headerBytes.subarray(0, 4));
          if (headerStr === 'RIFF') {
            const view = new DataView(payload);
            // sampleRate is at byte offset 24 in little-endian for standard WAV fmt with PCM
            const sampleRate = view.getUint32(24, true);

            // Locate 'data' chunk (start at offset 12)
            let dataOffset = 12;
            let dataSize = 0;
            while (dataOffset + 8 <= view.byteLength) {
              const chunkId = String.fromCharCode(
                view.getUint8(dataOffset),
                view.getUint8(dataOffset + 1),
                view.getUint8(dataOffset + 2),
                view.getUint8(dataOffset + 3)
              );
              const chunkSize = view.getUint32(dataOffset + 4, true);
              if (chunkId === 'data') {
                dataOffset += 8; // point to data bytes
                dataSize = chunkSize;
                break;
              }
              dataOffset += 8 + chunkSize;
            }

            if (dataSize === 0) {
              throw new Error('WAV data chunk not found');
            }

            // Assume 16-bit PCM
            const numSamples = dataSize / 2; // 2 bytes per sample
            const float32 = new Float32Array(numSamples);
            let srcOffset = dataOffset;
            for (let i = 0; i < numSamples; i++) {
              const int16 = view.getInt16(srcOffset, true);
              float32[i] = int16 < 0 ? int16 / 32768 : int16 / 32767;
              srcOffset += 2;
            }

            // Provide object with samples and sampleRate so pipeline/processor can use
            audioInput = { samples: float32, sampleRate };
          } else {
            // Treat as raw Float32 ArrayBuffer. If sender included sampleRate, pass as object.
            const samples = new Float32Array(payload);
            if (msg && msg.sampleRate) {
              audioInput = { samples, sampleRate: msg.sampleRate };
            } else {
              audioInput = samples;
            }
          }
        } catch (e) {
          // Fallback to Float32Array view
          audioInput = new Float32Array(payload);
        }
      } else if (ArrayBuffer.isView(payload)) {
        // Already a typed array (e.g., Float32Array)
        audioInput = payload;
      } else if (payload instanceof Blob) {
        // Blob (e.g., if main thread sent a Blob) - create URL for pipeline if needed
        audioInput = URL.createObjectURL(payload);
      } else {
        // Unknown form - passthrough
        audioInput = payload;
      }
    }

    // Call pipeline with callbacks for streaming updates
    const output = await pipelineInstance(audioInput, {
      top_k: 0,
      do_sample: false,
      chunk_length_s: msg.chunk_length_s || 30,
      stride_length_s: msg.stride_length_s || 5,
      language: msg.language || null,
      task: msg.subtask || "transcribe",
      return_timestamps: true,
      force_full_sequences: false,
      callback_function,
      chunk_callback,
    }).catch((err) => {
      throw err;
    });

    // Final result
    self.postMessage({ status: "complete", messageId, data: output });
  } catch (err) {
    console.error("[Worker] Fatal error:", err);
    self.postMessage({
      status: "error",
      messageId,
      data: { message: err.message },
    });
  }
});
