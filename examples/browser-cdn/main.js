import {
  configureOrtWeb,
  loadHuggingfaceModel,
  loadHuggingfaceVadModel,
} from "https://sandergi.com/cdn/onnx-asr-web/dist/browser.js";

const ASR_REPO_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx";
const VAD_REPO_ID = "onnx-community/silero-vad";

const output = document.getElementById("output");
const audioInput = document.getElementById("audioFile");
const runButton = document.getElementById("run");

configureOrtWeb({
  wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/",
});

let modelPromise = null;

runButton.addEventListener("click", async () => {
  const file = audioInput.files?.[0];
  if (!file) {
    output.textContent = "Select a WAV file first.";
    return;
  }

  try {
    output.textContent = "Loading model...";
    if (!modelPromise) {
      const vadModel = await loadHuggingfaceVadModel(VAD_REPO_ID, {
        sessionOptions: { executionProviders: ["wasm"] },
      });

      modelPromise = loadHuggingfaceModel(ASR_REPO_ID, {
        sessionOptions: { executionProviders: ["wasm"] },
        vadModel,
      });
    }

    const model = await modelPromise;
    output.textContent = "Running inference...";

    const audioBytes = await file.arrayBuffer();
    const result = await model.transcribeWavBuffer(audioBytes);

    const lines = [result.text];
    if (Array.isArray(result.segments)) {
      lines.push("");
      lines.push(`VAD segments: ${result.segments.length}`);
    }
    output.textContent = lines.join("\n");
  } catch (error) {
    output.textContent = String(error);
    modelPromise = null;
  }
});
