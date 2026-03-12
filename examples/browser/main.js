import {
  configureOrtWeb,
  loadHuggingfaceVadModel,
  loadLocalModel,
  loadLocalVadModel,
} from "../../dist/browser.js";

const output = document.getElementById("output");
const wordsRoot = document.getElementById("words");
const audioInput = document.getElementById("audioFile");
const modelInput = document.getElementById("modelBaseUrl");
const vadModeInput = document.getElementById("vadMode");
const vadBaseInput = document.getElementById("vadBaseUrl");
const vadRepoInput = document.getElementById("vadRepoId");
const runButton = document.getElementById("run");

configureOrtWeb({
  wasmPaths: "/node_modules/onnxruntime-web/dist/",
});

let modelPromise = null;
let modelCacheKey = "";
let audioContext = null;
let decodedAudioBuffer = null;
let currentSource = null;

function formatSeconds(value) {
  return Number(value).toFixed(3);
}

function stopCurrentPlayback() {
  if (currentSource) {
    currentSource.stop();
    currentSource.disconnect();
    currentSource = null;
  }
}

async function getAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  return audioContext;
}

async function decodeAudioForPlayback(arrayBuffer) {
  const context = await getAudioContext();
  const copied = arrayBuffer.slice(0);
  decodedAudioBuffer = await context.decodeAudioData(copied);
}

async function playSegment(start, end) {
  if (!decodedAudioBuffer) {
    return;
  }

  const context = await getAudioContext();
  stopCurrentPlayback();

  const clampedStart = Math.max(
    0,
    Math.min(start, decodedAudioBuffer.duration),
  );
  const clampedEnd = Math.max(
    clampedStart,
    Math.min(end, decodedAudioBuffer.duration),
  );
  const duration = Math.max(0.01, clampedEnd - clampedStart);

  const source = context.createBufferSource();
  source.buffer = decodedAudioBuffer;
  source.connect(context.destination);
  source.start(0, clampedStart, duration);
  source.onended = () => {
    if (currentSource === source) {
      currentSource.disconnect();
      currentSource = null;
    }
  };
  currentSource = source;
}

function renderWords(words) {
  wordsRoot.innerHTML = "";

  if (!words || words.length === 0) {
    wordsRoot.textContent = "No word timestamps returned.";
    return;
  }

  const title = document.createElement("h2");
  title.textContent = "Word Timestamps";
  wordsRoot.appendChild(title);

  for (const item of words) {
    const row = document.createElement("div");

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Play";
    button.addEventListener("click", () => {
      playSegment(item.start, item.end).catch((error) => {
        output.textContent = String(error);
      });
    });

    const label = document.createElement("span");
    label.textContent = ` [${formatSeconds(item.start)} - ${formatSeconds(item.end)}] ${item.word}`;

    row.appendChild(button);
    row.appendChild(label);
    wordsRoot.appendChild(row);
  }
}

function getModelCacheKey() {
  return JSON.stringify({
    asrBaseUrl: modelInput.value.trim(),
    vadMode: vadModeInput.value,
    vadBaseUrl: vadBaseInput.value.trim(),
    vadRepoId: vadRepoInput.value.trim(),
  });
}

function isVadEnabled() {
  return vadModeInput.value !== "none";
}

async function resolveVadModel() {
  const mode = vadModeInput.value;
  if (mode === "none") {
    return null;
  }

  if (mode === "local") {
    const baseUrl = vadBaseInput.value.trim();
    if (!baseUrl) {
      throw new Error("Enter VAD local base URL.");
    }
    return loadLocalVadModel(baseUrl, {
      sessionOptions: { executionProviders: ["wasm"] },
    });
  }

  if (mode === "hf") {
    const repoId = vadRepoInput.value.trim();
    if (!repoId) {
      throw new Error("Enter VAD Hugging Face repo id.");
    }
    return loadHuggingfaceVadModel(repoId, {
      sessionOptions: { executionProviders: ["wasm"] },
    });
  }

  throw new Error(`Unsupported VAD mode: ${mode}`);
}

runButton.addEventListener("click", async () => {
  const file = audioInput.files?.[0];
  if (!file) {
    output.textContent = "Select a WAV file first.";
    return;
  }

  const baseUrl = modelInput.value.trim();
  if (!baseUrl) {
    output.textContent = "Enter model base URL first.";
    return;
  }

  try {
    output.textContent = "Loading model...";
    wordsRoot.textContent = "";
    const cacheKey = getModelCacheKey();
    if (!modelPromise || modelCacheKey !== cacheKey) {
      const vadModel = await resolveVadModel();
      modelPromise = loadLocalModel(baseUrl, {
        sessionOptions: { executionProviders: ["wasm"] },
        vadModel,
      });
      modelCacheKey = cacheKey;
    }

    const model = await modelPromise;
    output.textContent = "Running inference...";

    const audioBytes = await file.arrayBuffer();
    await decodeAudioForPlayback(audioBytes);

    const result = await model.transcribeWavBuffer(audioBytes);
    const lines = [result.text];
    if (isVadEnabled()) {
      if (!Array.isArray(result.segments)) {
        throw new Error(
          "VAD is enabled, but transcription result has no segments. Model was likely loaded without VAD.",
        );
      }
      lines.push("");
      lines.push(`VAD: enabled (${vadModeInput.value})`);
      lines.push(`VAD segments: ${result.segments.length}`);
      for (const segment of result.segments) {
        lines.push(
          `  [${formatSeconds(segment.startSec)} - ${formatSeconds(segment.endSec)}]`,
        );
      }
    }
    output.textContent = lines.join("\n");
    renderWords(result.words);
  } catch (error) {
    output.textContent = String(error);
    wordsRoot.textContent = "";
    modelPromise = null;
    modelCacheKey = "";
  }
});
