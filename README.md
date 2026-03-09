# onnx-asr-web

**onnx-asr-web** is a JavaScript package for Automatic Speech Recognition using ONNX models. It's a lightweight, fast, and easy-to-use pure JavaScript package with minimal dependencies (only [`onnxruntime-web`](https://github.com/microsoft/onnxruntime/tree/main/js/web)) that runs with both Node.js and in the browser. This package was heavily inspired by the Python [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr).

## Install

```bash
npm install
```

## Models

Supports all the models from [`istupakov/onnx-asr`](https://huggingface.co/collections/istupakov/onnx-asr). Highlights include:
* NVIDIA [FastConformer](istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx), [Parakeet](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx), and [Canary](https://huggingface.co/istupakov/canary-1b-v2-onnx)
* GigaChat [GigaAM v2/v3](https://huggingface.co/istupakov/gigaam-v3-onnx)
* Alpha Cephei [Vosk 0.54+](https://huggingface.co/alphacep/vosk-model-ru)
* T-Tech [T-one](https://huggingface.co/t-tech/T-one)
* OpenAI [Whisper](https://huggingface.co/istupakov/whisper-base-onnx)

## Usage
### Node.js usage (local model path)

```js
import { readFile } from "node:fs/promises";
import { createParakeetFromLocalDir } from "onnx-asr-web/node";

const asr = await createParakeetFromLocalDir("/absolute/path/to/model-dir", {
  quantization: "int8", // default, prefers *.int8.onnx when present
  sessionOptions: { executionProviders: ["wasm"] },
});

const wav = await readFile("/absolute/path/to/audio.wav");
const result = await asr.transcribeWavBuffer(
  wav.buffer.slice(wav.byteOffset, wav.byteOffset + wav.byteLength),
);

console.log(result.text);
console.log(result.words); // [{word, start, end}, ...] in seconds
```

### Node.js usage (Hugging Face repo id + cache dir)

```js
import { createParakeetFromHuggingFace } from "onnx-asr-web/node";

const asr = await createParakeetFromHuggingFace(
  "istupakov/parakeet-tdt-0.6b-v3-onnx",
  {
    cacheDir: "models", // saves under models/istupakov/parakeet-tdt-0.6b-v3-onnx
    quantization: "int8", // default
    revision: "main",
    sessionOptions: { executionProviders: ["wasm"] },
  },
);
```

You can also prefetch without loading the model:

```js
import { downloadParakeetFromHuggingFace } from "onnx-asr-web/node";

const localDir = await downloadParakeetFromHuggingFace(
  "istupakov/parakeet-tdt-0.6b-v3-onnx",
  { cacheDir: "models", quantization: "int8" },
);
console.log(localDir);
```

Try this with 
```bash
node examples/node/transcribe.mjs --repo-id istupakov/parakeet-tdt-0.6b-v2-onnx --audio test.wav
```

### Browser usage (local/served URL)

If you run directly in a browser without bundling, you need an import map that resolves
`onnxruntime-web` to `node_modules/onnxruntime-web/dist/ort.all.min.mjs` (the example page includes this).

```js
import { createParakeetFromBaseUrl, configureOrtWeb } from "onnx-asr-web/browser";

configureOrtWeb({
  wasmPaths: "/node_modules/onnxruntime-web/dist/",
});

const asr = await createParakeetFromBaseUrl("/models/parakeet-tdt-0.6b-v3-onnx/");
const result = await asr.transcribeWavBuffer(await file.arrayBuffer());
console.log(result.text);
```

Browser loader from Hugging Face repo id:

```js
import { createParakeetFromHuggingFace } from "onnx-asr-web/browser";

const asr = await createParakeetFromHuggingFace(
  "istupakov/parakeet-tdt-0.6b-v3-onnx",
  { revision: "main", quantization: "int8" },
);
```

Serve the project root with a static server (not `file://`), for example:

```bash
npx http-server .
```

Then open `/examples/browser/index.html`.

