# onnx-asr-web

JavaScript ONNX ASR for Node.js and browser using [`onnxruntime-web`](https://github.com/microsoft/onnxruntime/tree/main/js/web). This package was heavily inspired by the Python [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr).

## Supported Model Types

Detected automatically from `config.json`:

- `nemo-conformer-tdt`
- `nemo-conformer-rnnt`

## Install

```bash
npm install
```

## Node.js API

```js
import { loadLocalModel, loadHuggingfaceModel } from "onnx-asr-web/node";

const local = await loadLocalModel("models/istupakov/parakeet-tdt-0.6b-v3-onnx", {
  quantization: "int8", // default: prefers *.int8.onnx, falls back to *.onnx
  sessionOptions: { executionProviders: ["wasm"] },
});

const hf = await loadHuggingfaceModel("istupakov/parakeet-tdt-0.6b-v3-onnx", {
  cacheDir: "models",
  quantization: "int8",
  revision: "main",
});
```

`loadHuggingfaceModel()` downloads into `${cacheDir}/${repo_id}` and reuses cached files.

## Browser API

```js
import { configureOrtWeb, loadLocalModel, loadHuggingfaceModel } from "onnx-asr-web/browser";

configureOrtWeb({ wasmPaths: "/node_modules/onnxruntime-web/dist/" });

const modelA = await loadLocalModel("/models/parakeet-tdt-0.6b-v3-onnx/");
const modelB = await loadHuggingfaceModel("istupakov/parakeet-tdt-0.6b-v3-onnx");
```

## Transcription

```js
const result = await model.transcribeWavBuffer(await file.arrayBuffer());
console.log(result.text);
console.log(result.words); // [{word, start, end}] in seconds
```

## Model Files

`loadLocalModel()` expects `config.json` plus model files referenced by model type:

- TDT (`nemo-conformer-tdt`): `nemo128.onnx`, `encoder-model.onnx`, `decoder_joint-model.onnx`, and `vocab.txt` or `tokens.txt`
- RNNT (`nemo-conformer-rnnt`): `encoder-model.onnx`, `decoder_joint-model.onnx`, and `vocab.txt` or `tokens.txt`

When quantization is enabled (`quantization: "int8"`), `*.int8.onnx` is preferred.

For Node Hugging Face downloads, `*.onnx.data` sidecars are also fetched when present.
In browser mode, models are loaded by URL so ONNX Runtime can fetch sidecars automatically.

## Examples

### Node.js CLI
```bash
node examples/node/transcribe.mjs --repo-id istupakov/parakeet-tdt-0.6b-v3-onnx --cache-dir models --audio test.wav
```

### Browser UI
```bash
npx http-server . # then /examples/browser/index.html
```
