<p align="center">
  <img src="https://raw.githubusercontent.com/SanderGi/onnx-asr-web/main/logo.png" alt="onnx-asr-web logo" width="100"/>
</p>

# onnx-asr-web

[![npm version](https://img.shields.io/npm/v/onnx-asr-web)](https://www.npmjs.com/package/onnx-asr-web)
[![npm downloads](https://img.shields.io/npm/dm/onnx-asr-web)](https://www.npmjs.com/package/onnx-asr-web)
[![license](https://img.shields.io/npm/l/onnx-asr-web)](./LICENSE)

JavaScript ONNX ASR for Node.js and browser using [`onnxruntime-web`](https://github.com/microsoft/onnxruntime/tree/main/js/web). This package was heavily inspired by the Python [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr) and aims to be the minimalistic way to achieve state of the art automatic speech recognition with JavaScript.

[Online Demo Here!](https://sandergi.com/cdn/onnx-asr-web/example/index.html)

## Features

- Loads models from Hugging Face or local directories
- Autodetects model-type from files
- Supports quantized models
- Works with WAV files/buffers
- Uses Voice Activity Detection (VAD) to do long-form speech-to-text
- Extracts word-level timestamps
- Minimal dependencies (just `onnxruntime-web`)

## Supported Model Types

- Nvidia Parakeet, Canary, FastFormer, and Conformer
- OpenAI Whisper
- GigaChat GigaAM
- Kaldi Icefall Zipformer
- T-Tech T-one
- Custom CTC, RNNT, TDT, and Transformer models

## Install

```bash
npm install onnx-asr-web
```

`onnxruntime-web` must be `1.24.x` or newer. Earlier versions can fail on some models (notably browser VAD graphs).

## Node.js API

```js
import {
  loadLocalModel,
  loadHuggingfaceModel,
  loadLocalVadModel,
  loadHuggingfaceVadModel,
} from "onnx-asr-web/node";

const vad = await loadHuggingfaceVadModel("onnx-community/silero-vad", {
  cacheDir: "models",
  quantization: "int8",
});

const local = await loadLocalModel("models/istupakov/parakeet-tdt-0.6b-v3-onnx", {
  quantization: "int8", // default: prefers *.int8.onnx, falls back to *.onnx
  sessionOptions: { executionProviders: ["wasm"] },
  vadModel: vad, // optional: chunks long audio by non-speech
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
import {
  configureOrtWeb,
  loadLocalModel,
  loadHuggingfaceModel,
  loadHuggingfaceVadModel,
} from "onnx-asr-web/browser";

configureOrtWeb({ wasmPaths: "/node_modules/onnxruntime-web/dist/" });

const vad = await loadHuggingfaceVadModel("onnx-community/silero-vad");
const modelA = await loadLocalModel("/models/parakeet-tdt-0.6b-v3-onnx/", { vadModel: vad });
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
- CTC (`nemo-conformer-ctc`): `model.onnx` and `vocab.txt` or `tokens.txt`
- Canary AED (`nemo-conformer-aed`): `encoder-model.onnx`, `decoder-model.onnx`, and `vocab.txt` or `tokens.txt`
- FastConformer (`nemo-conformer`): prefers RNNT split (`encoder-model.onnx` + `decoder_joint-model.onnx`) and falls back to CTC `model.onnx`, with `vocab.txt` or `tokens.txt`
- GigaAM (`gigaam`): auto-detects `v2_*`/`v3_*` files, prefers RNNT triplet (`*_rnnt_encoder/decoder/joint`) and falls back to CTC (`*_ctc.onnx`), with `v2_vocab.txt`/`v3_vocab.txt`
- Tone CTC (`tone-ctc`): `model.onnx` with vocab from `decoder_params.vocabulary` in `config.json` (or `vocab.json`)
- Whisper ORT (`whisper-ort`): `*_beamsearch.onnx` model, plus `vocab.json` (and optionally `added_tokens.json`)
- Whisper HF (`whisper`): `onnx/encoder_model*.onnx`, `onnx/decoder_model_merged*.onnx`, plus `vocab.json` (and optionally `added_tokens.json`)
- Sherpa transducer (no config): `am-onnx/` (or `am/`) with `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, plus `lang/tokens.txt` (or `tokens.txt`)
- VAD (`onnx-community/silero-vad`): `onnx/model*.onnx` (e.g. `onnx/model_int8.onnx`)

When quantization is enabled (`quantization: "int8"`), `*.int8.onnx` is preferred.

For Node Hugging Face downloads, `*.onnx.data` sidecars are also fetched when present.
In browser mode, models are loaded by URL so ONNX Runtime can fetch sidecars automatically.

When `vadModel` is supplied to `loadLocalModel()` / `loadHuggingfaceModel()`, transcription runs on VAD speech chunks and returns a `segments` array in output.

Word timestamps are currently provided for NeMo transducer models. Whisper returns transcript text and token IDs; `words` is empty.

## Examples

### Node.js CLI
```bash
node examples/node/transcribe.mjs --repo-id istupakov/parakeet-tdt-0.6b-v3-onnx --cache-dir models --audio test.wav
```

### Browser UI
```bash
npx http-server . # then /examples/browser/index.html
```

### Browser UI (CDN package import)
```bash
npx http-server . # then /examples/browser-cdn/index.html
```

## Build and Publish

Create distributable artifacts:

```bash
npm run build
```

This produces:

- `dist/index.js`
- `dist/node.js`
- `dist/browser.js`

Publish to npm:

```bash
npm publish
```

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md).
