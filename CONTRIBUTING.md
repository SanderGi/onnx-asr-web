# Contributing

Thanks for contributing to `onnx-asr-web`.

## Development Setup

1. Clone the repo.
2. Install dependencies:

```bash
npm install
```

3. Run static checks:

```bash
npm run check
```

## Build

Build distributable files to `dist/`:

```bash
npm run build
```

Artifacts:

- `dist/index.js` (package root entry)
- `dist/node.js` (Node API entry)
- `dist/browser.js` (Browser API entry)

## Local Examples

Node CLI:

```bash
node examples/node/transcribe.mjs --model-dir models/istupakov/parakeet-rnnt-0.6b-onnx --audio test.wav
```

Browser examples:

```bash
npx http-server .
```

Then open:

- `/examples/browser/index.html` (local source import)
- `/examples/browser-cdn/index.html` (CDN import example)

## Adding Model Architectures

When adding architecture support:

1. Add detection in `src/model-types.js`.
2. Add loading logic in `src/node.js` and `src/browser.js`.
3. Add runtime inference in `src/asr-model.js` or a dedicated module.
4. Keep Node and browser APIs aligned.
5. Validate with `examples/node/transcribe.mjs` using a real model repo.

## Pull Requests

Before opening a PR:

1. Run `npm run check`.
2. Run `npm run build`.
3. Update `README.md` if API behavior changed.
4. Include exact model repo ids and commands used for validation.
