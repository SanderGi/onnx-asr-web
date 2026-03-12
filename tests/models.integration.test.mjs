import assert from "node:assert/strict";
import test from "node:test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { loadLocalModel, loadLocalVadModel } from "../dist/node.js";

const ROOT = resolve(process.cwd());
const MODELS_ROOT = resolve(ROOT, "models");
const TEST_WAV_EN = resolve(ROOT, "test.wav");
const TEST_WAV_RU = resolve(ROOT, "test_ru.wav");
const TEST_WAV_SILENCE = resolve(ROOT, "test_with_silence.wav");

function normalizeText(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function fileBuffer(path) {
  return readFile(path).then((buf) =>
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
}

function hasModelDir(...parts) {
  return existsSync(join(MODELS_ROOT, ...parts));
}

/**
 * Model cases:
 * - `contains`: all terms must appear in normalized transcript.
 * - `minTokens`: minimum token count sanity check.
 */
const CASES = [
  {
    name: "nemo-conformer-rnnt/parakeet-rnnt",
    modelDir: join(MODELS_ROOT, "istupakov/parakeet-rnnt-0.6b-onnx"),
    wavPath: TEST_WAV_EN,
    contains: ["actor", "away"],
    minTokens: 8,
  },
  {
    name: "nemo-conformer-tdt/parakeet-tdt-v2",
    modelDir: join(MODELS_ROOT, "istupakov/parakeet-tdt-0.6b-v2-onnx"),
    wavPath: TEST_WAV_EN,
    contains: ["actor", "away"],
    minTokens: 8,
  },
  {
    name: "nemo-conformer-ctc/parakeet-ctc",
    modelDir: join(MODELS_ROOT, "istupakov/parakeet-ctc-0.6b-onnx"),
    wavPath: TEST_WAV_EN,
    contains: ["actor", "away"],
    minTokens: 6,
  },
  {
    name: "nemo-conformer/fastconformer",
    modelDir: join(MODELS_ROOT, "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"),
    wavPath: TEST_WAV_RU,
    contains: ["привет"],
    minTokens: 1,
  },
  {
    name: "nemo-conformer-aed/canary",
    modelDir: join(MODELS_ROOT, "istupakov/canary-180m-flash-onnx"),
    wavPath: TEST_WAV_EN,
    contains: ["actor", "away"],
    minTokens: 8,
  },
  {
    name: "gigaam-v2-rnnt",
    modelDir: join(MODELS_ROOT, "istupakov/gigaam-v2-onnx"),
    wavPath: TEST_WAV_RU,
    contains: ["привет"],
    minTokens: 2,
  },
  {
    name: "gigaam-v3-rnnt",
    modelDir: join(MODELS_ROOT, "istupakov/gigaam-v3-onnx"),
    wavPath: TEST_WAV_RU,
    contains: ["привет"],
    minTokens: 2,
  },
  {
    name: "tone-ctc/t-tech",
    modelDir: join(MODELS_ROOT, "t-tech/T-one"),
    wavPath: TEST_WAV_RU,
    contains: ["прив"],
    minTokens: 2,
  },
  {
    name: "whisper-ort/base",
    modelDir: join(MODELS_ROOT, "istupakov/whisper-base-onnx"),
    wavPath: TEST_WAV_EN,
    contains: ["away"],
    minTokens: 20,
  },
];

for (const spec of CASES) {
  test(spec.name, async (t) => {
    if (!existsSync(spec.modelDir)) {
      t.skip(`Missing local model dir: ${spec.modelDir}`);
      return;
    }

    const model = await loadLocalModel(spec.modelDir, {
      quantization: "int8",
      sessionOptions: { executionProviders: ["wasm"] },
    });

    const result = await model.transcribeWavBuffer(await fileBuffer(spec.wavPath));
    const normalized = normalizeText(result.text);

    assert.ok(
      result.tokenIds.length >= spec.minTokens,
      `Expected at least ${spec.minTokens} tokens, got ${result.tokenIds.length}. Text="${result.text}"`,
    );
    for (const term of spec.contains) {
      assert.ok(
        normalized.includes(normalizeText(term)),
        `Expected transcript to include "${term}". Actual="${result.text}"`,
      );
    }
  });
}

test("sherpa-transducer/alphacep-vosk-small-ru", async (t) => {
  const modelDir = join(MODELS_ROOT, "alphacep/vosk-model-small-ru");
  if (!existsSync(modelDir)) {
    t.skip(`Missing optional sherpa model dir: ${modelDir}`);
    return;
  }

  const model = await loadLocalModel(modelDir, {
    quantization: "int8",
    sessionOptions: { executionProviders: ["wasm"] },
  });
  const result = await model.transcribeWavBuffer(await fileBuffer(TEST_WAV_RU));
  assert.ok(
    normalizeText(result.text).includes("привет"),
    `Expected sherpa transcript to include "привет". Actual="${result.text}"`,
  );
});

test("vad-chunking/silero-with-parakeet-rnnt", async (t) => {
  const asrDir = join(MODELS_ROOT, "istupakov/parakeet-rnnt-0.6b-onnx");
  const vadDir = join(MODELS_ROOT, "onnx-community/silero-vad");
  if (!existsSync(asrDir) || !existsSync(vadDir)) {
    t.skip(`Missing optional VAD/asr dirs: ${asrDir}, ${vadDir}`);
    return;
  }

  let vad;
  let asr;
  try {
    vad = await loadLocalVadModel(vadDir, {
      quantization: "none",
      sessionOptions: { executionProviders: ["wasm"] },
    });
    asr = await loadLocalModel(asrDir, {
      quantization: "int8",
      sessionOptions: { executionProviders: ["wasm"] },
      vadModel: vad,
    });
  } catch (error) {
    if (String(error?.message ?? error).includes("bad_alloc")) {
      t.skip("Insufficient memory to allocate VAD+ASR sessions in this process.");
      return;
    }
    throw error;
  }

  const result = await asr.transcribeWavBuffer(await fileBuffer(TEST_WAV_SILENCE));
  assert.ok(Array.isArray(result.segments), "Expected VAD segments in result.");
  assert.ok(result.segments.length >= 1, "Expected at least one detected VAD segment.");
  assert.ok(
    normalizeText(result.text).includes("actor") || normalizeText(result.text).includes("away"),
    `Expected speech text in VAD transcript. Actual="${result.text}"`,
  );
});
