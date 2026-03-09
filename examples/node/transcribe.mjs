import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import {
  loadHuggingfaceModel,
  loadHuggingfaceVadModel,
  loadLocalModel,
  loadLocalVadModel,
} from "../../src/node.js";

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const current = argv[i];
    if (current === "--model-dir") {
      args.modelDir = argv[++i];
    } else if (current === "--repo-id") {
      args.repoId = argv[++i];
    } else if (current === "--cache-dir") {
      args.cacheDir = argv[++i];
    } else if (current === "--quantization") {
      args.quantization = argv[++i];
    } else if (current === "--audio") {
      args.audio = argv[++i];
    } else if (current === "--vad-model-dir") {
      args.vadModelDir = argv[++i];
    } else if (current === "--vad-repo-id") {
      args.vadRepoId = argv[++i];
    } else if (current === "--help") {
      args.help = true;
    }
  }
  return args;
}

function printUsage() {
  console.log(
    "Usage: node examples/node/transcribe.mjs (--model-dir <path> | --repo-id <org/repo> [--cache-dir <path>]) --audio <wav> [--quantization int8|none] [--vad-model-dir <path> | --vad-repo-id <org/repo>]"
  );
  console.log("");
  console.log("Expected model files in model dir/cache:");
  console.log("  config.json");
  console.log("  encoder-model.onnx + decoder_joint-model.onnx (RNNT/TDT)");
  console.log("  or model.onnx (CTC)");
  console.log("  vocab.txt or tokens.txt");
  console.log("  optional: nemo128.onnx (required for nemo-conformer-tdt)");
  console.log("");
  console.log(
    "By default, --quantization is int8 (prefers *.int8.onnx when available)."
  );
  console.log("Optional VAD: onnx-community/silero-vad (onnx/model*.onnx).");
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help || (!args.modelDir && !args.repoId) || !args.audio) {
    printUsage();
    process.exit(args.help ? 0 : 1);
  }

  const vadModel = args.vadRepoId
    ? await loadHuggingfaceVadModel(args.vadRepoId, {
        cacheDir: args.cacheDir,
        quantization: args.quantization,
        sessionOptions: { executionProviders: ["wasm"] },
      })
    : args.vadModelDir
    ? await loadLocalVadModel(args.vadModelDir, {
        quantization: args.quantization,
        sessionOptions: { executionProviders: ["wasm"] },
      })
    : null;

  const model = args.repoId
    ? await loadHuggingfaceModel(args.repoId, {
        cacheDir: args.cacheDir,
        quantization: args.quantization,
        sessionOptions: { executionProviders: ["wasm"] },
        vadModel,
      })
    : await loadLocalModel(args.modelDir, {
        quantization: args.quantization,
        sessionOptions: { executionProviders: ["wasm"] },
        vadModel,
      });

  const wavBuffer = await readFile(args.audio);
  const { text, tokenIds, words, segments } = await model.transcribeWavBuffer(
    wavBuffer.buffer.slice(
      wavBuffer.byteOffset,
      wavBuffer.byteOffset + wavBuffer.byteLength
    )
  );

  console.log(`Audio: ${basename(args.audio)}`);
  console.log(`Tokens: ${tokenIds.length}`);
  console.log(`Text: ${text}`);
  if (vadModel !== null) {
    console.log(`VAD segments: ${segments.length}`);
  }
  if (words.length > 0) {
    console.log("Words:");
    for (const word of words) {
      console.log(
        `  [${word.start.toFixed(3)} - ${word.end.toFixed(3)}] ${word.word}`
      );
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
