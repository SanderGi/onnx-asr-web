import * as ort from "onnxruntime-web";
import { decodeWav, normalize, resampleLinear } from "./audio.js";
import {
  DecoderTdtModel,
  EncoderModel,
  PreprocessorModel,
  TdtGreedyDecoder,
} from "./models.js";
import { Vocabulary } from "./vocabulary.js";

const DEFAULT_MODEL_FILES = {
  preprocessor: "nemo128.onnx",
  encoder: "encoder-model.onnx",
  decoderJoint: "decoder_joint-model.onnx",
  tokens: "vocab.txt",
};

export function configureOrtWeb(options = {}) {
  if (options.numThreads != null) {
    ort.env.wasm.numThreads = options.numThreads;
  }
  if (options.wasmPaths) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.simd != null) {
    ort.env.wasm.simd = options.simd;
  }
  if (options.proxy != null) {
    ort.env.wasm.proxy = options.proxy;
  }
}

export class AsrModel {
  constructor({
    preprocessor,
    encoder,
    decoder,
    vocabulary,
    sampleRate = 16000,
  }) {
    this.preprocessor = preprocessor;
    this.encoder = encoder;
    this.decoder = decoder;
    this.vocabulary = vocabulary;
    this.sampleRate = sampleRate;
  }

  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }

    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    prepared = normalize(prepared);

    const preprocessed = await this.preprocessor.run(prepared);
    const encoded = await this.encoder.run(
      preprocessed.signal,
      preprocessed.length
    );
    const decoded = await this.decoder.decode(
      encoded.encodedData,
      encoded.encodedDims,
      encoded.encodedLayout,
      encoded.encodedLength
    );
    const tokenIds = decoded.tokenIds;
    const tokenFrames = decoded.tokenFrames;
    const secondsPerFrame =
      encoded.encodedLength > 0
        ? prepared.length / this.sampleRate / encoded.encodedLength
        : 0;
    const words = this.vocabulary.wordTimestamps(
      tokenIds,
      tokenFrames,
      secondsPerFrame
    );

    return {
      tokenIds,
      tokenFrames,
      words,
      text: this.vocabulary.decode(tokenIds),
    };
  }

  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
}

export async function createParakeetAsrModel({
  preprocessorModel,
  encoderModel,
  decoderJointModel,
  tokensText,
  sessionOptions,
  decoderOptions,
} = {}) {
  if (
    !preprocessorModel ||
    !encoderModel ||
    !decoderJointModel ||
    !tokensText
  ) {
    throw new Error(
      "createParakeetAsrModel expects preprocessorModel, encoderModel, decoderJointModel, and tokensText."
    );
  }

  const [preprocessorSession, encoderSession, decoderSession] =
    await Promise.all([
      ort.InferenceSession.create(preprocessorModel, sessionOptions),
      ort.InferenceSession.create(encoderModel, sessionOptions),
      ort.InferenceSession.create(decoderJointModel, sessionOptions),
    ]);

  const parsedTokens = tokensText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(.*)\s+(\d+)$/);
      if (!match) {
        throw new Error(`Invalid token line: ${line}`);
      }
      return { token: match[1], id: Number(match[2]) };
    });

  const maxId = parsedTokens.reduce((acc, item) => Math.max(acc, item.id), 0);
  const tokens = new Array(maxId + 1);
  for (const item of parsedTokens) {
    tokens[item.id] = item.token;
  }
  const blankTokenId =
    parsedTokens.find((item) => item.token === "<blk>")?.id ?? maxId;

  return new AsrModel({
    preprocessor: new PreprocessorModel(preprocessorSession),
    encoder: new EncoderModel(encoderSession),
    decoder: new TdtGreedyDecoder(new DecoderTdtModel(decoderSession), {
      blankTokenId,
      vocabSize: tokens.length,
      ...decoderOptions,
    }),
    vocabulary: new Vocabulary(tokens),
  });
}

export { DEFAULT_MODEL_FILES };
