import * as ort from "onnxruntime-web";
import { decodeWav, normalize, resampleLinear } from "./audio.js";
import {
  DecoderTransducerModel,
  EncoderModel,
  PreprocessorModel,
  TransducerGreedyDecoder,
} from "./models.js";

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

function parseVocabulary(vocabularyText) {
  const lines = vocabularyText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const indexed = lines.every((line) => /^(.*)\s+(\d+)$/.test(line));
  if (indexed) {
    const parsed = lines.map((line) => {
      const match = line.match(/^(.*)\s+(\d+)$/);
      return { token: match[1], id: Number(match[2]) };
    });

    const maxId = parsed.reduce((acc, item) => Math.max(acc, item.id), 0);
    const tokens = new Array(maxId + 1);
    for (const item of parsed) {
      tokens[item.id] = item.token;
    }
    return tokens;
  }

  return lines;
}

function isControlToken(token) {
  return token && token.startsWith("<") && token.endsWith(">");
}

function decodeTokenPiece(token) {
  return token.replaceAll("▁", " ");
}

function decodeText(tokens, tokenIds) {
  return tokenIds
    .map((tokenId) => tokens[tokenId])
    .filter((token) => token && !isControlToken(token))
    .map((token) => decodeTokenPiece(token))
    .join("")
    .trim();
}

function wordTimestamps(tokens, tokenIds, tokenFrames, secondsPerFrame) {
  const words = [];
  let current = null;

  const closeCurrent = () => {
    if (!current) {
      return;
    }
    const cleaned = current.text.trim();
    if (cleaned) {
      words.push({
        word: cleaned,
        start: Number((current.startFrame * secondsPerFrame).toFixed(3)),
        end: Number((current.endFrame * secondsPerFrame).toFixed(3)),
      });
    }
    current = null;
  };

  for (let i = 0; i < tokenIds.length; i += 1) {
    const token = tokens[tokenIds[i]];
    const frame = tokenFrames[i];
    if (!token || !frame || isControlToken(token)) {
      continue;
    }

    const startsNewWord = token.startsWith("▁");
    const piece = decodeTokenPiece(token);

    if (startsNewWord) {
      closeCurrent();
      current = {
        text: piece.trimStart(),
        startFrame: frame.startFrame,
        endFrame: frame.endFrame,
      };
      continue;
    }

    if (!current) {
      current = {
        text: piece,
        startFrame: frame.startFrame,
        endFrame: frame.endFrame,
      };
    } else {
      current.text += piece;
      current.endFrame = frame.endFrame;
    }
  }

  closeCurrent();
  return words;
}

function detectBlankTokenId(tokens) {
  const preferred = ["<blk>", "<blank>"];
  for (const name of preferred) {
    const index = tokens.findIndex((token) => token === name);
    if (index >= 0) {
      return index;
    }
  }
  return tokens.length - 1;
}

export class AsrModel {
  constructor({ preprocessor, encoder, decoder, tokens, sampleRate = 16000 }) {
    this.preprocessor = preprocessor;
    this.encoder = encoder;
    this.decoder = decoder;
    this.tokens = tokens;
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
    if (this.preprocessor) {
      prepared = normalize(prepared);
    }

    const encoderInputs = this.preprocessor
      ? await this.preprocessor.run(prepared)
      : this.encoder.prepareInputsFromWaveform(prepared);

    const encoded = await this.encoder.run(encoderInputs.signal, encoderInputs.length);
    const decoded = await this.decoder.decode(
      encoded.encodedData,
      encoded.encodedDims,
      encoded.encodedLayout,
      encoded.encodedLength,
    );

    const tokenIds = decoded.tokenIds;
    const tokenFrames = decoded.tokenFrames;
    const secondsPerFrame =
      encoded.encodedLength > 0
        ? prepared.length / this.sampleRate / encoded.encodedLength
        : 0;

    return {
      tokenIds,
      tokenFrames,
      words: wordTimestamps(this.tokens, tokenIds, tokenFrames, secondsPerFrame),
      text: decodeText(this.tokens, tokenIds),
    };
  }

  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
}

export async function createAsrModel({
  modelType,
  decoderKind,
  config,
  preprocessorModel,
  encoderModel,
  decoderJointModel,
  vocabularyText,
  sessionOptions,
  decoderOptions,
} = {}) {
  if (!modelType || !decoderKind || !encoderModel || !decoderJointModel || !vocabularyText) {
    throw new Error(
      "createAsrModel expects modelType, decoderKind, encoderModel, decoderJointModel, and vocabularyText.",
    );
  }

  const sessionPromises = [
    ort.InferenceSession.create(encoderModel, sessionOptions),
    ort.InferenceSession.create(decoderJointModel, sessionOptions),
  ];

  if (preprocessorModel) {
    sessionPromises.unshift(ort.InferenceSession.create(preprocessorModel, sessionOptions));
  }

  const sessions = await Promise.all(sessionPromises);
  const preprocessorSession = preprocessorModel ? sessions[0] : null;
  const encoderSession = preprocessorModel ? sessions[1] : sessions[0];
  const decoderSession = preprocessorModel ? sessions[2] : sessions[1];

  const tokens = parseVocabulary(vocabularyText);
  const blankTokenId = detectBlankTokenId(tokens);
  const maxSymbols = config?.max_tokens_per_step ?? decoderOptions?.maxSymbols ?? 10;

  return new AsrModel({
    preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
    encoder: new EncoderModel(encoderSession, { config, sampleRate: 16000 }),
    decoder: new TransducerGreedyDecoder(
      new DecoderTransducerModel(decoderSession, {
        decoderKind,
        vocabSize: tokens.length,
      }),
      {
        blankTokenId,
        maxSymbols,
        ...decoderOptions,
      },
    ),
    tokens,
  });
}
