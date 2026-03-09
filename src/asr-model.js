import * as ort from "onnxruntime-web";
import { decodeWav, normalize, resampleLinear } from "./audio.js";
import {
  CtcAcousticModel,
  CtcGreedyDecoder,
  DecoderTransducerModel,
  EncoderModel,
  PreprocessorModel,
  TransducerGreedyDecoder,
} from "./models.js";
import { WhisperHfModel, WhisperOrtModel } from "./whisper.js";

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

function intTensorFor(type, values, dims) {
  if (type === "int64") {
    return new ort.Tensor("int64", BigInt64Array.from(values.map((x) => BigInt(x))), dims);
  }
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}

function zerosTensor(type, shape) {
  const size = shape.reduce((acc, value) => acc * value, 1);
  if (type !== "float32") {
    throw new Error(`Unsupported tensor init type '${type}'.`);
  }
  return new ort.Tensor("float32", new Float32Array(size), shape);
}

function argmaxSlice(data, start, length) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < length; i += 1) {
    const value = data[start + i];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export class NemoAedModel {
  constructor({ config, tokens, encoderSession, decoderSession }) {
    this.config = config;
    this.tokens = tokens;
    this.tokenToId = new Map();
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i];
      if (token != null && !this.tokenToId.has(token)) {
        this.tokenToId.set(token, i);
      }
    }

    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.encoderHelper = new EncoderModel(encoderSession, { config, sampleRate: 16000 });
    this.sampleRate = 16000;
    this.maxSequenceLength = config?.max_sequence_length ?? 1024;

    this.encoderOutputEmbeddingsName = encoderSession.outputNames.includes("encoder_embeddings")
      ? "encoder_embeddings"
      : encoderSession.outputNames[0];
    this.encoderOutputMaskName = encoderSession.outputNames.includes("encoder_mask")
      ? "encoder_mask"
      : encoderSession.outputNames[1];

    this.decoderInputIdName = decoderSession.inputNames.includes("input_ids")
      ? "input_ids"
      : decoderSession.inputNames[0];
    this.decoderEncoderEmbeddingsName = decoderSession.inputNames.includes("encoder_embeddings")
      ? "encoder_embeddings"
      : decoderSession.inputNames[1];
    this.decoderEncoderMaskName = decoderSession.inputNames.includes("encoder_mask")
      ? "encoder_mask"
      : decoderSession.inputNames[2];
    this.decoderMemsName = decoderSession.inputNames.includes("decoder_mems")
      ? "decoder_mems"
      : decoderSession.inputNames[3];

    this.decoderInputIdType =
      decoderSession.inputMetadata.find((meta) => meta.name === this.decoderInputIdName)?.type ?? "int32";
    const memMeta = decoderSession.inputMetadata.find((meta) => meta.name === this.decoderMemsName);
    if (!memMeta) {
      throw new Error("Decoder input metadata for decoder_mems is missing.");
    }
    this.decoderMemsType = memMeta.type;
    this.decoderMemsShapeTemplate = memMeta.shape;

    this.logitsName = decoderSession.outputNames.includes("logits")
      ? "logits"
      : decoderSession.outputNames[0];
    this.decoderHiddenStatesName = decoderSession.outputNames.includes("decoder_hidden_states")
      ? "decoder_hidden_states"
      : decoderSession.outputNames[1];
  }

  canaryPrefix(options = {}) {
    const fallbackLanguage = options.language ?? "en";
    const targetLanguage = options.targetLanguage ?? "en";
    const pnc = options.pnc ?? "yes";

    const values = [
      "<|startofcontext|>",
      "<|startoftranscript|>",
      "<|emo:undefined|>",
      `<|${fallbackLanguage}|>`,
      `<|${targetLanguage}|>`,
      pnc === "yes" ? "<|pnc|>" : "<|nopnc|>",
      "<|noitn|>",
      "<|notimestamp|>",
      "<|nodiarize|>",
    ];

    return values.map((token) => {
      const id = this.tokenToId.get(token);
      if (id == null) {
        throw new Error(`Required Canary token not found in vocab: ${token}`);
      }
      return id;
    });
  }

  initialMems(batchSize) {
    const shape = this.decoderMemsShapeTemplate.map((dim, index) => {
      if (typeof dim === "number") {
        return dim;
      }
      if (index === 1) {
        return batchSize;
      }
      if (index === 2) {
        return 0;
      }
      return 1;
    });
    return zerosTensor(this.decoderMemsType, shape);
  }

  async transcribeSamples(samples, sampleRate = this.sampleRate, options = {}) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }

    const encoderInputs = this.encoderHelper.prepareInputsFromWaveform(prepared);
    const encoderOutputs = await this.encoderSession.run({
      [this.encoderHelper.audioSignalName]: encoderInputs.signal,
      [this.encoderHelper.lengthName]: encoderInputs.length,
    });

    const encoderEmbeddings = encoderOutputs[this.encoderOutputEmbeddingsName];
    const encoderMask = encoderOutputs[this.encoderOutputMaskName];
    if (!encoderEmbeddings || !encoderMask) {
      throw new Error("Canary encoder outputs are missing required tensors.");
    }

    const prefix = this.canaryPrefix(options);
    const batchTokens = [prefix.slice()];
    const prefixLength = batchTokens[0].length;
    const eosId = this.tokenToId.get("<|endoftext|>");
    if (eosId == null) {
      throw new Error("Canary vocab is missing <|endoftext|> token.");
    }

    let decoderMems = this.initialMems(batchTokens.length);
    while (batchTokens[0].length < this.maxSequenceLength) {
      const inputIds = decoderMems.dims[2] === 0
        ? batchTokens.flat()
        : batchTokens.map((row) => row[row.length - 1]);
      const sequenceLength = decoderMems.dims[2] === 0 ? batchTokens[0].length : 1;
      const inputTensor = intTensorFor(this.decoderInputIdType, inputIds, [batchTokens.length, sequenceLength]);

      const decoderOutputs = await this.decoderSession.run({
        [this.decoderInputIdName]: inputTensor,
        [this.decoderEncoderEmbeddingsName]: encoderEmbeddings,
        [this.decoderEncoderMaskName]: encoderMask,
        [this.decoderMemsName]: decoderMems,
      });

      const logits = decoderOutputs[this.logitsName];
      const nextMems = decoderOutputs[this.decoderHiddenStatesName];
      if (!logits || !nextMems) {
        throw new Error("Canary decoder outputs are missing logits or decoder state.");
      }

      decoderMems = nextMems;

      const [batchSize, seq, vocab] = logits.dims;
      let allEos = true;
      for (let b = 0; b < batchSize; b += 1) {
        const offset = b * seq * vocab + (seq - 1) * vocab;
        const nextToken = argmaxSlice(logits.data, offset, vocab);
        batchTokens[b].push(nextToken);
        if (nextToken !== eosId) {
          allEos = false;
        }
      }

      if (allEos) {
        break;
      }
    }

    const tokenIds = batchTokens[0]
      .slice(prefixLength)
      .filter((id) => this.tokens[id] && !this.tokens[id].startsWith("<|"));

    return {
      tokenIds,
      tokenFrames: [],
      words: [],
      text: decodeText(this.tokens, tokenIds),
    };
  }

  async transcribeWavBuffer(arrayBuffer, options = {}) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
}

export async function createAsrModel({
  modelType,
  decoderKind,
  config,
  preprocessorModel,
  encoderModel,
  decoderJointModel,
  whisperModel,
  vocabularyText,
  vocabJson,
  addedTokensJson,
  sessionOptions,
  decoderOptions,
} = {}) {
  if (!modelType || !decoderKind) {
    throw new Error(
      "createAsrModel expects modelType and decoderKind.",
    );
  }
  if (decoderKind === "whisper-ort") {
    if (!whisperModel || !vocabJson) {
      throw new Error("whisper-ort requires whisperModel and vocabJson.");
    }
    const session = await ort.InferenceSession.create(whisperModel, sessionOptions);
    return new WhisperOrtModel({
      config,
      vocab: JSON.parse(vocabJson),
      addedTokens: addedTokensJson ? JSON.parse(addedTokensJson) : {},
      session,
    });
  }

  if (decoderKind === "whisper-hf") {
    if (!encoderModel || !decoderJointModel || !vocabJson) {
      throw new Error("whisper requires encoderModel, decoderJointModel, and vocabJson.");
    }
    const [encoderSession, decoderSession] = await Promise.all([
      ort.InferenceSession.create(encoderModel, sessionOptions),
      ort.InferenceSession.create(decoderJointModel, sessionOptions),
    ]);
    return new WhisperHfModel({
      config,
      vocab: JSON.parse(vocabJson),
      addedTokens: addedTokensJson ? JSON.parse(addedTokensJson) : {},
      encoderSession,
      decoderSession,
    });
  }

  if (decoderKind === "aed") {
    if (!encoderModel || !decoderJointModel || !vocabularyText) {
      throw new Error("nemo-conformer-aed requires encoderModel, decoderModel, and vocabularyText.");
    }
    const [encoderSession, decoderSession] = await Promise.all([
      ort.InferenceSession.create(encoderModel, sessionOptions),
      ort.InferenceSession.create(decoderJointModel, sessionOptions),
    ]);
    const tokens = parseVocabulary(vocabularyText);
    return new NemoAedModel({
      config,
      tokens,
      encoderSession,
      decoderSession,
    });
  }

  if (!encoderModel || !vocabularyText) {
    throw new Error("createAsrModel expects encoderModel and vocabularyText for non-whisper models.");
  }
  if (decoderKind !== "ctc" && !decoderJointModel) {
    throw new Error("Transducer models require decoderJointModel.");
  }

  const sessionPromises = [ort.InferenceSession.create(encoderModel, sessionOptions)];
  if (decoderJointModel) {
    sessionPromises.push(ort.InferenceSession.create(decoderJointModel, sessionOptions));
  }

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

  if (decoderKind === "ctc") {
    return new AsrModel({
      preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
      encoder: new CtcAcousticModel(encoderSession, { config, sampleRate: 16000, vocabSize: tokens.length }),
      decoder: new CtcGreedyDecoder({
        blankTokenId,
      }),
      tokens,
    });
  }

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
