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
  if (type === "float32") {
    return new ort.Tensor("float32", new Float32Array(size), shape);
  }
  if (type === "float16") {
    return new ort.Tensor("float16", new Uint16Array(size), shape);
  }
  throw new Error(`Unsupported tensor init type '${type}'.`);
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

function firstExistingName(names, candidates, fallbackIndex = 0) {
  for (const candidate of candidates) {
    if (names.includes(candidate)) {
      return candidate;
    }
  }
  return names[fallbackIndex];
}

function metaForName(session, name) {
  return session.inputMetadata.find((meta) => meta.name === name)
    ?? session.outputMetadata.find((meta) => meta.name === name)
    ?? null;
}

function shapeFromMeta(meta, options = {}) {
  const batch = options.batch ?? 1;
  return (meta?.shape ?? []).map((dim, idx) => {
    if (typeof dim === "number") {
      return dim >= 0 ? dim : 1;
    }
    if (idx === 0 || String(dim).toLowerCase().includes("batch")) {
      return batch;
    }
    if (String(dim).toLowerCase().includes("time")) {
      return 1;
    }
    return 1;
  });
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

export class GigaamRnntModel {
  constructor({ config, tokens, encoderSession, decoderSession, jointSession }) {
    this.config = config;
    this.tokens = tokens;
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.jointSession = jointSession;
    this.sampleRate = 16000;

    this.encoderHelper = new EncoderModel(encoderSession, { config, sampleRate: this.sampleRate });
    this.blankTokenId = detectBlankTokenId(tokens);
    this.maxTokensPerStep = config?.max_tokens_per_step ?? 3;

    this.decoderTargetName = firstExistingName(decoderSession.inputNames, ["targets", "target", "tokens"], 0);
    this.decoderTargetLengthName = decoderSession.inputNames.includes("target_length")
      ? "target_length"
      : null;
    this.decoderTargetType = metaForName(decoderSession, this.decoderTargetName)?.type ?? "int32";
    this.decoderTargetLengthType = this.decoderTargetLengthName
      ? metaForName(decoderSession, this.decoderTargetLengthName)?.type ?? "int32"
      : "int32";

    this.decoderStateInputNames = decoderSession.inputNames.filter(
      (name) => name !== this.decoderTargetName && name !== this.decoderTargetLengthName,
    );
    this.decoderVectorOutputName = decoderSession.outputNames.find((name) =>
      /dec|pred|output/i.test(name),
    ) ?? decoderSession.outputNames[0];
    this.decoderStateOutputNames = decoderSession.outputNames.filter(
      (name) => name !== this.decoderVectorOutputName && !/length/i.test(name),
    );

    this.jointEncoderInputName = firstExistingName(
      jointSession.inputNames,
      ["encoder_outputs", "encoder_output", "enc_out", "encoder"],
      0,
    );
    this.jointDecoderInputName = jointSession.inputNames.find((name) => name !== this.jointEncoderInputName)
      ?? jointSession.inputNames[1];
    this.jointOutputName = jointSession.outputNames[0];
    this.jointEncoderShape = metaForName(jointSession, this.jointEncoderInputName)?.shape ?? [1, 1, 1];
    this.jointDecoderShape = metaForName(jointSession, this.jointDecoderInputName)?.shape ?? [1, 1, 1];
  }

  initialDecoderStates(batchSize = 1) {
    const states = new Map();
    for (const inputName of this.decoderStateInputNames) {
      const meta = metaForName(this.decoderSession, inputName);
      if (!meta || meta.type !== "float32") {
        continue;
      }
      const shape = shapeFromMeta(meta, { batch: batchSize });
      states.set(inputName, zerosTensor("float32", shape));
    }
    return states;
  }

  adaptJointInput(tensor, targetShape) {
    if (!Array.isArray(targetShape) || targetShape.length !== 3 || tensor.dims.length !== 3) {
      return tensor;
    }

    const expected1 = typeof targetShape[1] === "number" ? targetShape[1] : tensor.dims[1];
    const expected2 = typeof targetShape[2] === "number" ? targetShape[2] : tensor.dims[2];

    if (tensor.dims[1] === expected1 && tensor.dims[2] === expected2) {
      return tensor;
    }

    if (tensor.dims[1] === expected2 && tensor.dims[2] === expected1) {
      const [b, d1, d2] = tensor.dims;
      const out = new Float32Array(b * d2 * d1);
      for (let bi = 0; bi < b; bi += 1) {
        for (let i = 0; i < d1; i += 1) {
          for (let j = 0; j < d2; j += 1) {
            const src = bi * d1 * d2 + i * d2 + j;
            const dst = bi * d2 * d1 + j * d1 + i;
            out[dst] = tensor.data[src];
          }
        }
      }
      return new ort.Tensor("float32", out, [b, d2, d1]);
    }

    return tensor;
  }

  runJointArgmax(encoderFrameTensor, decoderVectorTensor) {
    const enc = this.adaptJointInput(encoderFrameTensor, this.jointEncoderShape);
    const dec = this.adaptJointInput(decoderVectorTensor, this.jointDecoderShape);
    return this.jointSession.run({
      [this.jointEncoderInputName]: enc,
      [this.jointDecoderInputName]: dec,
    }).then((outputs) => {
      const logits = outputs[this.jointOutputName];
      if (!logits) {
        throw new Error("GigaAM RNNT joint output tensor is missing.");
      }
      return argmaxSlice(logits.data, 0, logits.data.length);
    });
  }

  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }

    const encoderInputs = this.encoderHelper.prepareInputsFromWaveform(prepared);
    const encoded = await this.encoderHelper.run(encoderInputs.signal, encoderInputs.length);

    const [batch, channels, time] = encoded.encodedDims;
    if (batch !== 1) {
      throw new Error(`GigaAM RNNT currently expects batch=1, got ${batch}.`);
    }

    const secondsPerFrame = encoded.encodedLength > 0
      ? prepared.length / this.sampleRate / encoded.encodedLength
      : 0;

    let states = this.initialDecoderStates(1);
    let currentToken = this.blankTokenId;
    const tokenIds = [];
    const tokenFrames = [];

    const limit = Math.min(encoded.encodedLength, time);
    for (let t = 0; t < limit; t += 1) {
      const frameData = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frameData[c] = encoded.encodedData[c * time + t];
      }
      const frameTensor = new ort.Tensor("float32", frameData, [1, channels, 1]);

      for (let n = 0; n < this.maxTokensPerStep; n += 1) {
        const feeds = {
          [this.decoderTargetName]: intTensorFor(this.decoderTargetType, [currentToken], [1, 1]),
        };
        if (this.decoderTargetLengthName) {
          feeds[this.decoderTargetLengthName] = intTensorFor(this.decoderTargetLengthType, [1], [1]);
        }
        for (const [name, tensor] of states.entries()) {
          feeds[name] = tensor;
        }

        const decoderOutputs = await this.decoderSession.run(feeds);
        const decoderVector = decoderOutputs[this.decoderVectorOutputName];
        if (!decoderVector) {
          throw new Error("GigaAM RNNT decoder vector output is missing.");
        }

        const nextStates = new Map(states);
        for (const inputName of this.decoderStateInputNames) {
          const exact = decoderOutputs[inputName];
          if (exact) {
            nextStates.set(inputName, exact);
            continue;
          }

          const normalizedInput = inputName.replace(/\.\d+$/, "");
          const outputName = this.decoderStateOutputNames.find(
            (name) =>
              name === inputName ||
              name.replace(/\.\d+$/, "") === normalizedInput,
          );
          if (outputName && decoderOutputs[outputName]) {
            nextStates.set(inputName, decoderOutputs[outputName]);
          }
        }

        const nextToken = await this.runJointArgmax(frameTensor, decoderVector);
        if (nextToken === this.blankTokenId) {
          break;
        }

        tokenIds.push(nextToken);
        tokenFrames.push({ startFrame: t, endFrame: t + 1 });
        currentToken = nextToken;
        states = nextStates;
      }
    }

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

export class ToneCtcModel {
  constructor({ config, tokens, session }) {
    this.config = config;
    this.tokens = tokens;
    this.session = session;
    this.sampleRate =
      Number(config?.feature_extraction_params?.sample_rate)
      || Number(config?.sample_rate)
      || 8000;
    this.blankTokenId =
      Number.isInteger(config?.pad_token_id) ? config.pad_token_id : detectBlankTokenId(tokens);

    this.signalName = session.inputNames[0];
    this.stateName = session.inputNames[1];
    this.logitsName = session.outputNames[0];
    this.nextStateName = session.outputNames[1];

    this.signalMeta = metaForName(session, this.signalName);
    this.stateMeta = metaForName(session, this.stateName);
    this.chunkSamples = typeof this.signalMeta?.shape?.[1] === "number"
      ? this.signalMeta.shape[1]
      : 2400;
    this.stateType = this.stateMeta?.type ?? "float16";
    this.stateShape = shapeFromMeta(this.stateMeta, { batch: 1 });
  }

  toIntPcm(samples) {
    const out = new Int32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      out[i] = Math.round(v * 32767);
    }
    return out;
  }

  decodeCtcGreedy(tokens, tokenFrames, sequence, frameStartIndex) {
    let prev = this.blankTokenId;
    for (let i = 0; i < sequence.length; i += 1) {
      const token = sequence[i];
      const t = frameStartIndex + i;
      if (token === this.blankTokenId) {
        prev = this.blankTokenId;
        continue;
      }
      if (token === prev) {
        const last = tokenFrames[tokenFrames.length - 1];
        if (last) {
          last.endFrame = t + 1;
        }
        continue;
      }
      tokens.push(token);
      tokenFrames.push({ startFrame: t, endFrame: t + 1 });
      prev = token;
    }
  }

  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }

    const pcm = this.toIntPcm(prepared);
    let state = zerosTensor(this.stateType, this.stateShape);

    const tokenIds = [];
    const tokenFrames = [];
    let frameOffset = 0;
    let consumedSamples = 0;

    while (consumedSamples < pcm.length) {
      const end = Math.min(consumedSamples + this.chunkSamples, pcm.length);
      const chunk = new Int32Array(this.chunkSamples);
      chunk.set(pcm.subarray(consumedSamples, end), 0);
      const signal = new ort.Tensor("int32", chunk, [1, this.chunkSamples, 1]);

      const outputs = await this.session.run({
        [this.signalName]: signal,
        [this.stateName]: state,
      });
      const logprobs = outputs[this.logitsName];
      const nextState = outputs[this.nextStateName];
      if (!logprobs || !nextState) {
        throw new Error("Tone CTC model outputs are missing logprobs or state_next.");
      }
      state = nextState;

      const [batch, timeSteps, vocabSize] = logprobs.dims;
      if (batch !== 1) {
        throw new Error(`Tone CTC currently expects batch=1, got ${batch}.`);
      }

      const frameTokens = [];
      for (let t = 0; t < timeSteps; t += 1) {
        const offset = t * vocabSize;
        frameTokens.push(argmaxSlice(logprobs.data, offset, vocabSize));
      }
      this.decodeCtcGreedy(tokenIds, tokenFrames, frameTokens, frameOffset);
      frameOffset += timeSteps;
      consumedSamples = end;
    }

    const secondsPerFrame = frameOffset > 0
      ? prepared.length / this.sampleRate / frameOffset
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
  decoderModel,
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
  const configuredSampleRate =
    Number(config?.feature_extraction_params?.sample_rate)
    || Number(config?.sample_rate)
    || 16000;

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

  if (decoderKind === "gigaam-rnnt") {
    if (!encoderModel || !decoderModel || !decoderJointModel || !vocabularyText) {
      throw new Error(
        "gigaam rnnt requires encoderModel, decoderModel, jointModel, and vocabularyText.",
      );
    }

    const [encoderSession, decoderSession, jointSession] = await Promise.all([
      ort.InferenceSession.create(encoderModel, sessionOptions),
      ort.InferenceSession.create(decoderModel, sessionOptions),
      ort.InferenceSession.create(decoderJointModel, sessionOptions),
    ]);
    const tokens = parseVocabulary(vocabularyText);
    return new GigaamRnntModel({
      config,
      tokens,
      encoderSession,
      decoderSession,
      jointSession,
    });
  }

  if (decoderKind === "tone-ctc") {
    if (!encoderModel || !vocabularyText) {
      throw new Error("tone-ctc requires model.onnx and vocabulary.");
    }
    const session = await ort.InferenceSession.create(encoderModel, sessionOptions);
    const tokens = parseVocabulary(vocabularyText);
    return new ToneCtcModel({
      config,
      tokens,
      session,
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
    const blankTokenId =
      Number.isInteger(config?.pad_token_id) ? config.pad_token_id : detectBlankTokenId(tokens);
    return new AsrModel({
      preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
      encoder: new CtcAcousticModel(encoderSession, {
        config,
        sampleRate: configuredSampleRate,
        vocabSize: tokens.length,
      }),
      decoder: new CtcGreedyDecoder({
        blankTokenId,
      }),
      tokens,
      sampleRate: configuredSampleRate,
    });
  }

  return new AsrModel({
    preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
    encoder: new EncoderModel(encoderSession, { config, sampleRate: configuredSampleRate }),
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
    sampleRate: configuredSampleRate,
  });
}
