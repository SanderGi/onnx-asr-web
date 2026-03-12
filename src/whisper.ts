import * as ort from "onnxruntime-web";
import { decodeWav, resampleLinear } from "./audio.js";
import type { ModelConfig } from "./model-types.js";
import type { AsrTranscriber, AudioSamples, OrtTensor, TranscriptResult, TensorMap } from "./types.js";

type TokenMap = Record<string, number>;
type TokenMatrix = number[][];
type TensorMetadata = { name: string; type?: string; shape?: readonly (number | string | undefined)[] };

interface WhisperLogMelOptions {
  sampleRate?: number;
  nMels?: number;
}

interface WhisperDecodeOptions {
  language?: string;
  maxLength?: number;
}

interface WhisperBaseOptions {
  config: ModelConfig;
  vocab: TokenMap;
  addedTokens: TokenMap;
}

interface WhisperOrtOptions extends WhisperBaseOptions {
  session: ort.InferenceSession;
}

interface WhisperHfOptions extends WhisperBaseOptions {
  encoderSession: ort.InferenceSession;
  decoderSession: ort.InferenceSession;
}

function tensorValueToNumber(value: string | number | bigint): number {
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (typeof value === "string") {
    return Number(value);
  }
  return value;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildHannWindow(length: number): Float32Array {
  const window = new Float32Array(length);
  if (length === 1) {
    window[0] = 1;
    return window;
  }
  for (let i = 0; i < length; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return window;
}

function buildMelFilterBank(
  { sampleRate, nFft, nMels, fMin, fMax }: { sampleRate: number; nFft: number; nMels: number; fMin: number; fMax: number },
): Float32Array[] {
  const nFreqs = Math.floor(nFft / 2) + 1;
  const bank = Array.from({ length: nMels }, () => new Float32Array(nFreqs));

  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (nMels + 1);
  }

  const hzPoints = Array.from(melPoints, melToHz);
  const bin = hzPoints.map((hz) => Math.floor(((nFft + 1) * hz) / sampleRate));

  for (let m = 1; m <= nMels; m += 1) {
    const left = bin[m - 1];
    const center = bin[m];
    const right = bin[m + 1];

    for (let k = left; k < center; k += 1) {
      if (k >= 0 && k < nFreqs && center > left) {
        bank[m - 1][k] = (k - left) / (center - left);
      }
    }
    for (let k = center; k < right; k += 1) {
      if (k >= 0 && k < nFreqs && right > center) {
        bank[m - 1][k] = (right - k) / (right - center);
      }
    }
  }

  return bank;
}

function powerSpectrum(frame: Float32Array, nFft: number): Float32Array {
  const bins = Math.floor(nFft / 2) + 1;
  const out = new Float32Array(bins);

  for (let k = 0; k < bins; k += 1) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < nFft; n += 1) {
      const sample = frame[n] ?? 0;
      const angle = (2 * Math.PI * k * n) / nFft;
      real += sample * Math.cos(angle);
      imag -= sample * Math.sin(angle);
    }
    out[k] = real * real + imag * imag;
  }

  return out;
}

function whisperLogMelSpectrogram(samples: Float32Array, options: WhisperLogMelOptions = {}): ort.Tensor {
  const sampleRate = options.sampleRate ?? 16000;
  const nMels = options.nMels ?? 80;
  const nFft = 400;
  const hopLength = 160;
  const winLength = 400;
  const chunkSamples = 30 * sampleRate;
  const pad = Math.floor(nFft / 2);

  if (samples.length < chunkSamples) {
    const padded = new Float32Array(chunkSamples);
    padded.set(samples, 0);
    samples = padded;
  } else if (samples.length > chunkSamples) {
    samples = samples.slice(0, chunkSamples);
  }

  const padded = new Float32Array(samples.length + 2 * pad);
  padded.set(samples, pad);

  const rawFrameCount = 1 + Math.floor((padded.length - nFft) / hopLength);
  const frameCount = Math.max(1, rawFrameCount - 1);
  const frameWindow = buildHannWindow(winLength);
  const melBank = buildMelFilterBank({ sampleRate, nFft, nMels, fMin: 0, fMax: sampleRate / 2 });

  const mel = new Float32Array(nMels * frameCount);
  const fftFrame = new Float32Array(nFft);

  for (let t = 0; t < frameCount; t += 1) {
    const start = t * hopLength;
    fftFrame.fill(0);

    for (let i = 0; i < winLength; i += 1) {
      fftFrame[i] = padded[start + i] * frameWindow[i];
    }

    const spectrum = powerSpectrum(fftFrame, nFft);
    for (let m = 0; m < nMels; m += 1) {
      const filter = melBank[m];
      let energy = 0;
      for (let k = 0; k < spectrum.length; k += 1) {
        energy += spectrum[k] * filter[k];
      }
      mel[m * frameCount + t] = Math.max(1e-10, energy);
    }
  }

  let maxLog = -Infinity;
  for (let i = 0; i < mel.length; i += 1) {
    const logValue = Math.log10(mel[i]);
    mel[i] = logValue;
    if (logValue > maxLog) {
      maxLog = logValue;
    }
  }

  const floor = maxLog - 8;
  for (let i = 0; i < mel.length; i += 1) {
    mel[i] = (Math.max(mel[i], floor) + 4) / 4;
  }

  return new ort.Tensor("float32", mel, [1, nMels, frameCount]);
}

function bytesToUnicode(): Map<number, string> {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const unicode = cs.map((value) => String.fromCharCode(value));
  const out = new Map();
  for (let i = 0; i < bs.length; i += 1) {
    out.set(bs[i], unicode[i]);
  }
  return out;
}

function argmax(values: ArrayLike<number>): number {
  let idx = 0;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
      idx = i;
    }
  }
  return idx;
}

function hasAnyState(stateMap: Map<string, OrtTensor>): boolean {
  for (const value of stateMap.values()) {
    if (value.data.length > 0) {
      return true;
    }
  }
  return false;
}

function intTensorFor(type: string, values: readonly number[], dims: readonly number[]): ort.Tensor {
  if (type === "int64") {
    return new ort.Tensor("int64", BigInt64Array.from(values.map((x) => BigInt(x))), dims);
  }
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}

function boolTensor(values: readonly boolean[], dims: readonly number[]): ort.Tensor {
  return new ort.Tensor("bool", Uint8Array.from(values.map((v) => (v ? 1 : 0))), dims);
}

function tokenIdsToText(
  tokenIds: readonly number[],
  vocabById: Map<number, string>,
  byteDecoder: Map<string, number>,
): string {
  let text = "";
  for (const id of tokenIds) {
    const token = vocabById.get(id);
    if (token && !token.startsWith("<|")) {
      text += token;
    }
  }

  const bytes = [];
  for (const ch of text) {
    const value = byteDecoder.get(ch);
    if (value != null) {
      bytes.push(value);
    }
  }

  return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes)).replace(/^ /, "");
}

class WhisperBaseModel implements AsrTranscriber {
  readonly config: ModelConfig;
  readonly tokens: TokenMap;
  readonly vocabById: Map<number, string>;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly transcribeTokenId: number;
  readonly notimestampsTokenId: number;
  readonly transcribeInput: TokenMatrix;
  readonly detectLangInput: TokenMatrix;
  readonly byteDecoder: Map<string, number>;
  readonly sampleRate: number;
  readonly nMels: number;

  constructor({ config, vocab, addedTokens }: WhisperBaseOptions) {
    this.config = config;
    this.tokens = { ...vocab, ...addedTokens };
    this.vocabById = new Map();
    for (const [token, id] of Object.entries(this.tokens)) {
      this.vocabById.set(Number(id), token);
    }

    this.bosTokenId = this.tokens["<|startoftranscript|>"];
    this.eosTokenId = this.tokens["<|endoftext|>"];
    this.transcribeTokenId = this.tokens["<|transcribe|>"];
    this.notimestampsTokenId = this.tokens["<|notimestamps|>"];

    this.transcribeInput = [[
      this.bosTokenId,
      this.eosTokenId,
      this.transcribeTokenId,
      this.notimestampsTokenId,
    ]];
    this.detectLangInput = [[this.bosTokenId]];

    const unicodeMap = bytesToUnicode();
    this.byteDecoder = new Map();
    for (const [k, v] of unicodeMap.entries()) {
      this.byteDecoder.set(v, k);
    }

    this.sampleRate = 16000;
    this.nMels = Number(config.features_size ?? config.num_mel_bins ?? 80);
  }

  _prepareFeatures(samples: AudioSamples, sampleRate: number): ort.Tensor {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    return whisperLogMelSpectrogram(prepared, { sampleRate: this.sampleRate, nMels: this.nMels });
  }

  _decodeTokens(tokens: readonly number[]): string {
    return tokenIdsToText(tokens, this.vocabById, this.byteDecoder);
  }

  async _decoding(_inputFeatures: ort.Tensor, _tokens: TokenMatrix, _maxLength = 448): Promise<TokenMatrix> {
    throw new Error("WhisperBaseModel._decoding() must be implemented by subclasses.");
  }

  async _recognizeFeatures(inputFeatures: ort.Tensor, options: WhisperDecodeOptions = {}): Promise<TranscriptResult> {
    let prompt = this.transcribeInput.map((row) => row.slice());
    if (options.language) {
      const languageToken = this.tokens[`<|${options.language}|>`];
      if (languageToken != null) {
        prompt[0][1] = languageToken;
      }
    } else {
      const detected = await this._decoding(inputFeatures, this.detectLangInput, 3);
      if (detected.length > 0 && detected[0].length > 1) {
        prompt[0][1] = detected[0][1];
      }
    }

    const tokenMatrix = await this._decoding(inputFeatures, prompt, options.maxLength ?? 448);
    const tokenIds = tokenMatrix[0] ?? [];
    return {
      tokenIds,
      tokenFrames: [],
      words: [],
      text: this._decodeTokens(tokenIds),
    };
  }

  async transcribeSamples(
    samples: AudioSamples,
    sampleRate = this.sampleRate,
    options: WhisperDecodeOptions = {},
  ): Promise<TranscriptResult> {
    const features = this._prepareFeatures(samples, sampleRate);
    return this._recognizeFeatures(features, options);
  }

  async transcribeWavBuffer(arrayBuffer: ArrayBuffer, options: WhisperDecodeOptions = {}): Promise<TranscriptResult> {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
}

export class WhisperOrtModel extends WhisperBaseModel {
  readonly session: ort.InferenceSession;
  readonly inputMetadata: Map<string, TensorMetadata>;
  readonly outputName: string;

  constructor({ config, vocab, addedTokens, session }: WhisperOrtOptions) {
    super({ config, vocab, addedTokens });
    this.session = session;
    this.inputMetadata = new Map(session.inputMetadata.map((meta) => [meta.name, meta]));
    this.outputName = session.outputNames.includes("sequences") ? "sequences" : session.outputNames[0];
  }

  _paramTensor(name: string, value: number): ort.Tensor | null {
    const meta = this.inputMetadata.get(name);
    if (!meta) {
      return null;
    }
    if (meta.type === "float" || meta.type === "float32") {
      return new ort.Tensor("float32", Float32Array.from([value]), [1]);
    }
    if (meta.type === "int64") {
      return intTensorFor("int64", [value], [1]);
    }
    return intTensorFor("int32", [value], [1]);
  }

  async _decoding(inputFeatures: ort.Tensor, tokens: TokenMatrix, maxLength = 448): Promise<TokenMatrix> {
    const decoderInput = Int32Array.from(tokens.flat());
    const decoderTensor = new ort.Tensor("int32", decoderInput, [tokens.length, tokens[0].length]);

    const feeds: Record<string, ort.Tensor> = {
      input_features: inputFeatures,
      decoder_input_ids: decoderTensor,
    };

    const optionalParams = {
      max_length: maxLength,
      min_length: 0,
      num_beams: 1,
      num_return_sequences: 1,
      length_penalty: 1.0,
      repetition_penalty: 1.0,
    };

    for (const [key, value] of Object.entries(optionalParams)) {
      const tensor = this._paramTensor(key, value);
      if (tensor) {
        feeds[key] = tensor;
      }
    }

    const outputs = await this.session.run(feeds);
    const sequences = outputs[this.outputName];
    if (!sequences) {
      throw new Error("Whisper ORT decoding did not return sequences output.");
    }

    const data = Array.from(sequences.data as ArrayLike<string | number | bigint>, tensorValueToNumber);
    const shape = sequences.dims;
    if (shape.length === 3) {
      const [batch, beam, length] = shape;
      const out = [];
      for (let b = 0; b < batch; b += 1) {
        const start = b * beam * length;
        out.push(data.slice(start, start + length));
      }
      return out;
    }
    if (shape.length === 2) {
      const [batch, length] = shape;
      const out = [];
      for (let b = 0; b < batch; b += 1) {
        out.push(data.slice(b * length, (b + 1) * length));
      }
      return out;
    }

    throw new Error(`Unexpected whisper-ort sequences shape: [${shape.join(", ")}].`);
  }
}

export class WhisperHfModel extends WhisperBaseModel {
  readonly encoderSession: ort.InferenceSession;
  readonly decoderSession: ort.InferenceSession;
  readonly decoderInputMeta: Map<string, TensorMetadata>;
  readonly decoderOutputNames: readonly string[];
  readonly inputIdName: string;
  readonly encoderHiddenName: string;
  readonly useCacheBranchName: string | null;
  readonly pastInputNames: string[];

  constructor({ config, vocab, addedTokens, encoderSession, decoderSession }: WhisperHfOptions) {
    super({ config, vocab, addedTokens });
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;

    this.decoderInputMeta = new Map(decoderSession.inputMetadata.map((meta) => [meta.name, meta]));
    this.decoderOutputNames = decoderSession.outputNames;

    this.inputIdName = decoderSession.inputNames.includes("input_ids")
      ? "input_ids"
      : decoderSession.inputNames[0];
    const encoderHiddenName = decoderSession.inputNames.includes("encoder_hidden_states")
      ? "encoder_hidden_states"
      : decoderSession.inputNames.find((name) => name.includes("encoder"));
    if (!encoderHiddenName) {
      throw new Error("Whisper decoder is missing encoder hidden state input.");
    }
    this.encoderHiddenName = encoderHiddenName;
    this.useCacheBranchName = decoderSession.inputNames.includes("use_cache_branch")
      ? "use_cache_branch"
      : null;

    this.pastInputNames = decoderSession.inputNames.filter((name) => name.startsWith("past_key_values."));
  }

  _emptyStateTensor(meta: TensorMetadata): ort.Tensor {
    const shape = (meta.shape ?? []).map((dim) => {
      if (typeof dim === "number") {
        return dim >= 0 ? dim : 1;
      }
      if (typeof dim === "string") {
        const lowered = dim.toLowerCase();
        if (lowered.includes("past") || lowered.includes("sequence")) {
          return 0;
        }
      }
      return 1;
    });
    const size = shape.length === 0 ? 0 : shape.reduce((acc, v) => acc * v, 1);
    return new ort.Tensor("float32", new Float32Array(size), shape);
  }

  _createState(): Map<string, ort.Tensor> {
    const state = new Map<string, ort.Tensor>();
    for (const name of this.pastInputNames) {
      const meta = this.decoderInputMeta.get(name);
      if (!meta) {
        continue;
      }
      state.set(name, this._emptyStateTensor(meta));
    }
    return state;
  }

  _decoderInputTensor(tokens: TokenMatrix, useCache: boolean): ort.Tensor {
    const width = useCache ? 1 : tokens[0].length;
    const data = new Int32Array(tokens.length * width);
    for (let b = 0; b < tokens.length; b += 1) {
      if (useCache) {
        data[b] = tokens[b][tokens[b].length - 1];
      } else {
        data.set(tokens[b], b * width);
      }
    }
    const meta = this.decoderInputMeta.get(this.inputIdName);
    const type = meta?.type === "int64" ? "int64" : "int32";
    if (type === "int64") {
      return new ort.Tensor("int64", BigInt64Array.from(data, (x) => BigInt(x)), [tokens.length, width]);
    }
    return new ort.Tensor("int32", data, [tokens.length, width]);
  }

  async _encode(inputFeatures: ort.Tensor): Promise<OrtTensor> {
    const inputName = this.encoderSession.inputNames[0];
    const outputs = await this.encoderSession.run({ [inputName]: inputFeatures });
    return outputs[this.encoderSession.outputNames[0]];
  }

  async _decodeStep(tokens: TokenMatrix, state: Map<string, ort.Tensor>, encoderOut: OrtTensor) {
    const useCache = hasAnyState(state);
    const feeds: Record<string, ort.Tensor> = {
      [this.inputIdName]: this._decoderInputTensor(tokens, useCache),
      [this.encoderHiddenName]: encoderOut,
    };

    if (this.useCacheBranchName) {
      feeds[this.useCacheBranchName] = boolTensor([useCache], [1]);
    }

    for (const [name, value] of state.entries()) {
      feeds[name] = value;
    }

    const outputs = await this.decoderSession.run(feeds);
    const logits = outputs[this.decoderOutputNames[0]];

    const nextState = new Map<string, ort.Tensor>();
    for (const inputName of this.pastInputNames) {
      const outputName = inputName.replace("past_key_values.", "present.");
      const prev = state.get(inputName);
      const candidate = outputs[outputName] ?? prev;
      if (candidate && candidate.data.length > 0) {
        nextState.set(inputName, candidate);
      } else if (prev) {
        nextState.set(inputName, prev);
      }
    }

    return { logits, nextState };
  }

  async _decoding(inputFeatures: ort.Tensor, tokens: TokenMatrix, maxLength = 448): Promise<TokenMatrix> {
    const encoderOut = await this._encode(inputFeatures);
    let state = this._createState();
    let outputTokens = tokens.map((row) => row.slice());

    for (let step = outputTokens[0].length; step < maxLength; step += 1) {
      const { logits, nextState } = await this._decodeStep(outputTokens, state, encoderOut);
      state = nextState;

      const dims = logits.dims;
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims[dims.length - 2];
      const batch = dims[0];

      for (let b = 0; b < batch; b += 1) {
        if (outputTokens[b][outputTokens[b].length - 1] === this.eosTokenId) {
          outputTokens[b].push(this.eosTokenId);
          continue;
        }

        const offset = b * seqLen * vocabSize + (seqLen - 1) * vocabSize;
        const next = argmax(
          Array.from({ length: vocabSize }, (_, index) =>
            tensorValueToNumber((logits.data as ArrayLike<string | number | bigint>)[offset + index] ?? 0),
          ),
        );
        outputTokens[b].push(next);
      }

      if (outputTokens.every((row) => row[row.length - 1] === this.eosTokenId)) {
        break;
      }
    }

    return outputTokens;
  }
}
