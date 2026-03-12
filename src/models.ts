import * as ort from "onnxruntime-web";
import type { ModelConfig } from "./model-types.js";
import type { OrtTensor, TensorData, TensorMap, TokenFrame } from "./types.js";
import { firstExistingInputName, int64TensorValues, readScalarInt } from "./utils.js";

type TensorShape = readonly number[];
type MetadataShape = readonly (number | string | undefined)[] | undefined;
type EncoderLayout = "BCT" | "BTV" | "BVT";
type TensorMetadata = { name: string; type?: string; shape?: MetadataShape };

interface LogMelSpectrogramOptions {
  sampleRate?: number;
  nMels?: number;
  nFft?: number;
  winLength?: number;
  hopLength?: number;
  fMin?: number;
  fMax?: number;
  logEps?: number;
  preemphasis?: number;
  normalize?: boolean;
}

interface EncoderModelOptions {
  config?: ModelConfig;
  sampleRate?: number;
}

interface DecoderTransducerOptions {
  decoderKind?: "tdt" | "rnnt";
  vocabSize?: number;
}

interface TransducerCandidate {
  token: number;
  duration: number;
}

interface DecoderPrediction {
  candidates: TransducerCandidate[];
  nextStates: Map<string, OrtTensor>;
}

interface EncodedAudio {
  encodedData: TensorData;
  encodedDims: TensorShape;
  encodedLayout: EncoderLayout;
  encodedLength: number;
}

interface DecoderResult {
  tokenIds: number[];
  tokenFrames: TokenFrame[];
  totalFrames: number;
}

interface TransducerDecoderLike {
  initialStates(): Map<string, OrtTensor>;
  predict(
    encoderFrameData: Float32Array,
    encoderFrameDims: TensorShape,
    token: number,
    states: Map<string, OrtTensor>,
  ): Promise<DecoderPrediction>;
}

interface TransducerGreedyDecoderOptions {
  maxSymbols?: number;
  blankTokenId?: number;
  defaultDuration?: number;
}

interface CtcGreedyDecoderOptions {
  blankTokenId?: number;
}

function intTensor(type: string, values: readonly number[], dims: readonly number[]): ort.Tensor {
  if (type === "int64") {
    return new ort.Tensor("int64", int64TensorValues(values), dims);
  }
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}

function ensureOutput(session: ort.InferenceSession, outputMap: TensorMap, index: number): OrtTensor {
  const name = session.outputNames[index];
  const tensor = outputMap[name];
  if (!tensor) {
    throw new Error(`Missing output tensor at index ${index} (${name}).`);
  }
  return tensor;
}

function valueToNumber(value: string | number | bigint): number {
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (typeof value === "string") {
    return Number(value);
  }
  return value;
}

function numbersFromTensor(tensor: OrtTensor): number[] {
  return Array.from(tensor.data as ArrayLike<string | number | bigint>, valueToNumber);
}

function shapeFromMetadata(metadataShape: MetadataShape): number[] {
  if (!Array.isArray(metadataShape) || metadataShape.length === 0) {
    return [1];
  }

  return metadataShape.map((dimension) =>
    typeof dimension === "number" && dimension > 0 ? dimension : 1,
  );
}

function product(values: readonly number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
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
  { sampleRate, nFft, nMels, fMin, fMax }: Required<Pick<LogMelSpectrogramOptions, "sampleRate" | "nFft" | "nMels" | "fMin" | "fMax">>,
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

function logMelSpectrogram(samples: Float32Array, options: LogMelSpectrogramOptions = {}) {
  const sampleRate = options.sampleRate ?? 16000;
  const nMels = options.nMels ?? 80;
  const nFft = options.nFft ?? 512;
  const winLength = options.winLength ?? 400;
  const hopLength = options.hopLength ?? 160;
  const fMin = options.fMin ?? 0;
  const fMax = options.fMax ?? sampleRate / 2;
  const logEps = options.logEps ?? 1e-10;
  const preemphasis = options.preemphasis ?? 0.97;
  const normalize = options.normalize ?? true;

  if (preemphasis > 0) {
    const emphasized = new Float32Array(samples.length);
    if (samples.length > 0) {
      emphasized[0] = samples[0];
    }
    for (let i = 1; i < samples.length; i += 1) {
      emphasized[i] = samples[i] - preemphasis * samples[i - 1];
    }
    samples = emphasized;
  }

  if (samples.length < winLength) {
    const padded = new Float32Array(winLength);
    padded.set(samples, 0);
    samples = padded;
  }

  const frameCount = 1 + Math.floor((samples.length - winLength) / hopLength);
  const frameWindow = buildHannWindow(winLength);
  const melBank = buildMelFilterBank({ sampleRate, nFft, nMels, fMin, fMax });

  const features = new Float32Array(nMels * frameCount);
  const fftFrame = new Float32Array(nFft);

  for (let t = 0; t < frameCount; t += 1) {
    const start = t * hopLength;
    fftFrame.fill(0);

    for (let i = 0; i < winLength; i += 1) {
      fftFrame[i] = samples[start + i] * frameWindow[i];
    }

    const spectrum = powerSpectrum(fftFrame, nFft);
    for (let m = 0; m < nMels; m += 1) {
      const filter = melBank[m];
      let energy = 0;
      for (let k = 0; k < spectrum.length; k += 1) {
        energy += spectrum[k] * filter[k];
      }
      features[m * frameCount + t] = Math.log(Math.max(logEps, energy));
    }
  }

  if (normalize) {
    // NeMo default behavior uses per-feature normalization.
    for (let m = 0; m < nMels; m += 1) {
      const offset = m * frameCount;
      let mean = 0;
      for (let t = 0; t < frameCount; t += 1) {
        mean += features[offset + t];
      }
      mean /= frameCount;

      let variance = 0;
      for (let t = 0; t < frameCount; t += 1) {
        const delta = features[offset + t] - mean;
        variance += delta * delta;
      }
      const std = Math.sqrt(variance / frameCount) + 1e-5;
      for (let t = 0; t < frameCount; t += 1) {
        features[offset + t] = (features[offset + t] - mean) / std;
      }
    }
  }

  return {
    features,
    frameCount,
    nMels,
  };
}

export class PreprocessorModel {
  readonly session: ort.InferenceSession;
  readonly inputSignalName: string;
  readonly lengthName: string;
  readonly lengthType: string;

  constructor(session: ort.InferenceSession) {
    this.session = session;
    this.inputSignalName = firstExistingInputName(session, ["input_signal", "waveforms"], 0);
    this.lengthName = firstExistingInputName(session, ["length", "waveforms_lens"], 1);

    this.lengthType = (session.inputMetadata.find((item) => item.name === this.lengthName) as TensorMetadata | undefined)?.type ?? "int64";
  }

  async run(audioSamples: Float32Array): Promise<{ signal: OrtTensor; length: OrtTensor }> {
    const audioTensor = new ort.Tensor("float32", audioSamples, [1, audioSamples.length]);
    const lengthTensor = intTensor(this.lengthType, [audioSamples.length], [1]);

    const outputs = await this.session.run({
      [this.inputSignalName]: audioTensor,
      [this.lengthName]: lengthTensor,
    });

    return {
      signal: ensureOutput(this.session, outputs, 0),
      length: ensureOutput(this.session, outputs, 1),
    };
  }
}

export class EncoderModel {
  readonly session: ort.InferenceSession;
  readonly options: EncoderModelOptions;
  readonly audioSignalName: string;
  readonly lengthName: string;
  readonly audioMetadata: TensorMetadata | undefined;
  readonly lengthMetadata: TensorMetadata | undefined;

  constructor(session: ort.InferenceSession, options: EncoderModelOptions = {}) {
    this.session = session;
    this.options = options;
    this.audioSignalName = firstExistingInputName(session, ["audio_signal", "features", "waveforms"], 0);
    this.lengthName = firstExistingInputName(session, ["length", "features_lens", "waveforms_lens"], 1);

    this.audioMetadata = session.inputMetadata.find((item) => item.name === this.audioSignalName) as TensorMetadata | undefined;
    this.lengthMetadata = session.inputMetadata.find((item) => item.name === this.lengthName) as TensorMetadata | undefined;
  }

  prepareInputsFromWaveform(samples: Float32Array): { signal: OrtTensor; length: OrtTensor } {
    const inputRank = this.audioMetadata?.shape?.length ?? 2;
    if (inputRank === 2) {
      const audioTensor = new ort.Tensor("float32", samples, [1, samples.length]);
      const lengthType = this.lengthMetadata?.type ?? "int64";
      const lengthTensor = intTensor(lengthType, [samples.length], [1]);
      return { signal: audioTensor, length: lengthTensor };
    }

    if (inputRank !== 3) {
      throw new Error(`Unsupported encoder input rank ${inputRank} for '${this.audioSignalName}'.`);
    }

    const nMelsFromShape = this.audioMetadata?.shape?.[1];
    const fe = this.options.config?.feature_extraction_params;
    const configNels = this.options.config?.features_size ?? fe?.n_mels;
    const nMels = typeof nMelsFromShape === "number" && Number.isFinite(nMelsFromShape)
      ? nMelsFromShape
      : configNels ?? 80;
    const isGigaam = this.options.config?.model_type === "gigaam";
    const sampleRate = this.options.sampleRate ?? fe?.sample_rate ?? 16000;
    const nFft = fe?.n_fft ?? 512;
    const winLength = fe?.window_size
      ? Math.round(Number(fe.window_size) * sampleRate)
      : 400;
    const hopLength = fe?.window_stride
      ? Math.round(Number(fe.window_stride) * sampleRate)
      : 160;
    const mel = logMelSpectrogram(samples, {
      sampleRate,
      nMels,
      nFft,
      winLength,
      hopLength,
      preemphasis: fe?.preemphasis_coefficient ?? (isGigaam ? 0 : 0.97),
      normalize: !isGigaam,
    });

    const audioTensor = new ort.Tensor("float32", mel.features, [1, mel.nMels, mel.frameCount]);
    const lengthType = this.lengthMetadata?.type ?? "int64";
    const lengthTensor = intTensor(lengthType, [mel.frameCount], [1]);
    return { signal: audioTensor, length: lengthTensor };
  }

  async run(processedSignalTensor: OrtTensor, processedLengthTensor: OrtTensor): Promise<EncodedAudio> {
    const outputs = await this.session.run({
      [this.audioSignalName]: processedSignalTensor,
      [this.lengthName]: processedLengthTensor,
    });

    const encoded = ensureOutput(this.session, outputs, 0);
    const encodedLength = ensureOutput(this.session, outputs, 1);

    if (encoded.dims.length === 3 && encoded.dims[0] === 1) {
      return {
        encodedData: encoded.data,
        encodedDims: encoded.dims,
        encodedLayout: "BCT",
        encodedLength: readScalarInt(encodedLength),
      };
    }

    throw new Error(
      `Unexpected encoder output shape: [${encoded.dims.join(", ")}]. Expected [1, C, T].`,
    );
  }
}

export class CtcAcousticModel extends EncoderModel {
  readonly vocabSize?: number;

  constructor(session: ort.InferenceSession, options: EncoderModelOptions & { vocabSize?: number } = {}) {
    super(session, options);
    this.vocabSize = options.vocabSize;
  }

  async run(processedSignalTensor: OrtTensor, processedLengthTensor: OrtTensor): Promise<EncodedAudio> {
    const outputs = await this.session.run({
      [this.audioSignalName]: processedSignalTensor,
      [this.lengthName]: processedLengthTensor,
    });

    const logits = ensureOutput(this.session, outputs, 0);
    const lengthTensor = this.session.outputNames.length > 1 ? outputs[this.session.outputNames[1]] : null;
    if (logits.dims.length !== 3) {
      throw new Error(
        `Unexpected CTC output shape: [${logits.dims.join(", ")}]. Expected rank-3 [1, T, V] or [1, V, T].`,
      );
    }

    let d1;
    let d2;
    if (logits.dims[0] === 1) {
      d1 = logits.dims[1];
      d2 = logits.dims[2];
    } else if (logits.dims[1] === 1) {
      d1 = logits.dims[0];
      d2 = logits.dims[2];
    } else {
      throw new Error(`Unexpected CTC batch layout: [${logits.dims.join(", ")}].`);
    }

    let layout: EncoderLayout = "BTV";
    if (this.vocabSize && d1 === this.vocabSize && d2 !== this.vocabSize) {
      layout = "BVT";
    } else if (this.vocabSize && d2 === this.vocabSize) {
      layout = "BTV";
    }

    return {
      encodedData: logits.data,
      encodedDims: [1, d1, d2],
      encodedLayout: layout,
      encodedLength: lengthTensor ? readScalarInt(lengthTensor) : (layout === "BVT" ? d2 : d1),
    };
  }
}

export class DecoderTransducerModel {
  readonly session: ort.InferenceSession;
  readonly decoderKind: "tdt" | "rnnt";
  readonly vocabSize?: number;
  readonly encoderOutputsName: string;
  readonly targetsName: string;
  readonly targetLengthName: string | null;
  readonly targetsType: string;
  readonly targetLengthType: string;
  readonly stateInputNames: string[];
  readonly stateInputMetadata: TensorMetadata[];
  readonly stateOutputNames: string[];

  constructor(session: ort.InferenceSession, options: DecoderTransducerOptions = {}) {
    this.session = session;
    this.decoderKind = options.decoderKind ?? "tdt";
    this.vocabSize = options.vocabSize;

    this.encoderOutputsName = firstExistingInputName(session, ["encoder_outputs"], 0);
    this.targetsName = firstExistingInputName(session, ["targets"], 1);
    this.targetLengthName = session.inputNames.includes("target_length")
      ? "target_length"
      : null;

    this.targetsType = (session.inputMetadata.find((item) => item.name === this.targetsName) as TensorMetadata | undefined)?.type ?? "int32";
    this.targetLengthType = this.targetLengthName
      ? (session.inputMetadata.find((item) => item.name === this.targetLengthName) as TensorMetadata | undefined)?.type ?? "int32"
      : "int32";

    this.stateInputNames = session.inputNames.filter(
      (name) => name !== this.encoderOutputsName && name !== this.targetsName && name !== this.targetLengthName,
    );

    this.stateInputMetadata = this.stateInputNames.map((name) => {
      const meta = session.inputMetadata.find((item) => item.name === name) as TensorMetadata | undefined;
      if (!meta) {
        throw new Error(`Missing input metadata for decoder state '${name}'.`);
      }
      return meta;
    });

    const stateOutputCandidates = session.outputNames.filter((name) => /state/i.test(name));
    this.stateOutputNames = stateOutputCandidates.length > 0
      ? stateOutputCandidates
      : session.outputNames.slice(1).filter((name) => !/length/i.test(name));
  }

  initialStates(): Map<string, OrtTensor> {
    const states = new Map();
    for (const meta of this.stateInputMetadata) {
      const shape = shapeFromMetadata(meta.shape);
      const length = product(shape);
      if (meta.type !== "float32") {
        throw new Error(`Unsupported decoder state type '${meta.type}' for '${meta.name}'.`);
      }
      states.set(meta.name, new ort.Tensor("float32", new Float32Array(length), shape));
    }
    return states;
  }

  resolveNextStates(outputs: TensorMap, currentStates: Map<string, OrtTensor>): Map<string, OrtTensor> {
    const nextStates = new Map();
    for (let i = 0; i < this.stateInputNames.length; i += 1) {
      const inputName = this.stateInputNames[i];
      const outputName = this.stateOutputNames[i];
      if (outputName && outputs[outputName]) {
        nextStates.set(inputName, outputs[outputName]);
      } else {
        nextStates.set(inputName, currentStates.get(inputName));
      }
    }
    return nextStates;
  }

  argmax(data: TensorData, start: number, end: number): number {
    let maxValue = -Infinity;
    let maxIndex = start;
    for (let i = start; i < end; i += 1) {
      const value = valueToNumber(data[i]);
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  rnntCandidates(mainOutputTensor: OrtTensor): TransducerCandidate[] {
    if (mainOutputTensor.type === "int32" || mainOutputTensor.type === "int64") {
      return numbersFromTensor(mainOutputTensor).map((token) => ({ token, duration: 0 }));
    }

    const flat = mainOutputTensor.data;
    return [{ token: this.argmax(flat, 0, flat.length), duration: 0 }];
  }

  tdtCandidates(mainOutputTensor: OrtTensor): TransducerCandidate[] {
    if (!this.vocabSize || this.vocabSize <= 0) {
      throw new Error("TDT decoder requires a positive vocabSize.");
    }

    const flat = mainOutputTensor.data;
    if (flat.length < this.vocabSize) {
      throw new Error(`TDT logits length ${flat.length} is smaller than vocabSize ${this.vocabSize}.`);
    }

    const token = this.argmax(flat, 0, this.vocabSize);
    let duration = 0;
    if (flat.length > this.vocabSize) {
      duration = this.argmax(flat, this.vocabSize, flat.length) - this.vocabSize;
    }

    return [{ token, duration }];
  }

  async predict(
    encoderFrameData: Float32Array,
    encoderFrameDims: TensorShape,
    token: number,
    states: Map<string, OrtTensor>,
  ): Promise<DecoderPrediction> {
    const feeds = {
      [this.encoderOutputsName]: new ort.Tensor("float32", encoderFrameData, encoderFrameDims),
      [this.targetsName]: intTensor(this.targetsType, [token], [1, 1]),
    };

    if (this.targetLengthName) {
      feeds[this.targetLengthName] = intTensor(this.targetLengthType, [1], [1]);
    }

    for (const stateName of this.stateInputNames) {
      const state = states.get(stateName);
      if (!state) {
        throw new Error(`Missing decoder state '${stateName}'.`);
      }
      feeds[stateName] = state;
    }

    const outputs = await this.session.run(feeds);
    const mainOutput = ensureOutput(this.session, outputs, 0);
    const nextStates = this.resolveNextStates(outputs, states);

    const candidates = this.decoderKind === "rnnt"
      ? this.rnntCandidates(mainOutput)
      : this.tdtCandidates(mainOutput);

    return { candidates, nextStates };
  }
}

export class TransducerGreedyDecoder {
  readonly model: TransducerDecoderLike;
  readonly maxSymbols: number;
  readonly blankTokenId: number;
  readonly defaultDuration: number;

  constructor(model: TransducerDecoderLike, options: TransducerGreedyDecoderOptions = {}) {
    this.model = model;
    this.maxSymbols = options.maxSymbols ?? 10;
    this.blankTokenId = options.blankTokenId ?? 0;
    this.defaultDuration = options.defaultDuration ?? 1;
  }

  frameAt(encodedData: TensorData, encodedDims: TensorShape, encodedLayout: EncoderLayout, t: number) {
    if (encodedLayout === "BCT") {
      const channels = encodedDims[1];
      const time = encodedDims[2];
      if (t >= time) {
        throw new Error(`Time index ${t} out of range for encoder output time=${time}.`);
      }

      const frame = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frame[c] = valueToNumber(encodedData[c * time + t]);
      }
      return { data: frame, dims: [1, channels, 1] };
    }

    throw new Error(`Unsupported encoder layout: ${encodedLayout}`);
  }

  async decode(
    encodedData: TensorData,
    encodedDims: TensorShape,
    encodedLayout: EncoderLayout,
    encodedLength: number,
  ): Promise<DecoderResult> {
    const tokenIds: number[] = [];
    const tokenFrames: TokenFrame[] = [];
    let currentToken = this.blankTokenId;
    let states = this.model.initialStates();
    let t = 0;

    const timeSteps = encodedLayout === "BCT" ? encodedDims[2] : 0;
    const limit = Math.min(encodedLength, timeSteps);

    while (t < limit) {
      const frame = this.frameAt(encodedData, encodedDims, encodedLayout, t);
      let nextT = t + this.defaultDuration;
      const emittedIndexes = [];

      for (let n = 0; n < this.maxSymbols; n += 1) {
        const { candidates, nextStates } = await this.model.predict(
          frame.data,
          frame.dims,
          currentToken,
          states,
        );

        let consumed = false;

        for (const candidate of candidates) {
          const token = valueToNumber(candidate.token);
          const duration = valueToNumber(candidate.duration);

          if (token !== this.blankTokenId) {
            const emissionIndex = tokenIds.length;
            tokenIds.push(token);
            tokenFrames.push({ startFrame: t, endFrame: t + this.defaultDuration });
            emittedIndexes.push(emissionIndex);
            currentToken = token;
            states = nextStates;
          }

          if (duration > 0) {
            nextT = t + duration;
            consumed = true;
            break;
          }
        }

        if (consumed) {
          break;
        }
      }

      for (const emissionIndex of emittedIndexes) {
        tokenFrames[emissionIndex].endFrame = nextT;
      }
      t = nextT;
    }

    return { tokenIds, tokenFrames, totalFrames: limit };
  }
}

export class CtcGreedyDecoder {
  readonly blankTokenId: number;

  constructor(options: CtcGreedyDecoderOptions = {}) {
    this.blankTokenId = options.blankTokenId ?? 0;
  }

  argmaxAt(data: TensorData, start: number, size: number): number {
    let best = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < size; i += 1) {
      const value = valueToNumber(data[start + i]);
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    return best;
  }

  async decode(
    encodedData: TensorData,
    encodedDims: TensorShape,
    encodedLayout: EncoderLayout,
    encodedLength: number,
  ): Promise<DecoderResult> {
    const tokenIds: number[] = [];
    const tokenFrames: TokenFrame[] = [];
    let previous = this.blankTokenId;

    const timeSteps = encodedLayout === "BVT" ? encodedDims[2] : encodedDims[1];
    const vocabSize = encodedLayout === "BVT" ? encodedDims[1] : encodedDims[2];
    const limit = Math.min(encodedLength, timeSteps);

    for (let t = 0; t < limit; t += 1) {
      let token;
      if (encodedLayout === "BVT") {
        let best = 0;
        let bestValue = -Infinity;
        for (let v = 0; v < vocabSize; v += 1) {
          const value = valueToNumber(encodedData[v * timeSteps + t]);
          if (value > bestValue) {
            bestValue = value;
            best = v;
          }
        }
        token = best;
      } else {
        token = this.argmaxAt(encodedData, t * vocabSize, vocabSize);
      }

      if (token === this.blankTokenId) {
        previous = this.blankTokenId;
        continue;
      }
      if (token === previous) {
        const last = tokenFrames[tokenFrames.length - 1];
        if (last) {
          last.endFrame = t + 1;
        }
        continue;
      }

      tokenIds.push(token);
      tokenFrames.push({ startFrame: t, endFrame: t + 1 });
      previous = token;
    }

    return { tokenIds, tokenFrames, totalFrames: limit };
  }
}
