import type * as ort from "onnxruntime-web";

export type AudioSamples = Float32Array | number[] | readonly number[];

/** Frame-aligned token timing emitted by a decoder. */
export interface TokenFrame {
  startFrame: number;
  endFrame: number;
}

/** Word-level timestamp in seconds. */
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

/** Speech segment timestamp in both samples and seconds. */
export interface SegmentTimestamp {
  start: number;
  end: number;
  startSec: number;
  endSec: number;
}

/** Normalized transcription result returned by ASR models. */
export interface TranscriptResult {
  text: string;
  tokenIds: number[];
  tokenFrames: TokenFrame[];
  words: WordTimestamp[];
  segments?: SegmentTimestamp[];
}

/** Raw VAD probability trace over processed windows. */
export interface VadSpeechProbabilities {
  probs: number[];
  processedSamples: number;
}

/** Runtime tuning knobs for VAD chunking. */
export interface VadRuntimeOptions {
  sampleRate?: number;
  threshold?: number;
  negThreshold?: number;
  minSpeechMs?: number;
  minSilenceMs?: number;
  speechPadMs?: number;
  windowSamples?: number;
}

/** Common ASR transcription interface shared by Node and browser loaders. */
export interface AsrTranscriber {
  sampleRate: number;
  transcribeSamples(
    samples: AudioSamples,
    sampleRate?: number,
    options?: Record<string, unknown>,
  ): Promise<TranscriptResult>;
  transcribeWavBuffer(
    arrayBuffer: ArrayBuffer,
    options?: Record<string, unknown>,
  ): Promise<TranscriptResult>;
}

/** Voice activity detector contract used by `withVadModel()`. */
export interface VadDetector {
  sampleRate: number;
  detectSpeechSegments(
    samples: AudioSamples,
    sampleRate?: number,
    overrides?: VadRuntimeOptions,
  ): Promise<SegmentTimestamp[]>;
  speechProbabilities?(
    samples: AudioSamples,
    sampleRate?: number,
  ): Promise<VadSpeechProbabilities>;
}

/** Global ONNX Runtime Web environment options. */
export interface ConfigureOrtWebOptions {
  numThreads?: number;
  wasmPaths?: string | Record<string, string>;
  simd?: boolean;
  proxy?: boolean;
}

/** Generic decoder tuning supported by transducer loaders. */
export interface DecoderOptions {
  maxSymbols?: number;
}

export type SessionOptions = ort.InferenceSession.SessionOptions;
export type OrtTensor = ort.Tensor;
export type TensorData = ort.Tensor["data"];
export type TensorMap = ort.InferenceSession.OnnxValueMapType;

/** Shared load options accepted by Node and browser model loaders. */
export interface CommonModelLoadOptions {
  quantization?: "int8" | "none" | string;
  sessionOptions?: SessionOptions;
  decoderOptions?: DecoderOptions;
  sampleRate?: number;
  vadModel?: VadDetector | null;
  vad?: VadDetector | null;
  vadOptions?: VadRuntimeOptions;
}
