import * as ort from "onnxruntime-web";
import { decodeWav, resampleLinear } from "./audio.js";

function firstExistingName(names, candidates, fallbackIndex = 0) {
  for (const candidate of candidates) {
    if (names.includes(candidate)) {
      return candidate;
    }
  }
  return names[fallbackIndex];
}

function int64Scalar(value) {
  return new ort.Tensor("int64", new BigInt64Array([BigInt(value)]), []);
}

function joinText(parts) {
  return parts
    .filter((text) => typeof text === "string" && text.trim().length > 0)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function toFloat32(samples) {
  if (samples instanceof Float32Array) {
    return samples;
  }
  return Float32Array.from(samples);
}

export class SileroVadModel {
  constructor(session, options = {}) {
    this.session = session;
    this.sampleRate = Number(options.sampleRate) || 16000;
    this.threshold = options.threshold ?? 0.5;
    this.negThreshold = options.negThreshold ?? Math.max(0.05, this.threshold - 0.15);
    this.minSpeechMs = options.minSpeechMs ?? 250;
    this.minSilenceMs = options.minSilenceMs ?? 700;
    this.speechPadMs = options.speechPadMs ?? 200;
    this.windowSamples = Number(options.windowSamples)
      || (this.sampleRate === 8000 ? 256 : 512);

    this.inputName = firstExistingName(session.inputNames, ["input", "x"], 0);
    this.stateInputName = firstExistingName(session.inputNames, ["state"], 1);
    this.srInputName = firstExistingName(session.inputNames, ["sr", "sample_rate"], 2);
    this.outputName = firstExistingName(session.outputNames, ["output", "prob", "speech_prob"], 0);
    this.stateOutputName = firstExistingName(session.outputNames, ["stateN", "state", "new_state"], 1);
  }

  initialState() {
    return new ort.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
  }

  async speechProbabilities(samples, sampleRate = this.sampleRate) {
    let prepared = toFloat32(samples);
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }

    const probs = [];
    let state = this.initialState();
    let offset = 0;
    while (offset < prepared.length) {
      const end = Math.min(offset + this.windowSamples, prepared.length);
      const chunk = new Float32Array(this.windowSamples);
      chunk.set(prepared.subarray(offset, end), 0);

      const outputs = await this.session.run({
        [this.inputName]: new ort.Tensor("float32", chunk, [1, this.windowSamples]),
        [this.stateInputName]: state,
        [this.srInputName]: int64Scalar(this.sampleRate),
      });

      const prob = outputs[this.outputName];
      const nextState = outputs[this.stateOutputName];
      if (!prob || !nextState) {
        throw new Error("Silero VAD outputs are missing probability or next state.");
      }

      probs.push(Number(prob.data[0] ?? 0));
      state = nextState;
      offset = end;
    }

    return { probs, processedSamples: prepared.length };
  }

  /**
   * Returns speech segments in input sample-rate coordinates.
   * Segment bounds are [start, end), in samples.
   */
  async detectSpeechSegments(samples, sampleRate = this.sampleRate, overrides = {}) {
    const threshold = overrides.threshold ?? this.threshold;
    const negThreshold = overrides.negThreshold ?? this.negThreshold;
    const minSpeechSamples = Math.round(((overrides.minSpeechMs ?? this.minSpeechMs) / 1000) * this.sampleRate);
    const minSilenceSamples = Math.round(((overrides.minSilenceMs ?? this.minSilenceMs) / 1000) * this.sampleRate);
    const speechPadSamples = Math.round(((overrides.speechPadMs ?? this.speechPadMs) / 1000) * this.sampleRate);

    const prepared = toFloat32(samples);
    const ratioToVad = sampleRate / this.sampleRate;
    const { probs, processedSamples } = await this.speechProbabilities(prepared, sampleRate);
    const maxVadSamples = sampleRate === this.sampleRate
      ? prepared.length
      : Math.round(processedSamples * ratioToVad);

    const raw = [];
    let activeStart = -1;
    let pendingEnd = -1;

    for (let i = 0; i < probs.length; i += 1) {
      const frameStart = i * this.windowSamples;
      const p = probs[i];

      if (p >= threshold) {
        if (activeStart < 0) {
          activeStart = frameStart;
        }
        pendingEnd = -1;
        continue;
      }

      if (activeStart < 0) {
        continue;
      }

      if (p <= negThreshold && pendingEnd < 0) {
        pendingEnd = frameStart;
      }
      if (pendingEnd >= 0 && frameStart - pendingEnd >= minSilenceSamples) {
        if (pendingEnd - activeStart >= minSpeechSamples) {
          raw.push({ start: activeStart, end: pendingEnd });
        }
        activeStart = -1;
        pendingEnd = -1;
      }
    }

    if (activeStart >= 0) {
      const end = probs.length * this.windowSamples;
      if (end - activeStart >= minSpeechSamples) {
        raw.push({ start: activeStart, end });
      }
    }

    if (raw.length === 0) {
      return [];
    }

    const padded = raw.map((segment) => ({
      start: Math.max(0, segment.start - speechPadSamples),
      end: segment.end + speechPadSamples,
    }));

    const merged = [];
    for (const segment of padded) {
      const last = merged[merged.length - 1];
      if (!last || segment.start > last.end + minSilenceSamples) {
        merged.push({ ...segment });
      } else {
        last.end = Math.max(last.end, segment.end);
      }
    }

    return merged.map((segment) => {
      const startSec = segment.start / this.sampleRate;
      const endSec = segment.end / this.sampleRate;
      const start = Math.max(0, Math.round(startSec * sampleRate));
      const end = Math.min(maxVadSamples, Math.round(endSec * sampleRate));
      return {
        start,
        end: Math.max(start + 1, end),
        startSec: start / sampleRate,
        endSec: Math.max(start + 1, end) / sampleRate,
      };
    });
  }
}

export class VadChunkedAsrModel {
  constructor(baseModel, vadModel, options = {}) {
    this.baseModel = baseModel;
    this.vadModel = vadModel;
    this.options = options;
    this.sampleRate = baseModel.sampleRate ?? vadModel.sampleRate ?? 16000;
  }

  async transcribeSamples(samples, sampleRate = this.sampleRate, options = {}) {
    const prepared = toFloat32(samples);
    const vadOptions = options.vadOptions ?? this.options;
    const segments = await this.vadModel.detectSpeechSegments(prepared, sampleRate, vadOptions);

    if (segments.length === 0) {
      return {
        tokenIds: [],
        tokenFrames: [],
        words: [],
        text: "",
        segments: [],
      };
    }

    const tokenIds = [];
    const words = [];
    const texts = [];
    for (const segment of segments) {
      const chunk = prepared.subarray(segment.start, segment.end);
      const chunkResult = await this.baseModel.transcribeSamples(chunk, sampleRate, options);
      tokenIds.push(...(chunkResult.tokenIds ?? []));
      if (chunkResult.text) {
        texts.push(chunkResult.text);
      }
      for (const word of chunkResult.words ?? []) {
        words.push({
          word: word.word,
          start: Number((word.start + segment.startSec).toFixed(3)),
          end: Number((word.end + segment.startSec).toFixed(3)),
        });
      }
    }

    return {
      tokenIds,
      tokenFrames: [],
      words,
      text: joinText(texts),
      segments,
    };
  }

  async transcribeWavBuffer(arrayBuffer, options = {}) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
}

export function withVadModel(asrModel, vadModel, options = {}) {
  if (!vadModel || typeof vadModel.detectSpeechSegments !== "function") {
    throw new Error("Invalid VAD model: expected detectSpeechSegments(samples, sampleRate, options).");
  }
  return new VadChunkedAsrModel(asrModel, vadModel, options);
}

export async function createSileroVadModel({
  modelPath,
  sessionOptions,
  options = {},
} = {}) {
  if (!modelPath) {
    throw new Error("createSileroVadModel expects modelPath.");
  }
  const session = await ort.InferenceSession.create(modelPath, sessionOptions);
  return new SileroVadModel(session, options);
}
