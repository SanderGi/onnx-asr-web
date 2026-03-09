import * as ort from "onnxruntime-web";
import { firstExistingInputName, int64TensorValues, readScalarInt } from "./utils.js";

function int64Tensor(values, dims) {
  return new ort.Tensor("int64", int64TensorValues(values), dims);
}

function int32Tensor(values, dims) {
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}

function ensureOutput(session, outputMap, index) {
  const name = session.outputNames[index];
  const tensor = outputMap[name];
  if (!tensor) {
    throw new Error(`Missing output tensor at index ${index} (${name}).`);
  }
  return tensor;
}

export class PreprocessorModel {
  constructor(session) {
    this.session = session;
    this.inputSignalName = firstExistingInputName(session, ["input_signal", "waveforms"], 0);
    this.lengthName = firstExistingInputName(session, ["length", "waveforms_lens"], 1);
  }

  async run(audioSamples) {
    const audioTensor = new ort.Tensor("float32", audioSamples, [1, audioSamples.length]);
    const lengthTensor = int64Tensor([audioSamples.length], [1]);

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
  constructor(session) {
    this.session = session;
    this.audioSignalName = firstExistingInputName(session, ["audio_signal"], 0);
    this.lengthName = firstExistingInputName(session, ["length"], 1);
  }

  async run(processedSignalTensor, processedLengthTensor) {
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

export class DecoderTdtModel {
  constructor(session) {
    this.session = session;
    this.encoderOutputsName = firstExistingInputName(session, ["encoder_outputs"], 0);
    this.targetsName = firstExistingInputName(session, ["targets"], 1);
    this.targetLengthName = firstExistingInputName(session, ["target_length"], 2);
    this.state1InName = firstExistingInputName(session, ["input_states_1"], 3);
    this.state2InName = firstExistingInputName(session, ["input_states_2"], 4);

    this.state1OutName = session.outputNames.includes("output_states_1")
      ? "output_states_1"
      : session.outputNames[2];
    this.state2OutName = session.outputNames.includes("output_states_2")
      ? "output_states_2"
      : session.outputNames[3];
  }

  async predict(encoderFrameData, encoderFrameDims, token, state1, state2) {
    const outputs = await this.session.run({
      [this.encoderOutputsName]: new ort.Tensor("float32", encoderFrameData, encoderFrameDims),
      [this.targetsName]: int32Tensor([token], [1, 1]),
      [this.targetLengthName]: int32Tensor([1], [1]),
      [this.state1InName]: state1,
      [this.state2InName]: state2,
    });

    const logits = ensureOutput(this.session, outputs, 0);
    const nextState1 = outputs[this.state1OutName];
    const nextState2 = outputs[this.state2OutName];
    if (!nextState1 || !nextState2) {
      throw new Error("Decoder did not return recurrent state outputs.");
    }

    return { logits, nextState1, nextState2 };
  }
}

export class TdtGreedyDecoder {
  constructor(model, options = {}) {
    this.model = model;
    this.maxSymbols = options.maxSymbols ?? 10;
    this.blankTokenId = options.blankTokenId ?? 8192;
    this.vocabSize = options.vocabSize ?? this.blankTokenId + 1;
    this.defaultDuration = options.defaultDuration ?? 1;
  }

  frameAt(encodedData, encodedDims, encodedLayout, t) {
    if (encodedLayout === "BCT") {
      const channels = encodedDims[1];
      const time = encodedDims[2];
      if (t >= time) {
        throw new Error(`Time index ${t} out of range for encoder output time=${time}.`);
      }

      const frame = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frame[c] = encodedData[c * time + t];
      }
      return { data: frame, dims: [1, channels, 1] };
    }

    throw new Error(`Unsupported encoder layout: ${encodedLayout}`);
  }

  argmax(data, start, end) {
    let maxValue = -Infinity;
    let maxIndex = start;
    for (let i = start; i < end; i += 1) {
      const value = data[i];
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  async decode(encodedData, encodedDims, encodedLayout, encodedLength) {
    const tokenIds = [];
    const tokenFrames = [];
    let currentToken = this.blankTokenId;
    let state1 = new ort.Tensor("float32", new Float32Array(2 * 1 * 640), [2, 1, 640]);
    let state2 = new ort.Tensor("float32", new Float32Array(2 * 1 * 640), [2, 1, 640]);
    let t = 0;

    const timeSteps = encodedLayout === "BCT" ? encodedDims[2] : 0;
    const limit = Math.min(encodedLength, timeSteps);

    while (t < limit) {
      const frame = this.frameAt(encodedData, encodedDims, encodedLayout, t);
      let nextT = t + this.defaultDuration;
      const emittedIndexes = [];

      for (let n = 0; n < this.maxSymbols; n += 1) {
        const { logits, nextState1, nextState2 } = await this.model.predict(
          frame.data,
          frame.dims,
          currentToken,
          state1,
          state2,
        );

        const flat = logits.data;
        const token = this.argmax(flat, 0, this.vocabSize);
        const duration = this.argmax(flat, this.vocabSize, flat.length) - this.vocabSize;

        if (token !== this.blankTokenId) {
          const emissionIndex = tokenIds.length;
          tokenIds.push(token);
          tokenFrames.push({ startFrame: t, endFrame: t + this.defaultDuration });
          emittedIndexes.push(emissionIndex);
          currentToken = token;
          state1 = nextState1;
          state2 = nextState2;
        }

        if (duration > 0) {
          nextT = t + duration;
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
