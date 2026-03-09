const RIFF = 0x52494646;
const WAVE = 0x57415645;
const FMT = 0x666d7420;
const DATA = 0x64617461;

function readChunkId(view, offset) {
  return view.getUint32(offset, false);
}

function parseFmtChunk(view, offset) {
  const audioFormat = view.getUint16(offset + 8, true);
  const channels = view.getUint16(offset + 10, true);
  const sampleRate = view.getUint32(offset + 12, true);
  const bitsPerSample = view.getUint16(offset + 22, true);

  return { audioFormat, channels, sampleRate, bitsPerSample };
}

function decodePcm(view, offset, byteLength, fmt) {
  const { audioFormat, channels, bitsPerSample } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = byteLength / bytesPerSample;
  const frameCount = totalSamples / channels;
  const mono = new Float32Array(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    let acc = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleOffset =
        offset + (frame * channels + channel) * bytesPerSample;
      let value;
      if (audioFormat === 1 && bitsPerSample === 16) {
        value = view.getInt16(sampleOffset, true) / 32768;
      } else if (audioFormat === 1 && bitsPerSample === 32) {
        value = view.getInt32(sampleOffset, true) / 2147483648;
      } else if (audioFormat === 3 && bitsPerSample === 32) {
        value = view.getFloat32(sampleOffset, true);
      } else {
        throw new Error(
          `Unsupported WAV format: audioFormat=${audioFormat}, bitsPerSample=${bitsPerSample}`
        );
      }
      acc += value;
    }
    mono[frame] = acc / channels;
  }

  return mono;
}

export function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (readChunkId(view, 0) !== RIFF || readChunkId(view, 8) !== WAVE) {
    throw new Error("Invalid WAV file header.");
  }

  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readChunkId(view, offset);
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === FMT) {
      fmt = parseFmtChunk(view, offset);
    } else if (chunkId === DATA) {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize + (chunkSize % 2);
  }

  if (!fmt) {
    throw new Error("WAV file is missing fmt chunk.");
  }
  if (dataOffset < 0) {
    throw new Error("WAV file is missing data chunk.");
  }

  const samples = decodePcm(view, dataOffset, dataSize, fmt);
  return {
    sampleRate: fmt.sampleRate,
    samples,
  };
}

export function resampleLinear(samples, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return samples;
  }

  const ratio = inputRate / outputRate;
  const outLength = Math.max(1, Math.floor(samples.length / ratio));
  const out = new Float32Array(outLength);

  for (let i = 0; i < outLength; i += 1) {
    const srcPos = i * ratio;
    const left = Math.floor(srcPos);
    const right = Math.min(left + 1, samples.length - 1);
    const alpha = srcPos - left;
    out[i] = samples[left] * (1 - alpha) + samples[right] * alpha;
  }

  return out;
}

export function normalize(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i];
  }
  const mean = sum / samples.length;

  let varianceAcc = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const centered = samples[i] - mean;
    varianceAcc += centered * centered;
  }

  const std = Math.sqrt(varianceAcc / samples.length);
  const denom = std + 1e-5;
  const normalized = new Float32Array(samples.length);

  for (let i = 0; i < samples.length; i += 1) {
    normalized[i] = (samples[i] - mean) / denom;
  }

  return normalized;
}
