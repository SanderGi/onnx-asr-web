export interface ModelSpec {
  decoderKind: string;
  preprocessor?: string | null;
  encoder?: string | null;
  decoderJoint?: string | null;
  vocabCandidates: string[];
  whisperModelPattern?: RegExp;
}

export interface ModelConfig {
  model_type?: string;
  version?: string;
  features_size?: number;
  num_mel_bins?: number;
  subsampling_factor?: number;
  max_tokens_per_step?: number;
  max_sequence_length?: number;
  sample_rate?: number;
  feature_extraction_params?: {
    sample_rate?: number;
    n_mels?: number;
    n_fft?: number;
    window_size?: number;
    window_stride?: number;
    preemphasis_coefficient?: number;
    [key: string]: unknown;
  };
  architectures?: string[];
  decoder_params?: {
    vocabulary?: string[];
    [key: string]: unknown;
  };
  pad_token_id?: number;
  [key: string]: unknown;
}

export const MODEL_TYPES: Record<string, ModelSpec> = {
  "nemo-conformer-tdt": {
    decoderKind: "tdt",
    preprocessor: "nemo128.onnx",
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["tokens.txt", "vocab.txt"],
  },
  "nemo-conformer-rnnt": {
    decoderKind: "rnnt",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"],
  },
  "nemo-conformer-ctc": {
    decoderKind: "ctc",
    preprocessor: null,
    encoder: "model.onnx",
    decoderJoint: null,
    vocabCandidates: ["vocab.txt", "tokens.txt"],
  },
  "nemo-conformer-aed": {
    decoderKind: "aed",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"],
  },
  "nemo-conformer": {
    decoderKind: "nemo-conformer",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"],
  },
  gigaam: {
    decoderKind: "gigaam",
    preprocessor: null,
    encoder: null,
    decoderJoint: null,
    vocabCandidates: ["v3_vocab.txt", "v2_vocab.txt", "vocab.txt", "tokens.txt"],
  },
  "whisper-ort": {
    decoderKind: "whisper-ort",
    preprocessor: null,
    encoder: null,
    decoderJoint: null,
    vocabCandidates: [],
    whisperModelPattern: /_beamsearch(?:\\.int8)?\\.onnx$/,
  },
  whisper: {
    decoderKind: "whisper-hf",
    preprocessor: null,
    encoder: "onnx/encoder_model.onnx",
    decoderJoint: "onnx/decoder_model_merged.onnx",
    vocabCandidates: [],
  },
  "tone-ctc": {
    decoderKind: "tone-ctc",
    preprocessor: null,
    encoder: "model.onnx",
    decoderJoint: null,
    vocabCandidates: ["vocab.txt", "tokens.txt", "vocab.json"],
  },
};

/** Parse a model `config.json` payload into an object. */
export function parseConfigText(configText: string): ModelConfig {
  let parsed;
  try {
    parsed = JSON.parse(configText);
  } catch (error) {
    throw new Error(`Invalid config.json: ${error}`);
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("Invalid config.json: expected a JSON object.");
  }

  return parsed as ModelConfig;
}

/** Detect model family/runtime spec from parsed config. */
export function detectModelType(config: ModelConfig): { modelType: string; spec: ModelSpec } {
  const modelType = config?.model_type;
  if (!modelType || typeof modelType !== "string") {
    const architectures = Array.isArray(config?.architectures) ? config.architectures : [];
    if (architectures.includes("ToneForCTC")) {
      return { modelType: "tone-ctc", spec: MODEL_TYPES["tone-ctc"] };
    }
    throw new Error("config.json is missing string field 'model_type'.");
  }

  const spec = MODEL_TYPES[modelType];
  if (!spec) {
    throw new Error(`Unsupported model_type: ${modelType}`);
  }

  return { modelType, spec };
}

/** Build Tone CTC vocabulary text from config when available. */
export function toneVocabularyTextFromConfig(config: ModelConfig): string | null {
  const vocab = config?.decoder_params?.vocabulary;
  if (!Array.isArray(vocab) || vocab.length === 0) {
    return null;
  }

  const lines = vocab.map((token, index) => {
    const normalized = token === " " ? "▁" : token;
    return `${normalized} ${index}`;
  });
  const pad = config?.pad_token_id;
  if (typeof pad === "number" && pad >= vocab.length) {
    lines.push(`<blank> ${pad}`);
  }
  return `${lines.join("\n")}\n`;
}
