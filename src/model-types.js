export const MODEL_TYPES = {
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
};

export function parseConfigText(configText) {
  let parsed;
  try {
    parsed = JSON.parse(configText);
  } catch (error) {
    throw new Error(`Invalid config.json: ${error}`);
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("Invalid config.json: expected a JSON object.");
  }

  return parsed;
}

export function detectModelType(config) {
  const modelType = config?.model_type;
  if (!modelType || typeof modelType !== "string") {
    throw new Error("config.json is missing string field 'model_type'.");
  }

  const spec = MODEL_TYPES[modelType];
  if (!spec) {
    throw new Error(`Unsupported model_type: ${modelType}`);
  }

  return { modelType, spec };
}
