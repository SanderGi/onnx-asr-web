import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText } from "./model-types.js";

export { configureOrtWeb };

function modelFilenameCandidates(filename, quantization = "int8") {
  if (!filename.endsWith(".onnx")) {
    return [filename];
  }
  if (quantization !== "int8") {
    return [filename];
  }
  const dotInt8 = filename.replace(/\.onnx$/, ".int8.onnx");
  const underscoreInt8 = filename.replace(/\.onnx$/, "_int8.onnx");
  return [...new Set([dotInt8, underscoreInt8, filename])];
}

function joinUrl(baseUrl, file) {
  const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const pageHref =
    typeof globalThis.location?.href === "string"
      ? globalThis.location.href
      : "http://localhost/";
  const resolvedBase = new URL(withSlash, pageHref);
  return new URL(file, resolvedBase).toString();
}

async function fetchText(url, fetchImpl, headers) {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function probeUrl(url, fetchImpl, headers) {
  const head = await fetchImpl(url, { method: "HEAD", headers });
  if (head.ok) {
    return true;
  }
  if (head.status === 404) {
    return false;
  }

  const get = await fetchImpl(url, { headers });
  if (get.ok) {
    return true;
  }
  if (get.status === 404) {
    return false;
  }
  throw new Error(`Failed to probe ${url}: ${get.status} ${get.statusText}`);
}

async function resolveModelUrl(baseUrl, filename, options) {
  const { fetchImpl, headers, quantization } = options;
  const candidates = modelFilenameCandidates(filename, quantization);

  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    const exists = await probeUrl(url, fetchImpl, headers);
    if (exists) {
      return url;
    }
  }

  return joinUrl(baseUrl, candidates[candidates.length - 1]);
}

async function fetchFirstExistingText(baseUrl, candidates, fetchImpl, headers) {
  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    const exists = await probeUrl(url, fetchImpl, headers);
    if (exists) {
      return fetchText(url, fetchImpl, headers);
    }
  }
  throw new Error(`Missing file. Checked: ${candidates.join(", ")}`);
}

async function resolveFirstExistingUrl(baseUrl, candidates, fetchImpl, headers) {
  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }
  throw new Error(`Missing file. Checked: ${candidates.join(", ")}`);
}

async function listHuggingFaceRepoFiles(repoId, revision, endpoint, fetchImpl, headers) {
  const base = endpoint.replace(/\/$/, "");
  const url = `${base}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=1`;
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to list Hugging Face repo files: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json();
  return new Set(payload.map((item) => item.path).filter((path) => typeof path === "string"));
}

function pickWhisperBeamsearchFile(files, quantization) {
  const list = [...files].filter((path) => /_beamsearch(?:\.int8)?\.onnx$/.test(path));
  if (list.length === 0) {
    return null;
  }
  if (quantization === "int8") {
    const int8 = list.find((name) => /(?:\.int8|_int8)\.onnx$/.test(name));
    if (int8) {
      return int8;
    }
  }
  return list.find((name) => !/(?:\.int8|_int8)\.onnx$/.test(name)) ?? list[0];
}

export async function loadLocalModel(baseUrl, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const quantization = options.quantization ?? "int8";
  const headers = options.headers;

  const configText = await fetchText(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "whisper-ort") {
    const modelCandidates = options.whisperModelCandidates ?? [
      "whisper-base_beamsearch.int8.onnx",
      "whisper-base_beamsearch.onnx",
    ];
    const whisperModel = await resolveFirstExistingUrl(baseUrl, modelCandidates, fetchImpl, headers);

    const vocabJson = await fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers);
    let addedTokensJson = "{}";
    if (await probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)) {
      addedTokensJson = await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers);
    }

    return createAsrModel({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel,
      vocabJson,
      addedTokensJson,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  if (spec.decoderKind === "whisper-hf") {
    const [encoderModel, decoderJointModel, vocabJson, hasAddedTokens] = await Promise.all([
      resolveModelUrl(baseUrl, spec.encoder, { fetchImpl, headers, quantization }),
      resolveModelUrl(baseUrl, spec.decoderJoint, { fetchImpl, headers, quantization }),
      fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers),
      probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers),
    ]);
    const addedTokensJson = hasAddedTokens
      ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
      : "{}";

    return createAsrModel({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      encoderModel,
      decoderJointModel,
      vocabJson,
      addedTokensJson,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  const [encoderModel, decoderJointModel, vocabularyText, preprocessorModel] = await Promise.all([
    resolveModelUrl(baseUrl, spec.encoder, { fetchImpl, headers, quantization }),
    spec.decoderJoint
      ? resolveModelUrl(baseUrl, spec.decoderJoint, { fetchImpl, headers, quantization })
      : Promise.resolve(null),
    fetchFirstExistingText(baseUrl, spec.vocabCandidates, fetchImpl, headers),
    spec.preprocessor
      ? resolveModelUrl(baseUrl, spec.preprocessor, { fetchImpl, headers, quantization })
      : Promise.resolve(null),
  ]);

  return createAsrModel({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel,
    encoderModel,
    decoderJointModel,
    vocabularyText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  });
}

export async function loadHuggingfaceModel(repoId, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? "https://huggingface.co").replace(/\/$/, "");
  const headers = options.hfToken
    ? { ...(options.headers ?? {}), Authorization: `Bearer ${options.hfToken}` }
    : options.headers;

  const baseUrl = `${endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/`;

  if (options.skipRepoListing) {
    return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
  }

  const configText = await fetchText(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "whisper-ort") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, endpoint, fetchImpl, headers);
    const whisperModelPath = pickWhisperBeamsearchFile(repoFiles, options.quantization ?? "int8");
    if (!whisperModelPath) {
      throw new Error("Could not find whisper-ort beamsearch ONNX in Hugging Face repo.");
    }

    return createAsrModel({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel: joinUrl(baseUrl, whisperModelPath),
      vocabJson: await fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers),
      addedTokensJson: repoFiles.has("added_tokens.json")
        ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
        : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
}
