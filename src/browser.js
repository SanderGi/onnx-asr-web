import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText } from "./model-types.js";

export { configureOrtWeb };

function modelFilenameCandidates(filename, quantization = "int8") {
  if (
    quantization === "int8" &&
    filename.endsWith(".onnx") &&
    !filename.endsWith(".int8.onnx")
  ) {
    return [filename.replace(/\.onnx$/, ".int8.onnx"), filename];
  }
  return [filename];
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
  throw new Error(`Missing vocabulary file. Checked: ${candidates.join(", ")}`);
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
  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? "https://huggingface.co").replace(/\/$/, "");
  const baseUrl = `${endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/`;
  const headers = options.hfToken
    ? { ...(options.headers ?? {}), Authorization: `Bearer ${options.hfToken}` }
    : options.headers;

  return loadLocalModel(baseUrl, {
    ...options,
    headers,
  });
}
