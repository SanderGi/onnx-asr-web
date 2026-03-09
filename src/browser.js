import {
  createParakeetAsrModel,
  DEFAULT_MODEL_FILES,
  configureOrtWeb,
} from "./asr-model.js";

export { configureOrtWeb, DEFAULT_MODEL_FILES };

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

async function fetchText(url, fetchImpl, headers) {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${url}: ${response.status} ${response.statusText}`
    );
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

function joinUrl(baseUrl, file) {
  const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const pageHref =
    typeof globalThis.location?.href === "string"
      ? globalThis.location.href
      : "http://localhost/";
  const resolvedBase = new URL(withSlash, pageHref);
  return new URL(file, resolvedBase).toString();
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

export async function createParakeetFromBaseUrl(baseUrl, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const files = { ...DEFAULT_MODEL_FILES, ...(options.files ?? {}) };
  const quantization = options.quantization ?? "int8";
  const headers = options.headers;

  const [preprocessorModel, encoderModel, decoderJointModel, tokensText] =
    await Promise.all([
      resolveModelUrl(baseUrl, files.preprocessor, {
        fetchImpl,
        headers,
        quantization,
      }),
      resolveModelUrl(baseUrl, files.encoder, {
        fetchImpl,
        headers,
        quantization,
      }),
      resolveModelUrl(baseUrl, files.decoderJoint, {
        fetchImpl,
        headers,
        quantization,
      }),
      fetchText(joinUrl(baseUrl, files.tokens), fetchImpl, headers),
    ]);

  return createParakeetAsrModel({
    preprocessorModel,
    encoderModel,
    decoderJointModel,
    tokensText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  });
}

export async function createParakeetFromHuggingFace(repoId, options = {}) {
  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? "https://huggingface.co").replace(/\/$/, "");
  const baseUrl = `${endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/`;
  const headers = options.hfToken
    ? { ...(options.headers ?? {}), Authorization: `Bearer ${options.hfToken}` }
    : options.headers;

  return createParakeetFromBaseUrl(baseUrl, {
    ...options,
    headers,
  });
}
