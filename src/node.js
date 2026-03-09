import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText } from "./model-types.js";

export { configureOrtWeb };

function normalizeRepoId(repoId) {
  const parts = repoId
    .split("/")
    .map((part) => part.trim())
    .filter(Boolean);

  if (parts.length < 2) {
    throw new Error(`Invalid Hugging Face repo id: ${repoId}`);
  }
  if (parts.some((part) => part === "." || part === "..")) {
    throw new Error(`Unsafe Hugging Face repo id: ${repoId}`);
  }

  return parts;
}

function resolveHfBaseUrl(repoId, revision = "main", endpoint = "https://huggingface.co") {
  const safeRevision = encodeURIComponent(revision);
  return `${endpoint.replace(/\/$/, "")}/${repoId}/resolve/${safeRevision}`;
}

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

async function fileExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchRequired(url, { fetchImpl, headers }) {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  return response;
}

async function fetchOptional(url, { fetchImpl, headers }) {
  const response = await fetchImpl(url, { headers });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  return response;
}

async function writeResponseToFile(response, filePath) {
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(filePath, bytes);
}

async function selectExistingFile(dir, candidates) {
  for (const candidate of candidates) {
    if (await fileExists(join(dir, candidate))) {
      return candidate;
    }
  }
  return null;
}

async function selectOrFallbackModelFile(dir, filename, quantization) {
  const candidates = modelFilenameCandidates(filename, quantization);
  const existing = await selectExistingFile(dir, candidates);
  return existing ?? candidates[candidates.length - 1];
}

async function selectOrReadVocabulary(dir, candidates) {
  const existing = await selectExistingFile(dir, candidates);
  if (!existing) {
    throw new Error(`Missing vocabulary file. Checked: ${candidates.join(", ")}`);
  }

  const text = await readFile(join(dir, existing), "utf8");
  return { filename: existing, text };
}

async function downloadModelWithFallback(baseUrl, modelDir, filename, options) {
  const candidates = modelFilenameCandidates(filename, options.quantization);

  for (const candidate of candidates) {
    const modelUrl = `${baseUrl}/${candidate}`;
    const modelPath = join(modelDir, candidate);

    const response = await options.fetchImpl(modelUrl, { headers: options.headers });
    if (response.status === 404 && candidate !== candidates[candidates.length - 1]) {
      continue;
    }
    if (!response.ok) {
      throw new Error(`Failed to download ${modelUrl}: ${response.status} ${response.statusText}`);
    }

    await writeResponseToFile(response, modelPath);

    const sidecar = `${candidate}.data`;
    const sidecarResponse = await fetchOptional(`${baseUrl}/${sidecar}`, options);
    if (sidecarResponse) {
      await writeResponseToFile(sidecarResponse, join(modelDir, sidecar));
    }

    return candidate;
  }

  return candidates[candidates.length - 1];
}

export async function loadLocalModel(modelDir, options = {}) {
  const quantization = options.quantization ?? "int8";
  const configText = await readFile(join(modelDir, "config.json"), "utf8");
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  const encoderFile = await selectOrFallbackModelFile(modelDir, spec.encoder, quantization);
  const decoderFile = await selectOrFallbackModelFile(modelDir, spec.decoderJoint, quantization);
  const preprocessorFile = spec.preprocessor
    ? await selectOrFallbackModelFile(modelDir, spec.preprocessor, quantization)
    : null;
  const vocabulary = await selectOrReadVocabulary(modelDir, spec.vocabCandidates);

  return createAsrModel({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel: preprocessorFile ? join(modelDir, preprocessorFile) : null,
    encoderModel: join(modelDir, encoderFile),
    decoderJointModel: join(modelDir, decoderFile),
    vocabularyText: vocabulary.text,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  });
}

export async function downloadHuggingfaceModel(repoId, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const quantization = options.quantization ?? "int8";
  const repoParts = normalizeRepoId(repoId);
  const cacheDir = options.cacheDir ?? "models";
  const modelDir = join(cacheDir, ...repoParts);
  const revision = options.revision ?? "main";
  const baseUrl = resolveHfBaseUrl(repoId, revision, options.endpoint);

  const hfToken = options.hfToken ?? process.env.HF_TOKEN;
  const headers = hfToken ? { Authorization: `Bearer ${hfToken}` } : undefined;
  const requestOptions = { fetchImpl, headers, quantization };

  await mkdir(modelDir, { recursive: true });

  const configPath = join(modelDir, "config.json");
  if (options.forceDownload || !(await fileExists(configPath))) {
    const configResponse = await fetchRequired(`${baseUrl}/config.json`, requestOptions);
    await writeResponseToFile(configResponse, configPath);
  }

  const config = parseConfigText(await readFile(configPath, "utf8"));
  const { spec } = detectModelType(config);

  const modelFiles = [spec.encoder, spec.decoderJoint, ...(spec.preprocessor ? [spec.preprocessor] : [])];

  for (const filename of modelFiles) {
    const candidates = modelFilenameCandidates(filename, quantization);

    if (!options.forceDownload) {
      const existing = await selectExistingFile(modelDir, candidates);
      if (existing) {
        const sidecarName = `${existing}.data`;
        const sidecarPath = join(modelDir, sidecarName);
        if (!(await fileExists(sidecarPath))) {
          const sidecarResponse = await fetchOptional(`${baseUrl}/${sidecarName}`, requestOptions);
          if (sidecarResponse) {
            await writeResponseToFile(sidecarResponse, sidecarPath);
          }
        }
        continue;
      }
    }

    await downloadModelWithFallback(baseUrl, modelDir, filename, requestOptions);
  }

  const vocabularyName = await selectExistingFile(modelDir, spec.vocabCandidates);
  if (!vocabularyName || options.forceDownload) {
    let downloaded = false;
    for (const vocabCandidate of spec.vocabCandidates) {
      const response = await fetchOptional(`${baseUrl}/${vocabCandidate}`, requestOptions);
      if (!response) {
        continue;
      }
      await writeResponseToFile(response, join(modelDir, vocabCandidate));
      downloaded = true;
      break;
    }
    if (!downloaded && !vocabularyName) {
      throw new Error(`Unable to download vocabulary file. Tried: ${spec.vocabCandidates.join(", ")}`);
    }
  }

  return modelDir;
}

export async function loadHuggingfaceModel(repoId, options = {}) {
  const modelDir = await downloadHuggingfaceModel(repoId, options);
  return loadLocalModel(modelDir, options);
}
