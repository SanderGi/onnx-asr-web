import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  createParakeetAsrModel,
  DEFAULT_MODEL_FILES,
  configureOrtWeb,
} from "./asr-model.js";

export { configureOrtWeb, DEFAULT_MODEL_FILES };

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

async function selectLocalModelFile(modelDir, filename, quantization) {
  const candidates = modelFilenameCandidates(filename, quantization);
  for (const candidate of candidates) {
    if (await fileExists(join(modelDir, candidate))) {
      return candidate;
    }
  }

  return candidates[candidates.length - 1];
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

    // External ONNX data file is optional but required for some exports.
    const sidecarName = `${candidate}.data`;
    const sidecarUrl = `${baseUrl}/${sidecarName}`;
    const sidecarResponse = await fetchOptional(sidecarUrl, options);
    if (sidecarResponse) {
      await writeResponseToFile(sidecarResponse, join(modelDir, sidecarName));
    }

    return candidate;
  }

  return candidates[candidates.length - 1];
}

export async function createParakeetFromLocalDir(modelDir, options = {}) {
  const files = { ...DEFAULT_MODEL_FILES, ...(options.files ?? {}) };
  const quantization = options.quantization ?? "int8";

  const [preprocessorFile, encoderFile, decoderJointFile] = await Promise.all([
    selectLocalModelFile(modelDir, files.preprocessor, quantization),
    selectLocalModelFile(modelDir, files.encoder, quantization),
    selectLocalModelFile(modelDir, files.decoderJoint, quantization),
  ]);

  const tokensText = await readFile(join(modelDir, files.tokens), "utf8");

  return createParakeetAsrModel({
    preprocessorModel: join(modelDir, preprocessorFile),
    encoderModel: join(modelDir, encoderFile),
    decoderJointModel: join(modelDir, decoderJointFile),
    tokensText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  });
}

export async function downloadParakeetFromHuggingFace(repoId, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const files = { ...DEFAULT_MODEL_FILES, ...(options.files ?? {}) };
  const repoParts = normalizeRepoId(repoId);
  const cacheDir = options.cacheDir ?? "models";
  const modelDir = join(cacheDir, ...repoParts);
  const revision = options.revision ?? "main";
  const quantization = options.quantization ?? "int8";
  const baseUrl = resolveHfBaseUrl(repoId, revision, options.endpoint);

  const hfToken = options.hfToken ?? process.env.HF_TOKEN;
  const headers = hfToken ? { Authorization: `Bearer ${hfToken}` } : undefined;
  const requestOptions = { fetchImpl, headers, quantization };

  await mkdir(modelDir, { recursive: true });

  const modelEntries = [files.preprocessor, files.encoder, files.decoderJoint];

  for (const filename of modelEntries) {
    const candidates = modelFilenameCandidates(filename, quantization);

    if (!options.forceDownload) {
      let hasAny = false;
      for (const candidate of candidates) {
        if (await fileExists(join(modelDir, candidate))) {
          hasAny = true;
          const sidecar = `${candidate}.data`;
          const sidecarPath = join(modelDir, sidecar);
          if (!(await fileExists(sidecarPath))) {
            const sidecarUrl = `${baseUrl}/${sidecar}`;
            const sidecarResponse = await fetchOptional(sidecarUrl, requestOptions);
            if (sidecarResponse) {
              await writeResponseToFile(sidecarResponse, sidecarPath);
            }
          }
          break;
        }
      }
      if (hasAny) {
        continue;
      }
    }

    await downloadModelWithFallback(baseUrl, modelDir, filename, requestOptions);
  }

  const tokensPath = join(modelDir, files.tokens);
  if (options.forceDownload || !(await fileExists(tokensPath))) {
    const tokensUrl = `${baseUrl}/${files.tokens}`;
    const tokensResponse = await fetchRequired(tokensUrl, requestOptions);
    await writeResponseToFile(tokensResponse, tokensPath);
  }

  return modelDir;
}

export async function createParakeetFromHuggingFace(repoId, options = {}) {
  const modelDir = await downloadParakeetFromHuggingFace(repoId, options);
  return createParakeetFromLocalDir(modelDir, options);
}
