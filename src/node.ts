import { access, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText, toneVocabularyTextFromConfig } from "./model-types.js";
import type {
  AsrTranscriber,
  CommonModelLoadOptions,
  DecoderOptions,
  SessionOptions,
  VadDetector,
  VadRuntimeOptions,
} from "./types.js";
import {
  createSileroVadModel,
  withVadModel,
} from "./vad.js";

type HeadersMap = Record<string, string>;
type VadOptions = VadRuntimeOptions;

export interface NodeModelLoadOptions extends CommonModelLoadOptions {
  sessionOptions?: SessionOptions;
  decoderOptions?: DecoderOptions;
  vadModel?: VadDetector | null;
  vad?: VadDetector | null;
  vadOptions?: VadOptions;
}

/** Additional Node-only options for downloading from Hugging Face. */
export interface NodeHuggingFaceOptions extends NodeModelLoadOptions {
  fetch?: typeof fetch;
  cacheDir?: string;
  revision?: string;
  endpoint?: string;
  hfToken?: string;
  forceDownload?: boolean;
}

export { configureOrtWeb };

function requireString(value: string | null | undefined, message: string): string {
  if (!value) {
    throw new Error(message);
  }
  return value;
}

function normalizeRepoId(repoId: string): string[] {
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

function resolveHfBaseUrl(repoId: string, revision = "main", endpoint = "https://huggingface.co"): string {
  const safeRevision = encodeURIComponent(revision);
  return `${endpoint.replace(/\/$/, "")}/${repoId}/resolve/${safeRevision}`;
}

function resolveHfApiBase(endpoint = "https://huggingface.co"): string {
  return endpoint.replace(/\/$/, "");
}

function modelFilenameCandidates(filename: string, quantization = "int8"): string[] {
  if (!filename.endsWith(".onnx")) {
    return [filename];
  }

  if (quantization !== "int8") {
    return [filename];
  }

  const dotInt8 = filename.replace(/\.onnx$/, ".int8.onnx");
  const underscoreInt8 = filename.replace(/\.onnx$/, "_int8.onnx");
  const unique = new Set([dotInt8, underscoreInt8, filename]);
  return [...unique];
}

function sidecarCandidates(onnxPath: string): string[] {
  if (!onnxPath.endsWith(".onnx")) {
    return [];
  }
  return [
    `${onnxPath}.data`,
    onnxPath.replace(/\.onnx$/, ".onnx_data"),
  ];
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchRequired(
  url: string,
  { fetchImpl, headers }: { fetchImpl: typeof fetch; headers?: HeadersMap },
): Promise<Response> {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  return response;
}

async function fetchOptional(
  url: string,
  { fetchImpl, headers }: { fetchImpl: typeof fetch; headers?: HeadersMap },
): Promise<Response | null> {
  const response = await fetchImpl(url, { headers });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  return response;
}

async function writeResponseToFile(response: Response, filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(filePath, bytes);
}

async function selectExistingFile(dir: string, candidates: readonly string[]): Promise<string | null> {
  for (const candidate of candidates) {
    if (await fileExists(join(dir, candidate))) {
      return candidate;
    }
  }
  return null;
}

async function selectOrFallbackModelFile(dir: string, filename: string, quantization: string): Promise<string> {
  const candidates = modelFilenameCandidates(filename, quantization);
  const existing = await selectExistingFile(dir, candidates);
  return existing ?? candidates[candidates.length - 1];
}

async function selectOrReadVocabulary(
  dir: string,
  candidates: readonly string[],
): Promise<{ filename: string; text: string }> {
  const existing = await selectExistingFile(dir, candidates);
  if (!existing) {
    throw new Error(`Missing vocabulary file. Checked: ${candidates.join(", ")}`);
  }

  const text = await readFile(join(dir, existing), "utf8");
  return { filename: existing, text };
}

function vocabJsonToText(vocabJsonText: string): string {
  const parsed = JSON.parse(vocabJsonText);
  const entries = Object.entries(parsed)
    .filter(([, id]) => Number.isInteger(id))
    .sort((a, b) => Number(a[1]) - Number(b[1]));
  if (entries.length === 0) {
    throw new Error("Invalid vocab.json mapping.");
  }
  return `${entries.map(([token, id]) => `${token} ${id}`).join("\n")}\n`;
}

async function walkFiles(root: string, prefix = ""): Promise<string[]> {
  const entries = await readdir(join(root, prefix), { withFileTypes: true });
  const out = [];

  for (const entry of entries) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      out.push(...(await walkFiles(root, rel)));
    } else if (entry.isFile()) {
      out.push(rel);
    }
  }

  return out;
}

function pickQuantizedByPattern(files: readonly string[], pattern: RegExp, quantization = "int8"): string | null {
  const matches = files.filter((file) => pattern.test(file));
  if (matches.length === 0) {
    return null;
  }
  if (quantization === "int8") {
    const int8 = matches.find((name) => /(?:\.int8|_int8)\.onnx$/.test(name));
    if (int8) {
      return int8;
    }
  }
  const plain = matches.find((name) => name.endsWith(".onnx") && !/(?:\.int8|_int8)\.onnx$/.test(name));
  return plain ?? matches[0];
}

async function listHuggingFaceRepoFiles(
  repoId: string,
  revision: string,
  endpoint: string | undefined,
  requestOptions: { fetchImpl: typeof fetch; headers?: HeadersMap },
): Promise<Set<string>> {
  const apiBase = resolveHfApiBase(endpoint);
  const url = `${apiBase}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=1`;
  const response = await fetchRequired(url, requestOptions);
  const payload = await response.json() as Array<{ path?: string }>;
  const files = payload
    .map((item: { path?: string }) => item.path)
    .filter((path: string | undefined): path is string => typeof path === "string");
  return new Set(files);
}

async function downloadPath(
  baseUrl: string,
  modelDir: string,
  relativePath: string,
  requestOptions: { fetchImpl: typeof fetch; headers?: HeadersMap },
  { required = true }: { required?: boolean } = {},
): Promise<boolean> {
  const url = `${baseUrl}/${relativePath}`;
  const localPath = join(modelDir, relativePath);
  const response = required
    ? await fetchRequired(url, requestOptions)
    : await fetchOptional(url, requestOptions);

  if (!response) {
    return false;
  }

  await writeResponseToFile(response, localPath);
  return true;
}

async function ensureSidecars(
  baseUrl: string,
  modelDir: string,
  modelPath: string,
  requestOptions: { fetchImpl: typeof fetch; headers?: HeadersMap },
  repoFiles: ReadonlySet<string> | null = null,
): Promise<void> {
  for (const sidecar of sidecarCandidates(modelPath)) {
    const local = join(modelDir, sidecar);
    if (await fileExists(local)) {
      continue;
    }

    if (repoFiles && !repoFiles.has(sidecar)) {
      continue;
    }

    await downloadPath(baseUrl, modelDir, sidecar, requestOptions, { required: false });
  }
}

async function resolveWhisperLocalArtifacts(
  modelDir: string,
  quantization: string,
): Promise<{ beamsearch: string; vocabPath: string; addedTokensPath: string | null }> {
  const files = await walkFiles(modelDir);
  const beamsearch = pickQuantizedByPattern(files, /_beamsearch(?:\.int8)?\.onnx$/, quantization);
  if (!beamsearch) {
    throw new Error("Could not find whisper-ort beamsearch ONNX file.");
  }

  const vocabPath = files.includes("vocab.json") ? "vocab.json" : null;
  if (!vocabPath) {
    throw new Error("Missing vocab.json for whisper model.");
  }

  const addedTokensPath = files.includes("added_tokens.json") ? "added_tokens.json" : null;
  return { beamsearch, vocabPath, addedTokensPath };
}

function pickGigaamVersion(
  config: { version?: unknown; [key: string]: unknown },
  files: readonly string[],
): string {
  if (config.version) {
    return String(config.version);
  }
  if (files.some((name) => name.startsWith("v3_"))) {
    return "v3";
  }
  if (files.some((name) => name.startsWith("v2_"))) {
    return "v2";
  }
  return "v2";
}

function selectGigaamArtifacts(
  config: { version?: unknown; [key: string]: unknown },
  files: readonly string[],
  quantization: string,
): (
  | { mode: "rnnt"; version: string; encoder: string; decoder: string; joint: string; vocab: string }
  | { mode: "ctc"; version: string; ctcModel: string; vocab: string }
) {
  const version = pickGigaamVersion(config, files);
  const preferPrefixes = [`${version}_rnnt`, `${version}_e2e_rnnt`];

  for (const prefix of preferPrefixes) {
    const encoder = pickQuantizedByPattern(files, new RegExp(`^${prefix}_encoder(?:\\.int8)?\\.onnx$`), quantization);
    const decoder = pickQuantizedByPattern(files, new RegExp(`^${prefix}_decoder(?:\\.int8)?\\.onnx$`), quantization);
    const joint = pickQuantizedByPattern(files, new RegExp(`^${prefix}_joint(?:\\.int8)?\\.onnx$`), quantization);
    if (encoder && decoder && joint) {
      return {
        mode: "rnnt",
        version,
        encoder,
        decoder,
        joint,
        vocab: `${version}_vocab.txt`,
      };
    }
  }

  const ctcCandidates = [`${version}_ctc.onnx`, `${version}_e2e_ctc.onnx`];
  for (const base of ctcCandidates) {
    const stem = base.replace(/\.onnx$/, "");
    const model = pickQuantizedByPattern(files, new RegExp(`^${stem}(?:\\.int8)?\\.onnx$`), quantization);
    if (model) {
      return {
        mode: "ctc",
        version,
        ctcModel: model,
        vocab: `${version}_vocab.txt`,
      };
    }
  }

  throw new Error("Could not resolve GigaAM artifacts (RNNT or CTC) from local files.");
}

function selectNemoConformerArtifacts(
  files: readonly string[],
  quantization: string,
): (
  | { mode: "rnnt"; encoder: string; decoderJoint: string }
  | { mode: "ctc"; ctcModel: string }
) {
  const encoder = pickQuantizedByPattern(files, /^encoder-model(?:\.int8)?\.onnx$/, quantization);
  const decoderJoint = pickQuantizedByPattern(files, /^decoder_joint-model(?:\.int8)?\.onnx$/, quantization);
  if (encoder && decoderJoint) {
    return { mode: "rnnt", encoder, decoderJoint };
  }

  const ctcModel = pickQuantizedByPattern(files, /^model(?:\.int8)?\.onnx$/, quantization);
  if (ctcModel) {
    return { mode: "ctc", ctcModel };
  }

  throw new Error("Could not resolve nemo-conformer artifacts (RNNT or CTC).");
}

function selectSherpaArtifacts(files: readonly string[], quantization: string) {
  const amDir = files.some((name) => name.startsWith("am-onnx/")) ? "am-onnx" : "am";
  const encoder = pickQuantizedByPattern(files, new RegExp(`^${amDir}/encoder(?:\\.int8)?\\.onnx$`), quantization);
  const decoder = pickQuantizedByPattern(files, new RegExp(`^${amDir}/decoder(?:\\.int8)?\\.onnx$`), quantization);
  const joiner = pickQuantizedByPattern(files, new RegExp(`^${amDir}/joiner(?:\\.int8)?\\.onnx$`), quantization);
  if (!encoder || !decoder || !joiner) {
    throw new Error("Could not resolve sherpa transducer files under am-onnx/ or am/.");
  }
  const tokens = files.includes("lang/tokens.txt")
    ? "lang/tokens.txt"
    : (files.includes("tokens.txt") ? "tokens.txt" : null);
  if (!tokens) {
    throw new Error("Could not find tokens file (lang/tokens.txt or tokens.txt).");
  }
  return { encoder, decoder, joiner, tokens };
}

function vadModelCandidates(quantization = "int8"): string[] {
  if (quantization && quantization !== "none" && quantization !== "float32" && quantization !== "fp32") {
    return [`onnx/model_${quantization}.onnx`, "onnx/model.onnx", "model.onnx"];
  }
  return ["onnx/model.onnx", "onnx/model_int8.onnx", "model.onnx"];
}

function selectSileroVadArtifact(files: readonly string[], quantization: string): string {
  for (const candidate of vadModelCandidates(quantization)) {
    if (files.includes(candidate)) {
      return candidate;
    }
  }

  const discovered = pickQuantizedByPattern(files, /^(?:onnx\/)?model(?:_[a-z0-9]+)?\.onnx$/i, quantization);
  if (discovered) {
    return discovered;
  }
  throw new Error("Could not resolve Silero VAD model file (expected onnx/model*.onnx).");
}

function resolveVadOption(options: NodeModelLoadOptions = {}): VadDetector | null {
  return options.vadModel ?? options.vad ?? null;
}

function attachVadIfProvided(
  asrModel: AsrTranscriber,
  options: NodeModelLoadOptions = {},
): AsrTranscriber {
  const vadModel = resolveVadOption(options);
  if (!vadModel) {
    return asrModel;
  }
  return withVadModel(asrModel, vadModel, options.vadOptions);
}

async function createAsrModelWithVad(
  params: Record<string, unknown>,
  options: NodeModelLoadOptions = {},
): Promise<AsrTranscriber> {
  const asrModel = await createAsrModel(params);
  return attachVadIfProvided(asrModel, options);
}

/** Load a VAD model from a local directory or direct ONNX file path. */
export async function loadLocalVadModel(modelDir: string, options: NodeHuggingFaceOptions = {}) {
  const quantization = options.quantization ?? "int8";
  const modelPath = modelDir.endsWith(".onnx")
    ? modelDir
    : join(modelDir, selectSileroVadArtifact(await walkFiles(modelDir), quantization));
  return createSileroVadModel({
    modelPath,
    sessionOptions: options.sessionOptions,
    options: options.vadOptions,
  });
}

/** Download a VAD model repo from Hugging Face into local cache. */
export async function downloadHuggingfaceVadModel(repoId: string, options: NodeHuggingFaceOptions = {}) {
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
  const headers: HeadersMap | undefined = hfToken ? { Authorization: `Bearer ${hfToken}` } : undefined;
  const requestOptions = { fetchImpl, headers, quantization };

  await mkdir(modelDir, { recursive: true });

  const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);
  const modelFile = selectSileroVadArtifact([...repoFiles], quantization);
  if (options.forceDownload || !(await fileExists(join(modelDir, modelFile)))) {
    await downloadPath(baseUrl, modelDir, modelFile, requestOptions, { required: true });
  }

  return modelDir;
}

/** Load a VAD model by Hugging Face repo id. */
export async function loadHuggingfaceVadModel(repoId: string, options: NodeHuggingFaceOptions = {}) {
  const modelDir = await downloadHuggingfaceVadModel(repoId, options);
  return loadLocalVadModel(modelDir, options);
}

/** Load an ASR model from a local directory. */
export async function loadLocalModel(modelDir: string, options: NodeModelLoadOptions = {}) {
  const quantization = options.quantization ?? "int8";
  const files = await walkFiles(modelDir);

  if (!files.includes("config.json")) {
    const sherpa = selectSherpaArtifacts(files, quantization);
    return createAsrModelWithVad({
      modelType: "sherpa-transducer",
      decoderKind: "sherpa-transducer",
      config: { sample_rate: options.sampleRate ?? 16000, max_tokens_per_step: 10 },
      encoderModel: join(modelDir, sherpa.encoder),
      decoderModel: join(modelDir, sherpa.decoder),
      decoderJointModel: join(modelDir, sherpa.joiner),
      vocabularyText: await readFile(join(modelDir, sherpa.tokens), "utf8"),
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  const configText = await readFile(join(modelDir, "config.json"), "utf8");
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "nemo-conformer") {
    const artifacts = selectNemoConformerArtifacts(files, quantization);
    const vocabulary = await selectOrReadVocabulary(modelDir, spec.vocabCandidates);

    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "rnnt",
        config,
        encoderModel: join(modelDir, artifacts.encoder),
        decoderJointModel: join(modelDir, artifacts.decoderJoint),
        vocabularyText: vocabulary.text,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      }, options);
    }

    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: join(modelDir, artifacts.ctcModel),
      vocabularyText: vocabulary.text,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "gigaam") {
    const artifacts = selectGigaamArtifacts(config, files, quantization);
    const vocabPath = files.includes(artifacts.vocab) ? artifacts.vocab : (files.includes("vocab.txt") ? "vocab.txt" : null);
    if (!vocabPath) {
      throw new Error("GigaAM vocabulary file not found.");
    }

    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "gigaam-rnnt",
        config,
        encoderModel: join(modelDir, artifacts.encoder),
        decoderModel: join(modelDir, artifacts.decoder),
        decoderJointModel: join(modelDir, artifacts.joint),
        vocabularyText: await readFile(join(modelDir, vocabPath), "utf8"),
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      }, options);
    }

    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: join(modelDir, artifacts.ctcModel),
      vocabularyText: await readFile(join(modelDir, vocabPath), "utf8"),
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "whisper-ort") {
    const artifacts = await resolveWhisperLocalArtifacts(modelDir, quantization);
    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel: join(modelDir, artifacts.beamsearch),
      vocabJson: await readFile(join(modelDir, artifacts.vocabPath), "utf8"),
      addedTokensJson: artifacts.addedTokensPath
        ? await readFile(join(modelDir, artifacts.addedTokensPath), "utf8")
        : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "whisper-hf") {
    const encoderFile = await selectOrFallbackModelFile(
      modelDir,
      requireString(spec.encoder, "whisper config is missing encoder model path."),
      quantization,
    );
    const decoderFile = await selectOrFallbackModelFile(
      modelDir,
      requireString(spec.decoderJoint, "whisper config is missing decoder model path."),
      quantization,
    );

    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      encoderModel: join(modelDir, encoderFile),
      decoderJointModel: join(modelDir, decoderFile),
      vocabJson: await readFile(join(modelDir, "vocab.json"), "utf8"),
      addedTokensJson: (await fileExists(join(modelDir, "added_tokens.json")))
        ? await readFile(join(modelDir, "added_tokens.json"), "utf8")
        : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  const encoderFile = await selectOrFallbackModelFile(
    modelDir,
    requireString(spec.encoder, `model type '${modelType}' is missing encoder path.`),
    quantization,
  );
  const decoderFile = spec.decoderJoint
    ? await selectOrFallbackModelFile(modelDir, spec.decoderJoint, quantization)
    : null;
  const preprocessorFile = spec.preprocessor
    ? await selectOrFallbackModelFile(modelDir, spec.preprocessor, quantization)
    : null;
  let vocabularyText = toneVocabularyTextFromConfig(config);
  if (!vocabularyText) {
    const vocabulary = await selectOrReadVocabulary(modelDir, spec.vocabCandidates);
    if (vocabulary.filename.endsWith(".json")) {
      vocabularyText = vocabJsonToText(vocabulary.text);
    } else {
      vocabularyText = vocabulary.text;
    }
  }

  return createAsrModelWithVad({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel: preprocessorFile ? join(modelDir, preprocessorFile) : null,
    encoderModel: join(modelDir, encoderFile),
    decoderJointModel: decoderFile ? join(modelDir, decoderFile) : null,
    vocabularyText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  }, options);
}

/** Download an ASR model repo from Hugging Face into local cache. */
export async function downloadHuggingfaceModel(repoId: string, options: NodeHuggingFaceOptions = {}) {
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
  const headers: HeadersMap | undefined = hfToken ? { Authorization: `Bearer ${hfToken}` } : undefined;
  const requestOptions = { fetchImpl, headers, quantization };

  await mkdir(modelDir, { recursive: true });

  const configPath = join(modelDir, "config.json");
  if (options.forceDownload || !(await fileExists(configPath))) {
    await downloadPath(baseUrl, modelDir, "config.json", requestOptions, { required: false });
  }

  if (!(await fileExists(configPath))) {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);
    const sherpa = selectSherpaArtifacts([...repoFiles], quantization);
    for (const file of [sherpa.encoder, sherpa.decoder, sherpa.joiner, sherpa.tokens]) {
      if (options.forceDownload || !(await fileExists(join(modelDir, file)))) {
        await downloadPath(baseUrl, modelDir, file, requestOptions, { required: true });
      }
      await ensureSidecars(baseUrl, modelDir, file, requestOptions, repoFiles);
    }
    return modelDir;
  }

  const config = parseConfigText(await readFile(configPath, "utf8"));
  const { spec } = detectModelType(config);

  if (spec.decoderKind === "nemo-conformer") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);
    const artifacts = selectNemoConformerArtifacts([...repoFiles], quantization);
    const downloadList = artifacts.mode === "rnnt"
      ? [artifacts.encoder, artifacts.decoderJoint]
      : [artifacts.ctcModel];

    for (const file of downloadList) {
      if (options.forceDownload || !(await fileExists(join(modelDir, file)))) {
        await downloadPath(baseUrl, modelDir, file, requestOptions, { required: true });
      }
      await ensureSidecars(baseUrl, modelDir, file, requestOptions, repoFiles);
    }

    const vocabulary = await selectExistingFile(modelDir, ["vocab.txt", "tokens.txt"]);
    if (!vocabulary || options.forceDownload) {
      const vocabFile = repoFiles.has("vocab.txt") ? "vocab.txt" : "tokens.txt";
      await downloadPath(baseUrl, modelDir, vocabFile, requestOptions, { required: true });
    }
    return modelDir;
  }

  if (spec.decoderKind === "whisper-ort") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);
    const modelPath = pickQuantizedByPattern([...repoFiles], /_beamsearch(?:\.int8)?\.onnx$/, quantization);
    if (!modelPath) {
      throw new Error("Could not find whisper-ort beamsearch model in Hugging Face repo.");
    }

    if (options.forceDownload || !(await fileExists(join(modelDir, modelPath)))) {
      await downloadPath(baseUrl, modelDir, modelPath, requestOptions, { required: true });
    }

    if (repoFiles.has("vocab.json") && (options.forceDownload || !(await fileExists(join(modelDir, "vocab.json"))))) {
      await downloadPath(baseUrl, modelDir, "vocab.json", requestOptions, { required: true });
    }
    if (repoFiles.has("added_tokens.json") && (options.forceDownload || !(await fileExists(join(modelDir, "added_tokens.json"))))) {
      await downloadPath(baseUrl, modelDir, "added_tokens.json", requestOptions, { required: true });
    }

    return modelDir;
  }

  if (spec.decoderKind === "gigaam") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);
    const artifacts = selectGigaamArtifacts(config, [...repoFiles], quantization);
    const downloadList = artifacts.mode === "rnnt"
      ? [artifacts.encoder, artifacts.decoder, artifacts.joint]
      : [artifacts.ctcModel];

    for (const file of downloadList) {
      if (options.forceDownload || !(await fileExists(join(modelDir, file)))) {
        await downloadPath(baseUrl, modelDir, file, requestOptions, { required: true });
      }
      await ensureSidecars(baseUrl, modelDir, file, requestOptions, repoFiles);
    }

    const vocab = repoFiles.has(artifacts.vocab) ? artifacts.vocab : "vocab.txt";
    if (options.forceDownload || !(await fileExists(join(modelDir, vocab)))) {
      await downloadPath(baseUrl, modelDir, vocab, requestOptions, { required: true });
    }
    return modelDir;
  }

  if (spec.decoderKind === "whisper-hf") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, options.endpoint, requestOptions);

    const modelFiles = [
      requireString(spec.encoder, "whisper config is missing encoder model path."),
      requireString(spec.decoderJoint, "whisper config is missing decoder model path."),
    ];
    for (const baseFile of modelFiles) {
      const candidates = modelFilenameCandidates(baseFile, quantization);
      const selected = candidates.find((path) => repoFiles.has(path)) ?? candidates[candidates.length - 1];

      if (options.forceDownload || !(await fileExists(join(modelDir, selected)))) {
        await downloadPath(baseUrl, modelDir, selected, requestOptions, { required: true });
      }
      await ensureSidecars(baseUrl, modelDir, selected, requestOptions, repoFiles);
    }

    for (const file of ["vocab.json", "added_tokens.json", "tokenizer.json"]) {
      if (repoFiles.has(file) && (options.forceDownload || !(await fileExists(join(modelDir, file))))) {
        await downloadPath(baseUrl, modelDir, file, requestOptions, { required: true });
      }
    }

    return modelDir;
  }

  const modelFiles = [
    requireString(spec.encoder, "model config is missing encoder model path."),
    ...(spec.decoderJoint ? [spec.decoderJoint] : []),
    ...(spec.preprocessor ? [spec.preprocessor] : []),
  ];

  for (const filename of modelFiles) {
    const candidates = modelFilenameCandidates(filename, quantization);
    const existing = options.forceDownload ? null : await selectExistingFile(modelDir, candidates);
    const selected = existing ?? candidates[candidates.length - 1];

    if (!existing || options.forceDownload) {
      let downloaded = false;
      for (const candidate of candidates) {
        const ok = await downloadPath(baseUrl, modelDir, candidate, requestOptions, { required: false });
        if (ok) {
          downloaded = true;
          await ensureSidecars(baseUrl, modelDir, candidate, requestOptions);
          break;
        }
      }
      if (!downloaded) {
        throw new Error(`Unable to download model file for ${filename}`);
      }
    } else {
      await ensureSidecars(baseUrl, modelDir, selected, requestOptions);
    }
  }

  const vocabularyName = await selectExistingFile(modelDir, spec.vocabCandidates);
  if (!vocabularyName || options.forceDownload) {
    let downloaded = false;
    for (const vocabCandidate of spec.vocabCandidates) {
      const ok = await downloadPath(baseUrl, modelDir, vocabCandidate, requestOptions, { required: false });
      if (ok) {
        downloaded = true;
        break;
      }
    }
    if (!downloaded && !vocabularyName) {
      throw new Error(`Unable to download vocabulary file. Tried: ${spec.vocabCandidates.join(", ")}`);
    }
  }

  return modelDir;
}

/** Load an ASR model by Hugging Face repo id. */
export async function loadHuggingfaceModel(repoId: string, options: NodeHuggingFaceOptions = {}) {
  const modelDir = await downloadHuggingfaceModel(repoId, options);
  return loadLocalModel(modelDir, options);
}
