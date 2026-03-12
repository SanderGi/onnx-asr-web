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

/** Browser loader options for local URLs and Hugging Face model ids. */
export interface BrowserModelLoadOptions extends CommonModelLoadOptions {
  sessionOptions?: SessionOptions;
  decoderOptions?: DecoderOptions;
  vadModel?: VadDetector | null;
  vad?: VadDetector | null;
  vadOptions?: VadOptions;
  headers?: HeadersMap;
  fetch?: typeof fetch;
  whisperModelCandidates?: string[];
  skipRepoListing?: boolean;
  hfToken?: string;
  revision?: string;
  endpoint?: string;
}

export { configureOrtWeb };

function requireString(value: string | null | undefined, message: string): string {
  if (!value) {
    throw new Error(message);
  }
  return value;
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
  return [...new Set([dotInt8, underscoreInt8, filename])];
}

function joinUrl(baseUrl: string, file: string): string {
  const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const pageHref =
    typeof globalThis.location?.href === "string"
      ? globalThis.location.href
      : "http://localhost/";
  const resolvedBase = new URL(withSlash, pageHref);
  return new URL(file, resolvedBase).toString();
}

async function fetchText(url: string, fetchImpl: typeof fetch, headers?: HeadersMap): Promise<string> {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchTextOptional(url: string, fetchImpl: typeof fetch, headers?: HeadersMap): Promise<string | null> {
  const response = await fetchImpl(url, { headers });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function probeUrl(url: string, fetchImpl: typeof fetch, headers?: HeadersMap): Promise<boolean> {
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

async function resolveModelUrl(
  baseUrl: string,
  filename: string,
  options: { fetchImpl: typeof fetch; headers?: HeadersMap; quantization: string },
): Promise<string> {
  const { fetchImpl, headers, quantization } = options;
  const candidates = modelFilenameCandidates(filename, quantization);

  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }

  return joinUrl(baseUrl, candidates[candidates.length - 1]);
}

async function resolveFirstExistingUrl(
  baseUrl: string,
  candidates: readonly string[],
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<string> {
  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }
  throw new Error(`Missing file. Checked: ${candidates.join(", ")}`);
}

async function fetchFirstExistingText(
  baseUrl: string,
  candidates: readonly string[],
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<string> {
  const url = await resolveFirstExistingUrl(baseUrl, candidates, fetchImpl, headers);
  return fetchText(url, fetchImpl, headers);
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

async function listHuggingFaceRepoFiles(
  repoId: string,
  revision: string,
  endpoint: string,
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<Set<string>> {
  const base = endpoint.replace(/\/$/, "");
  const url = `${base}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=1`;
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to list Hugging Face repo files: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json() as Array<{ path?: string }>;
  return new Set(payload.map((item: { path?: string }) => item.path).filter((path: string | undefined): path is string => typeof path === "string"));
}

function pickWhisperBeamsearchFile(files: ReadonlySet<string>, quantization: string): string | null {
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

async function resolveGigaamFromBase(
  baseUrl: string,
  config: { version?: unknown; [key: string]: unknown },
  quantization: string,
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<
  | { mode: "rnnt"; encoder: string; decoder: string; joint: string; vocabularyText: string }
  | { mode: "ctc"; ctcModel: string; vocabularyText: string }
> {
  const version = config.version ?? "v2";
  const vocabName = `${version}_vocab.txt`;
  const vocab = await fetchFirstExistingText(baseUrl, [vocabName, "vocab.txt", "tokens.txt"], fetchImpl, headers);

  const rnntPrefixes = [`${version}_rnnt`, `${version}_e2e_rnnt`];
  for (const prefix of rnntPrefixes) {
    const encoder = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_encoder.onnx`, quantization),
      fetchImpl,
      headers,
    ).catch((): null => null);
    const decoder = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_decoder.onnx`, quantization),
      fetchImpl,
      headers,
    ).catch((): null => null);
    const joint = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_joint.onnx`, quantization),
      fetchImpl,
      headers,
    ).catch((): null => null);

    if (encoder && decoder && joint) {
      return {
        mode: "rnnt",
        encoder,
        decoder,
        joint,
        vocabularyText: vocab,
      };
    }
  }

  const ctcCandidates = [`${version}_ctc.onnx`, `${version}_e2e_ctc.onnx`];
  for (const candidate of ctcCandidates) {
    const model = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(candidate, quantization),
      fetchImpl,
      headers,
    ).catch((): null => null);
    if (model) {
      return {
        mode: "ctc",
        ctcModel: model,
        vocabularyText: vocab,
      };
    }
  }

  throw new Error("Could not resolve GigaAM RNNT or CTC model files.");
}

async function resolveNemoConformerFromBase(
  baseUrl: string,
  quantization: string,
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<
  | { mode: "rnnt"; encoder: string; decoderJoint: string }
  | { mode: "ctc"; ctcModel: string }
> {
  const encoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("encoder-model.onnx", quantization),
    fetchImpl,
    headers,
  ).catch((): null => null);
  const decoderJoint = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("decoder_joint-model.onnx", quantization),
    fetchImpl,
    headers,
  ).catch((): null => null);

  if (encoder && decoderJoint) {
    return { mode: "rnnt", encoder, decoderJoint };
  }

  const ctcModel = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("model.onnx", quantization),
    fetchImpl,
    headers,
  ).catch((): null => null);
  if (ctcModel) {
    return { mode: "ctc", ctcModel };
  }

  throw new Error("Could not resolve nemo-conformer RNNT or CTC artifacts.");
}

async function resolveSherpaFromBase(
  baseUrl: string,
  quantization: string,
  fetchImpl: typeof fetch,
  headers?: HeadersMap,
): Promise<{ encoder: string; decoder: string; joiner: string; tokensText: string } | null> {
  const amOnnxProbe = modelFilenameCandidates("am-onnx/encoder.onnx", quantization);
  const amProbe = modelFilenameCandidates("am/encoder.onnx", quantization);
  const amDir = await resolveFirstExistingUrl(baseUrl, [...amOnnxProbe, ...amProbe], fetchImpl, headers)
    .then((url): "am-onnx" | "am" => (url.includes("/am-onnx/") ? "am-onnx" : "am"))
    .catch((): null => null);
  if (!amDir) {
    return null;
  }

  const encoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/encoder.onnx`, quantization),
    fetchImpl,
    headers,
  );
  const decoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/decoder.onnx`, quantization),
    fetchImpl,
    headers,
  );
  const joiner = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/joiner.onnx`, quantization),
    fetchImpl,
    headers,
  );
  const tokensPath = await resolveFirstExistingUrl(baseUrl, ["lang/tokens.txt", "tokens.txt"], fetchImpl, headers);
  const tokensText = await fetchText(tokensPath, fetchImpl, headers);

  return { encoder, decoder, joiner, tokensText };
}

function vadModelCandidates(quantization = "int8"): string[] {
  if (quantization && quantization !== "none" && quantization !== "float32" && quantization !== "fp32") {
    return [`onnx/model_${quantization}.onnx`, "onnx/model.onnx", "model.onnx"];
  }
  return ["onnx/model.onnx", "onnx/model_int8.onnx", "model.onnx"];
}

function resolveVadOption(options: BrowserModelLoadOptions = {}): VadDetector | null {
  return options.vadModel ?? options.vad ?? null;
}

function attachVadIfProvided(
  asrModel: AsrTranscriber,
  options: BrowserModelLoadOptions = {},
): AsrTranscriber {
  const vadModel = resolveVadOption(options);
  if (!vadModel) {
    return asrModel;
  }
  return withVadModel(asrModel, vadModel, options.vadOptions);
}

async function createAsrModelWithVad(
  params: Record<string, unknown>,
  options: BrowserModelLoadOptions = {},
): Promise<AsrTranscriber> {
  const asrModel = await createAsrModel(params);
  return attachVadIfProvided(asrModel, options);
}

/** Load a VAD model from a browser-accessible base URL. */
export async function loadLocalVadModel(baseUrl: string, options: BrowserModelLoadOptions = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const quantization = options.quantization ?? "int8";
  const headers = options.headers;
  const modelUrl = await resolveFirstExistingUrl(baseUrl, vadModelCandidates(quantization), fetchImpl, headers);
  return createSileroVadModel({
    modelPath: modelUrl,
    sessionOptions: options.sessionOptions,
    options: options.vadOptions,
  });
}

/** Load an ASR model from a browser-accessible base URL. */
export async function loadLocalModel(baseUrl: string, options: BrowserModelLoadOptions = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }

  const quantization = options.quantization ?? "int8";
  const headers = options.headers;

  const configText = await fetchTextOptional(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  if (!configText) {
    const sherpa = await resolveSherpaFromBase(baseUrl, quantization, fetchImpl, headers);
    if (!sherpa) {
      throw new Error("Could not detect model type: missing config.json and no sherpa am-onnx/am files.");
    }
    return createAsrModelWithVad({
      modelType: "sherpa-transducer",
      decoderKind: "sherpa-transducer",
      config: { sample_rate: options.sampleRate ?? 16000, max_tokens_per_step: 10 },
      encoderModel: sherpa.encoder,
      decoderModel: sherpa.decoder,
      decoderJointModel: sherpa.joiner,
      vocabularyText: sherpa.tokensText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "whisper-ort") {
    const modelCandidates = options.whisperModelCandidates ?? [
      "whisper-base_beamsearch.int8.onnx",
      "whisper-base_beamsearch.onnx",
    ];
    const whisperModel = await resolveFirstExistingUrl(baseUrl, modelCandidates, fetchImpl, headers);

    const vocabJson = await fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers);
    const addedTokensJson = await probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
      ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
      : "{}";

    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel,
      vocabJson,
      addedTokensJson,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "whisper-hf") {
    const encoderPath = requireString(spec.encoder, "whisper config is missing encoder model path.");
    const decoderPath = requireString(spec.decoderJoint, "whisper config is missing decoder model path.");
    const [encoderModel, decoderJointModel, vocabJson, hasAddedTokens] = await Promise.all([
      resolveModelUrl(baseUrl, encoderPath, { fetchImpl, headers, quantization }),
      resolveModelUrl(baseUrl, decoderPath, { fetchImpl, headers, quantization }),
      fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers),
      probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers),
    ]);

    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      encoderModel,
      decoderJointModel,
      vocabJson,
      addedTokensJson: hasAddedTokens
        ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
        : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "gigaam") {
    const artifacts = await resolveGigaamFromBase(baseUrl, config, quantization, fetchImpl, headers);
    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "gigaam-rnnt",
        config,
        encoderModel: artifacts.encoder,
        decoderModel: artifacts.decoder,
        decoderJointModel: artifacts.joint,
        vocabularyText: artifacts.vocabularyText,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      }, options);
    }

    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: artifacts.ctcModel,
      vocabularyText: artifacts.vocabularyText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  if (spec.decoderKind === "nemo-conformer") {
    const artifacts = await resolveNemoConformerFromBase(baseUrl, quantization, fetchImpl, headers);
    const vocabularyText = await fetchFirstExistingText(baseUrl, spec.vocabCandidates, fetchImpl, headers);

    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "rnnt",
        config,
        encoderModel: artifacts.encoder,
        decoderJointModel: artifacts.decoderJoint,
        vocabularyText,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      }, options);
    }

    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: artifacts.ctcModel,
      vocabularyText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    }, options);
  }

  const [encoderModel, decoderJointModel, vocabularyText, preprocessorModel] = await Promise.all([
    resolveModelUrl(
      baseUrl,
      requireString(spec.encoder, `model type '${modelType}' is missing encoder path.`),
      { fetchImpl, headers, quantization },
    ),
    spec.decoderJoint
      ? resolveModelUrl(baseUrl, spec.decoderJoint, { fetchImpl, headers, quantization })
      : Promise.resolve(null),
    (async () => {
      const toneText = toneVocabularyTextFromConfig(config);
      if (toneText) {
        return toneText;
      }
      const vocab = await fetchFirstExistingText(baseUrl, spec.vocabCandidates, fetchImpl, headers);
      return spec.vocabCandidates.includes("vocab.json") && vocab.trim().startsWith("{")
        ? vocabJsonToText(vocab)
        : vocab;
    })(),
    spec.preprocessor
      ? resolveModelUrl(baseUrl, spec.preprocessor, { fetchImpl, headers, quantization })
      : Promise.resolve(null),
  ]);

  return createAsrModelWithVad({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel,
    encoderModel,
    decoderJointModel,
    vocabularyText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions,
  }, options);
}

/** Load an ASR model by Hugging Face repo id directly in browser. */
export async function loadHuggingfaceModel(repoId: string, options: BrowserModelLoadOptions = {}) {
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

  const configText = await fetchTextOptional(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  if (!configText) {
    return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
  }
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "whisper-ort") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, endpoint, fetchImpl, headers);
    const whisperModelPath = pickWhisperBeamsearchFile(repoFiles, options.quantization ?? "int8");
    if (!whisperModelPath) {
      throw new Error("Could not find whisper-ort beamsearch ONNX in Hugging Face repo.");
    }

    return createAsrModelWithVad({
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
    }, options);
  }

  return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
}

/** Load a VAD model by Hugging Face repo id directly in browser. */
export async function loadHuggingfaceVadModel(repoId: string, options: BrowserModelLoadOptions = {}) {
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

  return loadLocalVadModel(baseUrl, { ...options, headers, fetch: fetchImpl });
}
