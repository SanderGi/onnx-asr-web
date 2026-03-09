import { access, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText, toneVocabularyTextFromConfig } from "./model-types.js";

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

function resolveHfApiBase(endpoint = "https://huggingface.co") {
  return endpoint.replace(/\/$/, "");
}

function modelFilenameCandidates(filename, quantization = "int8") {
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

function sidecarCandidates(onnxPath) {
  if (!onnxPath.endsWith(".onnx")) {
    return [];
  }
  return [
    `${onnxPath}.data`,
    onnxPath.replace(/\.onnx$/, ".onnx_data"),
  ];
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
  await mkdir(dirname(filePath), { recursive: true });
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

function vocabJsonToText(vocabJsonText) {
  const parsed = JSON.parse(vocabJsonText);
  const entries = Object.entries(parsed)
    .filter(([, id]) => Number.isInteger(id))
    .sort((a, b) => a[1] - b[1]);
  if (entries.length === 0) {
    throw new Error("Invalid vocab.json mapping.");
  }
  return `${entries.map(([token, id]) => `${token} ${id}`).join("\n")}\n`;
}

async function walkFiles(root, prefix = "") {
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

function pickQuantizedByPattern(files, pattern, quantization = "int8") {
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

async function listHuggingFaceRepoFiles(repoId, revision, endpoint, requestOptions) {
  const apiBase = resolveHfApiBase(endpoint);
  const url = `${apiBase}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=1`;
  const response = await fetchRequired(url, requestOptions);
  const payload = await response.json();
  const files = payload
    .map((item) => item.path)
    .filter((path) => typeof path === "string");
  return new Set(files);
}

async function downloadPath(baseUrl, modelDir, relativePath, requestOptions, { required = true } = {}) {
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

async function ensureSidecars(baseUrl, modelDir, modelPath, requestOptions, repoFiles = null) {
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

async function resolveWhisperLocalArtifacts(modelDir, quantization) {
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

function pickGigaamVersion(config, files) {
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

function selectGigaamArtifacts(config, files, quantization) {
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

function selectNemoConformerArtifacts(files, quantization) {
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

function selectSherpaArtifacts(files, quantization) {
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

export async function loadLocalModel(modelDir, options = {}) {
  const quantization = options.quantization ?? "int8";
  const files = await walkFiles(modelDir);

  if (!files.includes("config.json")) {
    const sherpa = selectSherpaArtifacts(files, quantization);
    return createAsrModel({
      modelType: "sherpa-transducer",
      decoderKind: "sherpa-transducer",
      config: { sample_rate: options.sampleRate ?? 16000, max_tokens_per_step: 10 },
      encoderModel: join(modelDir, sherpa.encoder),
      decoderModel: join(modelDir, sherpa.decoder),
      decoderJointModel: join(modelDir, sherpa.joiner),
      vocabularyText: await readFile(join(modelDir, sherpa.tokens), "utf8"),
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  const configText = await readFile(join(modelDir, "config.json"), "utf8");
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);

  if (spec.decoderKind === "nemo-conformer") {
    const artifacts = selectNemoConformerArtifacts(files, quantization);
    const vocabulary = await selectOrReadVocabulary(modelDir, spec.vocabCandidates);

    if (artifacts.mode === "rnnt") {
      return createAsrModel({
        modelType,
        decoderKind: "rnnt",
        config,
        encoderModel: join(modelDir, artifacts.encoder),
        decoderJointModel: join(modelDir, artifacts.decoderJoint),
        vocabularyText: vocabulary.text,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      });
    }

    return createAsrModel({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: join(modelDir, artifacts.ctcModel),
      vocabularyText: vocabulary.text,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  if (spec.decoderKind === "gigaam") {
    const artifacts = selectGigaamArtifacts(config, files, quantization);
    const vocabPath = files.includes(artifacts.vocab) ? artifacts.vocab : (files.includes("vocab.txt") ? "vocab.txt" : null);
    if (!vocabPath) {
      throw new Error("GigaAM vocabulary file not found.");
    }

    if (artifacts.mode === "rnnt") {
      return createAsrModel({
        modelType,
        decoderKind: "gigaam-rnnt",
        config,
        encoderModel: join(modelDir, artifacts.encoder),
        decoderModel: join(modelDir, artifacts.decoder),
        decoderJointModel: join(modelDir, artifacts.joint),
        vocabularyText: await readFile(join(modelDir, vocabPath), "utf8"),
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      });
    }

    return createAsrModel({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: join(modelDir, artifacts.ctcModel),
      vocabularyText: await readFile(join(modelDir, vocabPath), "utf8"),
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  if (spec.decoderKind === "whisper-ort") {
    const artifacts = await resolveWhisperLocalArtifacts(modelDir, quantization);
    return createAsrModel({
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
    });
  }

  if (spec.decoderKind === "whisper-hf") {
    const encoderFile = await selectOrFallbackModelFile(modelDir, spec.encoder, quantization);
    const decoderFile = await selectOrFallbackModelFile(modelDir, spec.decoderJoint, quantization);

    return createAsrModel({
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
    });
  }

  const encoderFile = await selectOrFallbackModelFile(modelDir, spec.encoder, quantization);
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

  return createAsrModel({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel: preprocessorFile ? join(modelDir, preprocessorFile) : null,
    encoderModel: join(modelDir, encoderFile),
    decoderJointModel: decoderFile ? join(modelDir, decoderFile) : null,
    vocabularyText,
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

    const modelFiles = [spec.encoder, spec.decoderJoint];
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

  const modelFiles = [spec.encoder, ...(spec.decoderJoint ? [spec.decoderJoint] : []), ...(spec.preprocessor ? [spec.preprocessor] : [])];

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

export async function loadHuggingfaceModel(repoId, options = {}) {
  const modelDir = await downloadHuggingfaceModel(repoId, options);
  return loadLocalModel(modelDir, options);
}
