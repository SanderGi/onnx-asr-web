import { configureOrtWeb, createAsrModel } from "./asr-model.js";
import { detectModelType, parseConfigText, toneVocabularyTextFromConfig } from "./model-types.js";

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
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }

  return joinUrl(baseUrl, candidates[candidates.length - 1]);
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

async function fetchFirstExistingText(baseUrl, candidates, fetchImpl, headers) {
  const url = await resolveFirstExistingUrl(baseUrl, candidates, fetchImpl, headers);
  return fetchText(url, fetchImpl, headers);
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

async function resolveGigaamFromBase(baseUrl, config, quantization, fetchImpl, headers) {
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
    ).catch(() => null);
    const decoder = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_decoder.onnx`, quantization),
      fetchImpl,
      headers,
    ).catch(() => null);
    const joint = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_joint.onnx`, quantization),
      fetchImpl,
      headers,
    ).catch(() => null);

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
    ).catch(() => null);
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
    const addedTokensJson = await probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
      ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
      : "{}";

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

    return createAsrModel({
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
    });
  }

  if (spec.decoderKind === "gigaam") {
    const artifacts = await resolveGigaamFromBase(baseUrl, config, quantization, fetchImpl, headers);
    if (artifacts.mode === "rnnt") {
      return createAsrModel({
        modelType,
        decoderKind: "gigaam-rnnt",
        config,
        encoderModel: artifacts.encoder,
        decoderModel: artifacts.decoder,
        decoderJointModel: artifacts.joint,
        vocabularyText: artifacts.vocabularyText,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions,
      });
    }

    return createAsrModel({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: artifacts.ctcModel,
      vocabularyText: artifacts.vocabularyText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions,
    });
  }

  const [encoderModel, decoderJointModel, vocabularyText, preprocessorModel] = await Promise.all([
    resolveModelUrl(baseUrl, spec.encoder, { fetchImpl, headers, quantization }),
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
