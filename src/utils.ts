import type * as ort from "onnxruntime-web";

export function int64TensorValues(values: readonly number[]): BigInt64Array {
  return BigInt64Array.from(values.map((value) => BigInt(value)));
}

export function readScalarInt(tensor: ort.Tensor | null | undefined): number {
  if (!tensor || !tensor.data || tensor.data.length === 0) {
    throw new Error("Expected scalar tensor with data.");
  }
  const value = tensor.data[0];
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (typeof value === "number") {
    return value;
  }
  throw new Error(`Expected numeric scalar tensor, received ${typeof value}.`);
}

export function firstExistingInputName(
  session: ort.InferenceSession,
  candidates: readonly string[],
  fallbackIndex: number,
): string {
  for (const name of candidates) {
    if (session.inputNames.includes(name)) {
      return name;
    }
  }
  return session.inputNames[fallbackIndex];
}
