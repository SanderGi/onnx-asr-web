export function int64TensorValues(values) {
  return BigInt64Array.from(values.map((value) => BigInt(value)));
}

export function readScalarInt(tensor) {
  if (!tensor || !tensor.data || tensor.data.length === 0) {
    throw new Error("Expected scalar tensor with data.");
  }
  const value = tensor.data[0];
  return typeof value === "bigint" ? Number(value) : value;
}

export function firstExistingInputName(session, candidates, fallbackIndex) {
  for (const name of candidates) {
    if (session.inputNames.includes(name)) {
      return name;
    }
  }
  return session.inputNames[fallbackIndex];
}
