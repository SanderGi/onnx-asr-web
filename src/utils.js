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

export function readIntArray(tensor) {
  if (!tensor || !tensor.data) {
    throw new Error("Expected tensor with data.");
  }
  return Array.from(tensor.data, (value) =>
    typeof value === "bigint" ? Number(value) : value
  );
}

export function transpose3d102(data, dims) {
  const [d0, d1, d2] = dims;
  const out = new Float32Array(d1 * d0 * d2);

  for (let i0 = 0; i0 < d0; i0 += 1) {
    for (let i1 = 0; i1 < d1; i1 += 1) {
      for (let i2 = 0; i2 < d2; i2 += 1) {
        const inIndex = i0 * d1 * d2 + i1 * d2 + i2;
        const outIndex = i1 * d0 * d2 + i0 * d2 + i2;
        out[outIndex] = data[inIndex];
      }
    }
  }

  return { data: out, dims: [d1, d0, d2] };
}

export function transpose4d2103(data, dims, ctor) {
  const [d0, d1, d2, d3] = dims;
  const out = new ctor(d2 * d1 * d0 * d3);

  for (let i0 = 0; i0 < d0; i0 += 1) {
    for (let i1 = 0; i1 < d1; i1 += 1) {
      for (let i2 = 0; i2 < d2; i2 += 1) {
        for (let i3 = 0; i3 < d3; i3 += 1) {
          const inIndex = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;
          const outIndex = i2 * d1 * d0 * d3 + i1 * d0 * d3 + i0 * d3 + i3;
          out[outIndex] = data[inIndex];
        }
      }
    }
  }

  return { data: out, dims: [d2, d1, d0, d3] };
}

export function firstExistingInputName(session, candidates, fallbackIndex) {
  for (const name of candidates) {
    if (session.inputNames.includes(name)) {
      return name;
    }
  }
  return session.inputNames[fallbackIndex];
}

export function sliceFirstAxis(data, dims, index, ctor) {
  const chunkSize = dims.slice(1).reduce((acc, item) => acc * item, 1);
  const start = index * chunkSize;
  const end = start + chunkSize;
  return {
    data: new ctor(data.slice(start, end)),
    dims: dims.slice(1),
  };
}
