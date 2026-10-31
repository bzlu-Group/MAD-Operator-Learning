"""Convert generated tuples and the public MAD0 text data to R4 training files."""
import argparse
import json
from pathlib import Path

import numpy as np

from generate_balanced import sha, write_json


def prepare(generated, mad0_text, output):
    data = np.load(generated / "data.npy", mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 5402 or not np.isfinite(data).all():
        raise ValueError("Expected finite generated [g,f,u] tuples with 5402 columns")
    scale = np.max(np.abs(data[:, :2801]), axis=1)
    if np.any(scale <= 0):
        raise ValueError("Zero input scale")
    paired = np.asarray(data) / scale[:, None]
    mad0 = np.loadtxt(mad0_text, dtype=np.float32)
    if mad0.shape != data.shape or not np.isfinite(mad0).all():
        raise ValueError("MAD0 and solver arrays must have matching finite shapes")
    mad_scale = np.max(np.abs(mad0[:, :2801]), axis=1)
    if np.any(mad_scale <= 0):
        raise ValueError("Zero MAD0 input scale")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "inputs.npy", paired[:, :2801])
    np.save(output / "supervised.npy", paired)
    np.save(output / "mad0.npy", mad0 / mad_scale[:, None])
    record = {
        "samples": len(data),
        "normalization": "max(abs(g),abs(f)), independently for each tuple",
        "source_tuple_sha256": sha(generated / "data.npy"),
        "mad0_source_text_sha256": sha(mad0_text),
        "label_normalization_conversion_count": int(np.count_nonzero(scale != 1)),
        "mad0_input_scale_conversion_count": int(np.count_nonzero(mad_scale != 1)),
        "files": {name: sha(output / name) for name in
                  ("inputs.npy", "supervised.npy", "mad0.npy")},
    }
    write_json(output / "provenance.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--mad0-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.generated, args.mad0_text, args.output)
