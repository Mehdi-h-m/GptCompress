"""Read bounded safetensors slices without loading an inference runtime."""

import json
import mmap
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

from .inspection import stats

MAX_SIDE = 512
TYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}


@dataclass(frozen=True)
class Weight:
    path: Path
    name: str
    shape: tuple
    dtype: str
    stamp: tuple

    @classmethod
    def from_inventory(cls, version, tensor):
        path = Path(version["path"]) / tensor["file"]
        info = path.stat()
        return cls(path, tensor["name"], tuple(tensor["shape"]), tensor["dtype"],
                   (info.st_size, info.st_mtime_ns))

    def read(self, rows, cols):
        dtype = TYPES.get(self.dtype)
        if not dtype:
            raise ValueError(f"Weight dtype {self.dtype} is unavailable in this inspector.")
        with self.path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if self.stamp != (info.st_size, info.st_mtime_ns):
                raise ValueError("The checkpoint changed. Reopen the model before inspecting it.")
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size <= 100_000_000:
                raise ValueError("Invalid safetensors header.")
            entry = json.loads(stream.read(size)).get(self.name)
            if not entry or tuple(entry["shape"]) != self.shape or entry["dtype"] != self.dtype:
                raise ValueError("The tensor no longer matches the checkpoint inventory.")
            begin, end = entry["data_offsets"]
            if not 0 <= begin <= end <= info.st_size - size - 8:
                raise ValueError("Safetensors offsets are outside the file.")
            if end - begin != np.prod(self.shape, dtype=np.int64) * np.dtype(dtype).itemsize:
                raise ValueError("Safetensors offsets do not match the tensor shape.")
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                weight = np.ndarray(self.shape, dtype=dtype, buffer=mapping, offset=8 + size + begin)
                if self.dtype == "BF16":
                    tile = (weight[rows, cols].astype(np.uint32) << 16).view(np.float32)
                else:
                    tile = weight[rows, cols].astype(np.float32)
                del weight
        return tile


@dataclass(frozen=True)
class MatrixSource:
    label: str
    identity: str
    shape: tuple
    weights: tuple

    @property
    def representation(self):
        if len(self.weights) == 2:
            return "Logical weight · A × B reconstructed in FP32"
        return f"Stored {self.weights[0].dtype} values read as FP32"

    def validate(self):
        for weight in self.weights:
            info = weight.path.stat()
            if (info.st_size, info.st_mtime_ns) != weight.stamp:
                raise ValueError("The checkpoint changed. Reopen the matrix inspector to refresh it.")

    def read(self, rows, columns):
        if len(self.weights) == 1:
            return self.weights[0].read(rows, columns)
        a, b = self.weights
        return a.read(rows, slice(None)) @ b.read(slice(None), columns)


def projection_sources(version, projection):
    modules = {module["name"]: module for module in version["modules"]}
    module = modules.get(projection, {})
    matrix = module.get("matrix", {})
    tensors = {tensor["name"]: tensor for tensor in module.get("tensors", [])}
    identity = version["identity"]
    if "dense" in matrix:
        weight = Weight.from_inventory(version, tensors[matrix["dense"]])
        return [MatrixSource("Dense weight W", identity, weight.shape, (weight,))]
    if "left" in matrix and "right" in matrix:
        a, b = (Weight.from_inventory(version, tensors[matrix[key]]) for key in ("left", "right"))
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("The stored factors have incompatible shapes.")
        return [MatrixSource("Logical weight A × B", identity, (a.shape[0], b.shape[1]), (a, b)),
                MatrixSource("Factor A · Output", identity, a.shape, (a,)),
                MatrixSource("Factor B · Input", identity, b.shape, (b,))]
    return []


def heatmap_cells(tile):
    step = max(1, (max(tile.shape) + 63) // 64)
    rows, cols = tile.shape
    padded = np.pad(tile, ((0, (-rows) % step), (0, (-cols) % step)), constant_values=np.nan)
    boxes = padded.reshape(padded.shape[0] // step, step, padded.shape[1] // step, step)
    boxes = boxes.transpose(0, 2, 1, 3).reshape(padded.shape[0] // step, padded.shape[1] // step, -1)
    index = np.argmax(np.where(np.isfinite(boxes), np.abs(boxes), -1), axis=-1)
    return np.take_along_axis(boxes, index[..., None], axis=-1)[..., 0], step


@lru_cache(maxsize=16)
def inspect_slice(source, row, col, side, spectrum=False):
    """Return slice-only measurements, with bounded cached display data."""
    if not 1 <= side <= MAX_SIDE or not (0 <= row < source.shape[0] and 0 <= col < source.shape[1]):
        raise ValueError("Choose a slice inside the matrix, up to 512 rows and columns.")
    rows = slice(row, min(row + side, source.shape[0]))
    cols = slice(col, min(col + side, source.shape[1]))
    tile = source.read(rows, cols)
    finite = np.isfinite(tile)
    cleaned = np.where(finite, tile, 0).astype(np.float64)
    values = tile[finite].astype(np.float64)
    histogram = np.histogram(values, bins=32) if values.size else (np.zeros(32), np.linspace(-1, 1, 33))
    heatmap, step = heatmap_cells(tile)
    singular = np.linalg.svd(tile.astype(np.float64), compute_uv=False) if spectrum and finite.all() else None
    return {"shape": tile.shape, "statistics": stats(tile), "heatmap": heatmap, "step": step,
            "histogram": histogram, "row_norms": np.linalg.norm(cleaned, axis=1),
            "column_norms": np.linalg.norm(cleaned, axis=0), "spectrum": singular,
            "spectrum_unavailable": bool(spectrum and singular is None)}
