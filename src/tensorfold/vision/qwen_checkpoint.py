"""Read only a local checkpoint's vision tensors, including shards shared with language weights."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

PREFIXES = ("model.language_model.visual.", "model.visual.", "vision_tower.", "vision_model.", "visual.")
DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2", "I64": "<i8", "I32": "<i4",
          "I16": "<i2", "I8": "i1", "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "?"}


def vision_key(name: str) -> str | None:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"Incomplete safetensors header: {path.name}")
        length = struct.unpack("<Q", size)[0]
        if not 2 <= length <= min(64 * 1024**2, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    return header, 8 + length


def vision_tensors(model_dir: Path, *, weights_path: Path | None = None) -> dict[str, tuple[Path, dict, int]]:
    """Inspect headers only and return local tower names with their file, tensor metadata and data start."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    expected = None
    if weights_path is not None:
        files = [Path(weights_path)]
    elif index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        expected = {name: shard for name, shard in mapping.items() if vision_key(name) is not None}
        shards = sorted(set(expected.values()))
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in shards):
            raise ValueError("Vision checkpoint index contains an invalid shard path")
        files = [model_dir / shard for shard in shards]
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    result, seen = {}, set()
    for path in files:
        header, begin = _header(path)
        for name, item in header.items():
            local = vision_key(name)
            if local is None or "position_ids" in local:
                continue
            if expected is not None and name not in expected:
                continue
            if local in result:
                raise ValueError(f"Duplicate vision tensor: {local}")
            result[local] = (path, item, begin)
            seen.add(name)
    missing = set(expected or ()) - seen
    missing = {name for name in missing if "position_ids" not in name}
    if missing:
        raise ValueError(f"Vision checkpoint is missing indexed tensors: {sorted(missing)[:3]}")
    if not result:
        raise ValueError("This local checkpoint has no vision tower weights; use a complete multimodal checkpoint")
    return result


def load_vision_weights(tensors: dict[str, tuple[Path, dict, int]], mx: Any) -> dict[str, Any]:
    """Read selected byte ranges rather than materializing the language tensors in mixed shards."""
    weights = {}
    for name, (path, item, begin) in tensors.items():
        dtype = item.get("dtype")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported vision tensor dtype {dtype}: {name}")
        shape = item.get("shape", ())
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError(f"Invalid vision tensor shape: {name}")
        offsets = item.get("data_offsets", ())
        if len(offsets) != 2 or any(not isinstance(n, int) for n in offsets):
            raise ValueError(f"Invalid vision tensor offsets: {name}")
        start, end = offsets
        dt = np.dtype(DTYPES[dtype])
        if start < 0 or end - start != math.prod(shape) * dt.itemsize or begin + end > path.stat().st_size:
            raise ValueError(f"Invalid vision tensor range: {name}")
        with path.open("rb") as stream:
            stream.seek(begin + start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Incomplete vision tensor: {name}")
        array = mx.array(np.frombuffer(raw, dtype=dt).reshape(shape).copy())
        weights[name] = array.view(mx.bfloat16) if dtype == "BF16" else array
    return weights


def dequantize_vision_nvfp4(weight: np.ndarray, scale: np.ndarray, scale2: float) -> np.ndarray:
    """Decodifica NVFP4 ModelOpt [N,K/2] in pesi float32 [N,K] secondo il layout salvato."""
    from tensorfold.cuda.nvfp4 import format as nvfp4

    weight = np.asarray(weight)
    scale = np.asarray(scale)
    if weight.dtype != np.uint8 or weight.ndim != 2 or weight.shape[1] % 8:
        raise ValueError("pesi visivi NVFP4 non validi: attesi byte [N,K/2] con K divisibile per 16")
    rows, half = weight.shape
    width = half * 2
    if scale.dtype != np.uint8 or scale.shape != (rows, width // 16):
        raise ValueError("scale visive NVFP4 non valide: attese scale E4M3 [N,K/16]")
    if not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError("weight_scale_2 NVFP4 deve essere uno scalare positivo e finito")
    decoded_scale = nvfp4.e4m3(scale)
    if not np.isfinite(decoded_scale).all() or np.any(decoded_scale < 0):
        raise ValueError("le scale E4M3 visive NVFP4 contengono valori non validi")
    result = nvfp4.dequant("nvfp4", weight, scale, float(scale2))
    if not np.isfinite(result).all():
        raise ValueError("la decodifica NVFP4 visiva ha prodotto valori non finiti")
    return result


def dequantize_vision_mxfp8(weight_bits: np.ndarray, scale_bits: np.ndarray) -> np.ndarray:
    """Decodifica pesi visivi MXFP8 E4M3 con scale E8M0 per gruppi di 32 elementi."""
    from tensorfold.cuda.nvfp4 import format as nvfp4

    weight_bits = np.asarray(weight_bits)
    scale_bits = np.asarray(scale_bits)
    if weight_bits.dtype != np.uint8 or weight_bits.ndim != 2 or weight_bits.shape[1] % 32:
        raise ValueError("pesi visivi MXFP8 non validi: attesi bit E4M3 [N,K] con K divisibile per 32")
    expected = (weight_bits.shape[0], weight_bits.shape[1] // 32)
    if scale_bits.dtype != np.uint8 or scale_bits.shape != expected:
        raise ValueError("scale visive MXFP8 non valide: attesi byte E8M0 [N,K/32]")
    decoded_weight = nvfp4.e4m3(weight_bits)
    decoded_scale = nvfp4.e8m0(scale_bits)
    if not np.isfinite(decoded_weight).all() or not np.isfinite(decoded_scale).all():
        raise ValueError("i tensori visivi MXFP8 contengono codici non finiti")
    result = nvfp4.dequant("mxfp8", weight_bits, scale_bits)
    if not np.isfinite(result).all():
        raise ValueError("la decodifica MXFP8 visiva ha prodotto valori non finiti")
    return result


def load_vision_torch_weights(sources: dict[str, tuple[Path, dict, int]], read_tensor: Any,
                              device: Any) -> dict[str, Any]:
    """Carica la torre Qwen CUDA in BF16, espandendo solo i moduli ModelOpt NVFP4 e MXFP8."""
    import torch

    weights = {}
    consumed = set()

    def raw_bytes(name: str, tensor: Any) -> np.ndarray:
        dtype = sources[name][1]["dtype"]
        if dtype == "F8_E4M3":
            tensor = tensor.view(torch.uint8)
        elif dtype != "U8":
            raise ValueError(f"{name}: la rappresentazione quantizzata deve essere byte U8 o F8_E4M3")
        return tensor.detach().to(device="cpu").contiguous().numpy()

    for name in sorted(sources):
        if name in consumed:
            continue
        if name.endswith((".weight_scale", ".weight_scale_2")):
            raise ValueError(f"metadato di quantizzazione visivo senza peso associato: {name}")
        item = sources[name][1]
        dtype = item["dtype"]
        if name.endswith(".weight") and dtype in ("U8", "F8_E4M3"):
            base = name[:-7]
            scale_name, scale2_name = base + ".weight_scale", base + ".weight_scale_2"
            if scale_name not in sources:
                raise ValueError(f"{name}: manca weight_scale per il peso visivo quantizzato")
            packed = raw_bytes(name, read_tensor(name))
            scale = raw_bytes(scale_name, read_tensor(scale_name))
            consumed.update((name, scale_name))
            if dtype == "U8":
                if scale2_name not in sources or sources[scale2_name][1]["dtype"] != "F32":
                    raise ValueError(f"{name}: NVFP4 richiede weight_scale_2 F32 scalare")
                scale2 = read_tensor(scale2_name)
                if scale2.numel() != 1 or tuple(scale2.shape) != ():
                    raise ValueError(f"{scale2_name}: atteso uno scalare F32")
                decoded = dequantize_vision_nvfp4(packed, scale, float(scale2.item()))
                consumed.add(scale2_name)
            else:
                if scale2_name in sources:
                    raise ValueError(f"{name}: MXFP8 non ammette weight_scale_2")
                decoded = dequantize_vision_mxfp8(packed, scale)
            value = torch.from_numpy(np.ascontiguousarray(decoded)).to(dtype=torch.bfloat16)
        else:
            if dtype not in ("BF16", "F16", "F32"):
                raise ValueError(f"{name}: formato visivo non quantizzato non supportato ({dtype})")
            value = read_tensor(name)
        weights[name] = value.to(device=device, dtype=torch.bfloat16)
        consumed.add(name)

    if consumed != set(sources):
        raise ValueError(f"tensori visivi non caricati: {sorted(set(sources) - consumed)[:5]}")
    return weights


def quantization_predicate(config: dict, weights: dict[str, Any]):
    """Respect per-module overrides only where the checkpoint actually contains packed tensors."""
    quant = config.get("quantization") or config.get("quantization_config") or {}
    overrides = {vision_key(name): value for name, value in quant.items() if vision_key(name) is not None}

    def predicate(path: str, module: Any):
        if f"{path}.scales" not in weights:
            return False
        if not hasattr(module, "to_quantized"):
            raise ValueError(f"Vision module cannot load quantized weights: {path}")
        value = overrides.get(path)
        if value is False:
            raise ValueError(f"Vision quantization metadata contradicts packed weights: {path}")
        settings = {key: quant[key] for key in ("bits", "group_size", "mode") if key in quant}
        if isinstance(value, dict):
            settings.update({key: value[key] for key in ("bits", "group_size", "mode") if key in value})
        if "bits" not in settings or "group_size" not in settings:
            raise ValueError(f"Vision quantization metadata is missing bits or group_size: {path}")
        settings.setdefault("mode", "affine")
        return settings

    return predicate
