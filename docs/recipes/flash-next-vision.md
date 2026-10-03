# Flash Next CUDA vision

Run a complete local Flash Next checkpoint with `--vision --parallel 2` (or more), on one CUDA GPU.
Qwen3.8 Flash Next accepts images and videos. Image rows replace embeddings in every hyperconnection stream;
video frame groups use the checkpoint's video marker and timestamps. Multimodal RoPE follows the checkpoint's
sections, and text decode continues with the media-position offset. Media prompts always prefill from the start
and are never retained in the text-prefix cache, since identical placeholder IDs can name different pixels.
Grammar and ignore-EOS settings remain available.

Mia-AiLab's NVFP4 checkpoint stores a mixed precision vision tower: ModelOpt NVFP4 projections with E4M3
block scales and scalar `weight_scale_2`, MXFP8 projections with E8M0 block scales, and ordinary floating-point
tensors. CUDA expands the quantized vision projections to BF16 when loading the tower. Admission therefore
counts the expanded resident weights and workspace, rather than the smaller packed checkpoint bytes. Keep a
host-memory reserve appropriate for the rest of the system when sizing the model and request cache.

A floating-point tower in the indexed checkpoint is discovered normally. EXL3 packs whose vision tower is a
quantized sidecar need a one-time CPU conversion, stored outside the original model snapshot:

```bash
python -m tensorfold.vision.exl3_convert /models/vision_k6.safetensors /cache/vision-f16-v3.safetensors
TENSORFOLD_VISION_WEIGHTS=/cache/vision-f16-v3.safetensors tensorfold serve /models --vision --parallel 2
```

The converter decodes represented EXL3 weights, transposes matrices and combines split Q/K/V. It records the
source SHA256 and converter/dtype version; repeat conversions reuse a matching artifact and refuse to
replace a mismatched one. Conversion never runs in the serving loader. The FP16 artifact loads into the
existing BF16 CUDA tower; rounding may differ from native quantized vision execution, so compare image
features and quality for the checkpoint in use. The original weights stay unchanged.

Admission counts an external tower separately, including its expanded resident bytes. It reserves 4 GiB of
image workspace by default; `TENSORFOLD_VISION_WORKSPACE_MIB` sets a measured override from 0 to 16384 MiB.
Image and video requests cannot use yieldable background lanes. Video requires the CUDA path and
`--parallel 2` or more. Distributed vision and serial-only Flash Next visual serving are not supported by this
port.

The integration is adapted from MiaAI-Lab patch 0008, with its MIT license in `LICENSES/MiaAI-Lab-MIT.txt`.
