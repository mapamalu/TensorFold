from __future__ import annotations

import pytest

from tensorfold.vision.config import validate_vision_config
from tensorfold.vision.qwen_checkpoint import vision_key


def test_glm_vision_config_accepts_the_native_checkpoint():
    config = {"model_type": "glm5_next", "image_token_id": 154854, "image_start_token_id": 154830,
              "image_end_token_id": 154831, "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 4096, "hidden_size": 1024}}
    assert validate_vision_config(config, "glm5_next")["hidden_size"] == 1024


def test_glm_vision_config_requires_image_tokens_and_matching_width():
    config = {"model_type": "glm5_next", "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 2048}}
    with pytest.raises(ValueError, match="output width"):
        validate_vision_config(config, "glm5_next")
    config["vision_config"]["out_hidden_size"] = 4096
    with pytest.raises(ValueError, match="image-token configuration"):
        validate_vision_config(config, "glm5_next")


def test_glm_vision_loader_recognizes_legacy_mlx_vision_model_prefix():
    assert vision_key("vision_model.blocks.0.attn.qkv.weight") == "blocks.0.attn.qkv.weight"


def test_glm_vision_rejects_cuda_before_reading_checkpoint(monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    from tensorfold import families, serve_options

    def read_config(path):
        raise AssertionError('unsupported backend must be rejected before reading checkpoint')

    monkeypatch.setattr(families, 'read_config', read_config)
    with pytest.raises(ValueError, match='GLM.*MLX-only'):
        serve_options.check(Namespace(vision=True), SimpleNamespace(model_type='glm5_next'), 'cuda', 'unused')


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen3_8_flash_next"])
def test_flash_next_vision_options_cover_both_model_aliases(model_type, monkeypatch):
    from types import SimpleNamespace

    from tensorfold import families, serve_options
    from tensorfold.families import qwen4_exp

    config = {"model_type": model_type, "text_config": {"hidden_size": 2560},
              "vision_config": {"model_type": "qwen3_5_vision", "out_hidden_size": 2560, "hidden_size": 1024}}
    monkeypatch.setattr(families, "read_config", lambda _: config)
    args = SimpleNamespace(vision=True, kv_dtype="int8")
    family = SimpleNamespace(model_type=model_type, package=qwen4_exp)

    assert serve_options.check(args, family, "cuda", "unused") is None
    with pytest.raises(ValueError, match="CUDA engine; the MLX path has no image tower"):
        serve_options.check(args, family, "mlx", "unused")


def test_flash_next_family_alias_is_registered_for_vision():
    from tensorfold.families.qwen4_exp import MODEL_TYPES

    assert "qwen3_8_flash_next" in MODEL_TYPES
    assert validate_vision_config({"text_config": {"hidden_size": 2560},
                                   "vision_config": {"out_hidden_size": 2560}},
                                  "qwen3_8_flash_next")["out_hidden_size"] == 2560
