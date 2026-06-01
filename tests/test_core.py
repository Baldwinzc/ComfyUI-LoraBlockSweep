"""Smoke tests for the pure block-spec logic.

These don't run sampling — they cover the parsing/classification helpers that
are easy to break and have no ComfyUI dependency: BlockSpec name generation,
key classification, group parsing, custom-weight parsing, mode parsing, and
strength-map construction.
"""
import pytest

from lora_block_weight._core import (
    BlockSpec,
    build_block_strengths,
    build_group_strengths,
    classify_key,
    parse_custom_weights,
    parse_group,
    parse_modes,
)
from lora_block_weight._flux import FLUX_SPEC
from lora_block_weight._sd35 import SD35_SPEC


def test_flux_block_names():
    assert FLUX_SPEC.block_names[0] == "D00"
    assert FLUX_SPEC.block_names[18] == "D18"
    assert FLUX_SPEC.block_names[19] == "S00"
    assert FLUX_SPEC.block_names[-1] == "S37"
    assert len(FLUX_SPEC.block_names) == 57


def test_blockspec_rejects_multichar_prefix():
    with pytest.raises(ValueError):
        BlockSpec(model_key="x", display_name="X",
                  tag_groups={"DD": (2, r"foo\.(\d+)\.")},
                  default_groups="")


def test_classify_key_flux():
    assert classify_key(FLUX_SPEC,
                        "diffusion_model.double_blocks.0.img_attn.qkv.weight") == "D00"
    assert classify_key(FLUX_SPEC,
                        "diffusion_model.single_blocks.37.linear1.weight") == "S37"
    # out-of-range index and unrelated keys fall through to 'extras'
    assert classify_key(FLUX_SPEC,
                        "diffusion_model.double_blocks.99.x.weight") == "extras"
    assert classify_key(FLUX_SPEC, "diffusion_model.final_layer.weight") == "extras"


def test_classify_key_accepts_tuple_form():
    # comfy.lora.load_lora keys can be (name, offset, fn) tuples
    key = ("diffusion_model.single_blocks.5.linear1.weight", 0, None)
    assert classify_key(FLUX_SPEC, key) == "S05"


def test_parse_group_ranges_and_singles():
    assert parse_group(FLUX_SPEC, "D00-D02") == ["D00", "D01", "D02"]
    assert parse_group(FLUX_SPEC, "S15") == ["S15"]
    assert parse_group(FLUX_SPEC, "D00-D01,S20") == ["D00", "D01", "S20"]
    # reversed range is normalized
    assert parse_group(FLUX_SPEC, "D02-D00") == ["D00", "D01", "D02"]


def test_parse_group_rejects_cross_prefix_and_oob():
    with pytest.raises(ValueError):
        parse_group(FLUX_SPEC, "D18-S00")      # range crosses D and S
    with pytest.raises(ValueError):
        parse_group(FLUX_SPEC, "S99")          # out of range
    with pytest.raises(ValueError):
        parse_group(FLUX_SPEC, "D00-")         # malformed


def test_parse_custom_weights_ok():
    n = len(SD35_SPEC.block_names)
    weights = ",".join(["1.0"] * n)
    per_block = parse_custom_weights(SD35_SPEC, weights, baseline_weight=1.0)
    assert per_block["J00"] == 1.0
    assert per_block["extras"] == 1.0
    assert len(per_block) == n + 1  # blocks + extras


def test_parse_custom_weights_wrong_count_raises():
    with pytest.raises(ValueError):
        parse_custom_weights(SD35_SPEC, "1.0,0.5", baseline_weight=1.0)


def test_parse_custom_weights_non_numeric_raises():
    n = len(SD35_SPEC.block_names)
    parts = ["1.0"] * n
    parts[3] = "oops"
    with pytest.raises(ValueError):
        parse_custom_weights(SD35_SPEC, ",".join(parts), baseline_weight=1.0)


def test_parse_modes():
    modes = parse_modes("knockout,solo,full,off")
    names = [m[0] for m in modes]
    assert names == ["knockout", "solo", "full", "off"]
    # (value, baseline) pairs
    assert modes[0][1:] == (0.0, 1.0)   # knockout: group off, others on
    assert modes[1][1:] == (1.0, 0.0)   # solo: group on, others off


def test_parse_modes_rejects_unknown():
    with pytest.raises(ValueError):
        parse_modes("knockout,bogus")


def test_build_block_strengths():
    s = build_block_strengths(FLUX_SPEC, "D00", target_value=0.0,
                              baseline_weight=1.0)
    assert s["D00"] == 0.0
    assert s["D01"] == 1.0
    assert s["extras"] == 1.0


def test_build_group_strengths():
    tags = parse_group(FLUX_SPEC, "D00-D02")
    s = build_group_strengths(FLUX_SPEC, tags, group_value=0.0,
                              baseline_weight=1.0)
    assert s["D00"] == 0.0 and s["D02"] == 0.0
    assert s["D03"] == 1.0
