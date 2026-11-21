#!/usr/bin/env python3
"""
Requantize abliterated Gemma 4 (26B-A4B-it) into group-64 symmetric INT4 compressed-tensors.
Integrates the orthogonalized layers (10-24) into the calibrated INT4 Strix Halo model structure.
"""

import os
import sys
import time
import argparse
import shutil
import torch
import safetensors.torch
import compressed_tensors

DEFAULT_SRC_INT4_DIR = "/root/.cache/huggingface/hub/models--jcbtc--Gemma4-26B-CIRU-AMD-STRIX-vLLM/snapshots/fdf479770466055d7aa5de127435626a5ad91e34"
DEFAULT_SRC_ABLIT_DIR = "/root/.cache/huggingface/gemma-4-26b-abliterated"
DEFAULT_DST_INT4_DIR = "/root/.cache/huggingface/gemma-4-26b-abliterated-int4"

def quantize_group64(w: torch.Tensor):
    """
    Quantize 2D weight tensor w to 4-bit signed int [-8, 7] with group_size=64
    and bit-pack to int32 using compressed-tensors specification.
    """
    out_dim, in_dim = w.shape
    assert in_dim % 64 == 0, f"in_dim {in_dim} not divisible by 64"
    g = w.view(out_dim, in_dim // 64, 64)
    scale = (g.abs().amax(dim=-1) / 7.5).clamp(min=1e-8).to(torch.bfloat16)
    scale_exp = scale.repeat_interleave(64, dim=1)
    q = torch.clamp(torch.round(w / scale_exp), -8, 7).to(torch.int8)
    packed = compressed_tensors.pack_to_int32(q, 4)
    return packed, scale

def main():
    parser = argparse.ArgumentParser(description="Requantize abliterated Gemma 4 to INT4 (compressed-tensors group-64)")
    parser.add_argument("--source-int4", default=DEFAULT_SRC_INT4_DIR, help="Base calibrated INT4 model directory")
    parser.add_argument("--source-ablit", "--source", dest="source_ablit", default=DEFAULT_SRC_ABLIT_DIR, help="Abliterated BF16 model directory")
    parser.add_argument("--output-int4", "--output", dest="output_int4", default=DEFAULT_DST_INT4_DIR, help="Destination INT4 output directory")
    args = parser.parse_args()

    SRC_INT4_DIR = args.source_int4
    SRC_ABLIT_DIR = args.source_ablit
    DST_INT4_DIR = args.output_int4

    t0 = time.time()
    print("=" * 80)
    print("Requantizing Abliterated Gemma 4 26B to INT4 (compressed-tensors group-64)")
    print(f"Source INT4 base:        {SRC_INT4_DIR}")
    print(f"Source Abliterated BF16: {SRC_ABLIT_DIR}")
    print(f"Destination INT4 model:  {DST_INT4_DIR}")
    print("=" * 80)

    os.makedirs(DST_INT4_DIR, exist_ok=True)

    # 1. Copy config and metadata files
    config_files = [
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json"
    ]
    for cfg in config_files:
        src = os.path.join(SRC_INT4_DIR, cfg)
        if not os.path.exists(src):
            src = os.path.join(SRC_ABLIT_DIR, cfg)
        if os.path.exists(src):
            dst = os.path.join(DST_INT4_DIR, cfg)
            shutil.copy2(src, dst)
            print(f"Synced metadata file: {cfg}")

    # 2. Hardlink/copy untouched shards (layers 0-9, 25-29, other, vision)
    untouched_layers = [f"model-layer-{i:02d}.safetensors" for i in range(10)] + \
                       [f"model-layer-{i:02d}.safetensors" for i in range(25, 30)] + \
                       ["model-other.safetensors", "model-vision.safetensors"]

    for fname in untouched_layers:
        src = os.path.join(SRC_INT4_DIR, fname)
        dst = os.path.join(DST_INT4_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f"Linked {len(untouched_layers)} untouched layer/module shards.")

    # 3. Open abliterated BF16 shard 1
    ablit_shard_path = os.path.join(SRC_ABLIT_DIR, "model-00001-of-00002.safetensors")
    print(f"\nOpening abliterated shard: {ablit_shard_path}")
    f_ablit = safetensors.torch.safe_open(ablit_shard_path, framework="pt", device="cpu")

    # 4. Process modified layers 10 through 24
    print("\nProcessing modified layers 10 through 24...")
    for layer_idx in range(10, 25):
        t_layer = time.time()
        layer_fname = f"model-layer-{layer_idx:02d}.safetensors"
        src_layer_path = os.path.join(SRC_INT4_DIR, layer_fname)
        dst_layer_path = os.path.join(DST_INT4_DIR, layer_fname)

        tensors = safetensors.torch.load_file(src_layer_path)

        # 4a. Self-attention o_proj
        o_proj_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"
        w_o = f_ablit.get_tensor(o_proj_key)
        packed_o, scale_o = quantize_group64(w_o)
        tensors[f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight_packed"] = packed_o
        tensors[f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight_scale"] = scale_o

        # 4b. MLP down_proj
        mlp_down_key = f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"
        w_mlp = f_ablit.get_tensor(mlp_down_key)
        packed_mlp, scale_mlp = quantize_group64(w_mlp)
        tensors[f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight_packed"] = packed_mlp
        tensors[f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight_scale"] = scale_mlp

        # 4c. MoE Experts down_proj across all 128 experts
        moe_down_key = f"model.language_model.layers.{layer_idx}.experts.down_proj"
        w_moe = f_ablit.get_tensor(moe_down_key)  # [128, 2816, 704]
        for e in range(128):
            w_e = w_moe[e]  # [2816, 704]
            packed_e, scale_e = quantize_group64(w_e)
            tensors[f"model.language_model.layers.{layer_idx}.experts.{e}.down_proj.weight_packed"] = packed_e
            tensors[f"model.language_model.layers.{layer_idx}.experts.{e}.down_proj.weight_scale"] = scale_e

        # Save updated layer shard
        safetensors.torch.save_file(tensors, dst_layer_path)
        dt = time.time() - t_layer
        print(f"  Layer {layer_idx:02d}: quantized o_proj, mlp.down_proj, and 128 experts down_proj -> saved ({dt:.2f}s)")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"Requantization complete in {total_time:.2f} seconds!")
    print(f"Output model ready at: {DST_INT4_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
