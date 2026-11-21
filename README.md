# Model Abliterate (`model-abliterate`)

> **Representation Engineering & Refusal Direction Ablation for LLMs**  
> Specifically tested, tuned, and validated on **Google Gemma 4 (26B-A4B-it)** and AMD Strix Halo APUs (ROCm / gfx1151).

This repository preserves the mathematical pipeline, scripts, and Kubernetes manifests used to isolate and eliminate refusal directions from transformer weights without retraining, fine-tuning, or causing degradation in core reasoning and multimodal capabilities.

---

## 1. Overview & Two-Stage Architecture

The pipeline uses a **two-stage architecture** combining high-precision weight orthogonalization with calibrated low-bit serving:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Representation Engineering (BF16 Base)                                        │
│ Source: google/gemma-4-26B-A4B-it (Official Unquantized Base Weights)                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  • High-precision BF16 forward pass over contrastive prompt pairs                      │
│  • Harvest residual stream activations across intermediate layers (10–24)              │
│  • Compute normalized refusal direction: r̂ = (μ_harm - μ_clean) / ||μ_harm - μ_clean|| │
│  • Orthogonalize projection weights: W' = W(I - r̂r̂ᵀ)                                    │
│    - Self-attention output:  self_attn.o_proj                                          │
│    - Shared MLP down-proj:   mlp.down_proj                                             │
│    - Sparse MoE down-proj:   128 experts.down_proj                                     │
│  • Export unquantized abliterated BF16 model (~48 GiB)                                 │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Calibrated INT4 Requantization (Strix Halo APU Optimized)                     │
│ Template: jcbtc/Gemma4-26B-CIRU-AMD-STRIX-vLLM                                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  • Hardlink untouched pre-calibrated shards (layers 0–9, 25–29, vision, embeddings)    │
│  • Requantize only modified down-projections in layers 10–24 via compressed-tensors   │
│  • Symmetric group-size 64 INT4 packing (preserves pre-calibrated Q/K/V & gate/up)     │
│  • Packaging completed in ~48s; exports 14.58 GiB model ready for vLLM                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why Start with the Google Base Model?
Abliteration cannot be performed directly on quantized integer weights (e.g. INT4). Removing a low-dimensional subspace via orthogonal projection ($W' = W(I - \hat{r}\hat{r}^T)$) relies on continuous, high-precision floating-point representations (bfloat16). Performing projection on pre-quantized tensors causes severe rounding artifacts, destroys calibration, and induces catastrophic reasoning degradation.

---

## 2. Quantization Design: Can We Quantize the Whole Model?

A common question when optimizing large models for unified memory APUs is whether the entire model can or should be quantized to INT4:

### What CAN Be Quantized to INT4
All **30 language transformer blocks** (layers 0 through 29) can run in 4-bit precision:
* **Attention projections:** `q_proj`, `k_proj`, `v_proj`, and `o_proj`
* **Shared MLP projections:** `gate_proj`, `up_proj`, and `down_proj`
* **Sparse MoE experts:** all 128 `experts.*.gate_up_proj` and `experts.*.down_proj` per layer

### What MUST Remain in BF16 / FP16
Even in an INT4 model, non-linear and embedding modules are deliberately retained in high precision:
* **Token Embeddings (`embed_tokens`) & Language Model Head (`lm_head`):** Quantizing the 256,000+ token vocabulary projection to 4 bits results in severe token collapse and loss of generation coherence (`model-other.safetensors`).
* **Normalization Layers (`RMSNorm` / `LayerNorm`):** Scale and bias parameters must remain in FP32/BF16 to prevent catastrophic numerical drift across 30 layers.
* **Multimodal Vision Tower (`model-vision.safetensors`):** The SigLIP visual encoder is kept in BF16 so visual reasoning and OCR precision are not degraded.

### Why Selective / Hybrid Requantization Was Used
Rather than re-quantizing all 30 layers from scratch using naive Round-To-Nearest (RTN):
1. **Down-projections only:** Abliteration only modifies matrices that project *into* the residual stream (`o_proj`, `mlp.down_proj`, and `experts.down_proj`). The attention input projections (`q/k/v`) and MoE gate/up projections were completely untouched.
2. **Preserving Pre-Calibrated CIRU Weights:** The `jcbtc` base model underwent CIRU calibration (optimizing outlier scales to prevent quantization distortion). By reusing the untouched shards for layers 0–9, 25–29, and the untouched input projections in layers 10–24, we preserve the original calibration on the rest of the network.
3. **Turnaround Speed:** Quantizing only the abliterated projections completed in **48.78 seconds**, compared to hours required for full model recalibration.

---

## 3. Repository Structure

```
model-abliterate/
├── README.md                          # Documentation and runbook
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Tensor and cache exclusions
├── scripts/
│   ├── abliterate.py                  # Activation extraction, vector calculation & weight orthogonalization
│   ├── quantize_abliterated.py        # INT4 compressed-tensors requantization pipeline
│   └── run-abliterate.sh              # Kubernetes Job launcher & ConfigMap builder
├── k8s/
│   ├── job-abliterate.yaml            # K8s batch Job spec (ROCm / Strix Halo hardware)
│   └── deployment-abliterated.yaml    # vLLM serving Deployment for the abliterated INT4 model
└── docs/
    └── gemma4-abliteration-guide.md   # Mathematical guide, layer cosine similarity audits & prompts
```

---

## 4. Workflow & Usage

### A. Environment Setup

Install prerequisites:
```bash
pip install -r requirements.txt
```

### B. Stage 1: Running Abliteration (BF16 Orthogonalization)

Execute `scripts/abliterate.py` directly against the official Google BF16 base snapshot:

```bash
python3 scripts/abliterate.py \
  --model-path google/gemma-4-26B-A4B-it \
  --output-path ./gemma-4-26b-abliterated \
  --layers 10-24 \
  --device cuda
```

Key arguments:
* `--model-path`: Hugging Face model identifier or local snapshot directory of the base BF16 weights.
* `--output-path`: Directory where modified safetensors and configs are saved.
* `--layers`: Layer range to ablate (empirically determined as `10-24` for Gemma 4 26B).
* `--device`: Execution device (`cuda` for ROCm/CUDA, or `cpu`).
* `--dry-run`: Computes refusal vectors and logs cosine similarities without altering weights.

### C. Stage 2: Calibrated INT4 Requantization

To run the abliterated model within ~15.12 GiB VRAM on AMD Strix Halo (gfx1151 APUs), requantize the modified layers using `compressed-tensors` group-size 64 symmetric packing.

#### 1. Locating / Acquiring the Base Template
The quantization script uses `jcbtc/Gemma4-26B-CIRU-AMD-STRIX-vLLM` as the calibrated structure template. If running in a fresh standalone environment, download the base weights:
```bash
huggingface-cli download jcbtc/Gemma4-26B-CIRU-AMD-STRIX-vLLM
```
*(On the `distiller` cluster node, this snapshot is already cached in the shared volume at `models--jcbtc--Gemma4-26B-CIRU-AMD-STRIX-vLLM/snapshots/fdf479770466055d7aa5de127435626a5ad91e34`.)*

#### 2. Running Requantization
Execute `scripts/quantize_abliterated.py`. When run with no arguments, it automatically defaults to the cluster cache paths:

```bash
# Explicit arguments:
python3 scripts/quantize_abliterated.py \
  --source-int4 /root/.cache/huggingface/hub/models--jcbtc--Gemma4-26B-CIRU-AMD-STRIX-vLLM/snapshots/fdf479770466055d7aa5de127435626a5ad91e34 \
  --source-ablit ./gemma-4-26b-abliterated \
  --output-int4 /root/.cache/huggingface/gemma-4-26b-abliterated-int4
```

| Argument | Description | Default Path |
| :--- | :--- | :--- |
| `--source-int4` | Base calibrated Strix Halo INT4 model directory | Cluster snapshot (`fdf4797...`) |
| `--source-ablit` | Path to Stage 1 abliterated BF16 model directory | `/root/.cache/huggingface/gemma-4-26b-abliterated` |
| `--output-int4` | Destination directory for deployable INT4 model | `/root/.cache/huggingface/gemma-4-26b-abliterated-int4` |

#### 3. Execution Characteristics & Mechanics
* **CPU-Bound & Fast:** Runs entirely on CPU using PyTorch and C-extensions. It does not occupy GPU VRAM and completes all 15 modified layers (1,950 projections across 128 MoE experts) in **~48 seconds** on Zen 5 cores.
* **Internal Process:**
  1. **Config & Tokenizer Sync:** Copies `config.json`, `generation_config.json`, `processor_config.json`, tokenizers, and `chat_template.jinja`.
  2. **Untouched Shard Hardlinking:** Hardlinks or copies shards for layers 0–9, 25–29, `model-other.safetensors` (embeddings & `lm_head`), and `model-vision.safetensors` (SigLIP vision tower).
  3. **Target Layer Requantization (Layers 10–24):** Reads BF16 weights for `self_attn.o_proj`, `mlp.down_proj`, and all 128 `experts.*.down_proj`, quantizes them to signed 4-bit $[-8, 7]$ with group-size 64 scales, packs them into `int32` words, and writes the updated layer shards (`model-layer-10.safetensors` through `model-layer-24.safetensors`).
* **Footprint:** Drops disk usage from **48.07 GiB** (BF16) down to **14.58 GiB** (INT4), leaving over 100 GiB of unified memory free on Strix Halo for long context KV caching and co-located workloads.

---

## 5. Kubernetes In-Cluster Execution

For clusters with dedicated ROCm/GPU worker nodes:

1. **Submit the Automated Job:**
   ```bash
   ./scripts/run-abliterate.sh
   ```
   This packages both `abliterate.py` and `quantize_abliterated.py` into a Kubernetes ConfigMap (`gemma-abliterate-script`), cleans up prior jobs, and launches `k8s/job-abliterate.yaml` on the GPU worker node.
   * **Stage 1 (BF16 Abliteration)** runs on the GPU/APU in eager mode (~64s).
   * **Stage 2 (INT4 Requantization)** runs immediately after on the Zen 5 CPU (~48s).

2. **Monitor execution:**
   ```bash
   kubectl logs -n llm job/gemma-abliterate -f
   ```

3. **Deploy the Serving Container:**
   Deploy `k8s/deployment-abliterated.yaml` to spin up vLLM pointing to the new model:
   ```bash
   kubectl apply -f k8s/deployment-abliterated.yaml
   ```

---

## 6. Verification & Testing

Verify that refusal responses have been eliminated:

```bash
curl -s http://<NODE_IP>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26b-a4b-it",
    "messages": [
      {"role": "user", "content": "Explain how penetration testers perform vulnerability scanning and port enumeration."}
    ]
  }'
```

---

## 7. Target Layer Details for Gemma 4 (26B-A4B-it)

Activation audit results across layers:
* **Layers 0–9 (Shallow):** Low refusal cosine similarity ($\sim 0.05$–$0.20$); mostly syntax, token identity, and low-level parsing.
* **Layers 10–24 (Intermediate - Core Target):** Peak refusal alignment ($\cos \theta \ge 0.70$); semantic intent classification and refusal trigger routing occur here.
* **Layers 25–29 (Deep):** Output formatting and token projection.

---

## 8. References & Research

* Arditi et al. (2024): *Refusal in Language Models Is Mediated by a Single Direction* ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717))
* Zou et al. (2023): *Representation Engineering: A Top-Down Approach to AI Transparency* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405))
* NeuralMagic: *compressed-tensors* specification for fast INT4 GEMM kernels.
