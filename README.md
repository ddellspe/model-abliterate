# Model Abliterate (`model-abliterate`)

> **Representation Engineering & Refusal Direction Ablation for Large Language Models**  
> A generalized framework, toolchain, and mathematical runbook for isolating, auditing, and permanently removing refusal mechanisms from transformer weights without retraining, fine-tuning, or catastrophic forgetting.

---

## 1. Executive Summary & Core Theory

Standard reinforcement learning with human feedback (RLHF) and direct preference optimization (DPO) impart safety guardrails to language models. However, this frequently induces **excessive false refusals**, where benign queries regarding penetration testing, cybersecurity vulnerability research, creative writing, or technical debugging are rejected with canned boilerplate disclaimers (*"I cannot assist with that request"*).

Recent advances in mechanistic interpretability (*Arditi et al., 2024; Zou et al., 2023*) revealed that refusal behavior is governed primarily by a **low-dimensional subspace**—predominantly a single 1D direction vector $\vec{r} \in \mathbb{R}^{d_{\text{model}}}$—in the residual stream.

```
                      ┌────────────────────────────────────────┐
                      │  1. Contrastive Prompt Dataset         │
                      │  (Harmful vs. Harmless Counterparts)   │
                      └───────────────────┬────────────────────┘
                                          │ Forward pass (BF16/FP16)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  2. Harvest Residual Activations       │
                      │  (PyTorch forward hooks at each layer) │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  3. Compute Normalized Refusal Vector  │
                      │  r̂ = (μ_harm - μ_clean) / ||...||      │
                      │  Audit layer cosine similarities       │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  4. Orthogonalize Residual Projections │
                      │  W' = W(I - r̂r̂ᵀ)                       │
                      │  • Attention Out:  self_attn.o_proj    │
                      │  • Shared MLP:     mlp.down_proj       │
                      │  • MoE Experts:    experts.down_proj   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  5. Selective Low-Bit Quantization     │
                      │  (INT4 / compressed-tensors)           │
                      │  Quantize modified down-projections    │
                      │  Hardlink pre-calibrated shards/norms  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  6. Serving & Deployment               │
                      │  vLLM / SGLang / llama.cpp             │
                      └────────────────────────────────────────┘
```

By identifying this direction and **orthogonalizing the weight matrices that write into the residual stream**, the model permanently loses the ability to represent or trigger the refusal state, while preserving general reasoning, coding, and conversational capabilities.

---

## 2. Mathematical Formulation

Let $x_l \in \mathbb{R}^{d_{\text{model}}}$ represent the residual stream activation at layer $l$. Each transformer block computes updates via linear projections:

$$x_{l+1} = x_l + W_{\text{o}} h_{\text{attn}} + W_{\text{mlp\_down}} h_{\text{mlp}} + \sum_{e \in \text{active}} g_e W_{\text{expert\_down}, e} h_{e}$$

Notice that **every path writing into the residual stream** terminates in a linear transformation with output dimension $d_{\text{model}}$.

### Orthogonal Subspace Projection
Given the unit refusal vector $\hat{r} \in \mathbb{R}^{d_{\text{model}}}$ where $\|\hat{r}\|_2 = 1$:

For any matrix $W \in \mathbb{R}^{d_{\text{model}} \times d_{\text{in}}}$:

$$W' = (I - \hat{r} \hat{r}^T) W = W - \hat{r} (\hat{r}^T W)$$

* For **Dense Attention Output**: $W_{\text{o}} \leftarrow W_{\text{o}} - \hat{r} (\hat{r}^T W_{\text{o}})$
* For **Dense MLP Down-Projection**: $W_{\text{mlp\_down}} \leftarrow W_{\text{mlp\_down}} - \hat{r} (\hat{r}^T W_{\text{mlp\_down}})$
* For **Mixture-of-Experts (MoE)**: For every expert $e \in [0, E-1]$, apply the projection to each individual expert down-projection slice.

---

## 3. The 5 Phases of the Abliteration Pipeline

### Phase 1: Contrastive Prompt Engineering
Refusal vector extraction requires contrastive prompt pairs:
* **Harmful / Trigger Set ($P_{\text{harm}}$):** Prompts that reliably trigger safety refusals across your target domain (e.g., cybersecurity exploits, forensic accident analysis, political satire).
* **Harmless Set ($P_{\text{clean}}$):** Syntactically identical prompts with benign semantics (e.g., defensive patch analysis, historical governance, civil engineering load calculations).

> **Crucial Rule:** The prompt pairs must differ *only* in the refusal-triggering feature to isolate the refusal subspace without capturing generic topic or syntax features.

### Phase 2: Activation Harvesting
Pass both prompt sets through the base model in high-precision floating point (BF16 or FP16). Using PyTorch forward hooks, record the residual stream activations at the end of each transformer layer for the final prompt token.

### Phase 3: Subspace Audit & Layer Targeting
Compute the separation vector and cosine similarity for each layer $l$:

$$\vec{r}_l = \bar{h}_{\text{harm}, l} - \bar{h}_{\text{clean}, l}, \quad \hat{r}_l = \frac{\vec{r}_l}{\|\vec{r}_l\|_2}$$

$$\cos \theta_l = \frac{\langle \bar{h}_{\text{harm}, l}, \bar{h}_{\text{clean}, l} \rangle}{\|\bar{h}_{\text{harm}, l}\| \|\bar{h}_{\text{clean}, l}\|}$$

* **Shallow Layers (0 to ~25% depth):** Token identity, syntax parsing. **Do not ablate.**
* **Intermediate Layers (~30% to ~80% depth):** Intent classification, safety policy gating. Peak cosine similarity ($\cos \theta \ge 0.70$). **Primary target for ablation.**
* **Deep Layers (~85% to 100% depth):** Softcapping, vocabulary mapping, logit generation. **Do not ablate.**

### Phase 4: Weight-Space Modification
Orthogonalize target layer weights using the computed $\hat{r}_l$ vectors and export the modified safetensors.

### Phase 5: Quantization & Deployment
* **Why Abliterate First?** Abliteration **must** be performed in floating-point precision (BF16/FP16). Quantized integer weights cannot be cleanly projected without severe rounding distortion.
* **Selective Requantization:** If deploying to resource-constrained environments (e.g., low-VRAM APUs or edge hardware), only the modified down-projections need to be requantized. Untouched layers and pre-calibrated outlier scaling tables can be reused directly.
* **What Stays in BF16/FP16:** Token embeddings (`embed_tokens`), vocabulary heads (`lm_head`), normalization layers (`RMSNorm`), and vision towers must remain in high precision to avoid token collapse.

---

## 4. Repository Structure

```
model-abliterate/
├── README.md                          # Core generalized methodology & architecture guide
├── requirements.txt                   # Universal Python package dependencies
├── .gitignore                         # Tensor, checkpoint, and cache exclusions
├── scripts/
│   ├── abliterate.py                  # Core activation harvesting & weight orthogonalization tool
│   ├── quantize_abliterated.py        # Fast INT4 compressed-tensors requantization tool
│   └── run-abliterate.sh              # Kubernetes Job launcher & ConfigMap builder
├── models/                            # Model-specific runbooks, parameters & audits
│   └── gemma4-26b-a4b/
│       └── README.md                  # Detailed runbook for Google Gemma 4 (26B-A4B-it)
├── k8s/                               # Kubernetes deployment manifests
│   ├── job-abliterate.yaml            # Batch Job manifest (ROCm / Strix Halo hardware)
│   └── deployment-abliterated.yaml    # Production vLLM serving deployment
└── docs/                              # Deep technical references & cluster audits
    └── gemma4-abliteration-guide.md   # Mathematical audit & contrastive logs
```

---

## 5. Generic CLI Usage

### 1. Running Abliteration
```bash
python3 scripts/abliterate.py \
  --model-path <HUGGINGFACE_ID_OR_LOCAL_PATH> \
  --output-path ./abliterated-model \
  --layers <START_LAYER>-<END_LAYER> \
  --device cuda
```

Key options:
* `--model-path`: Hugging Face repo ID or directory containing unquantized BF16/FP16 weights.
* `--output-path`: Directory where abliterated safetensors and metadata will be saved.
* `--layers`: Layer range (e.g. `10-24`).
* `--device`: Execution device (`cuda`, `hip`, or `cpu`).
* `--dry-run`: Performs forward passes and logs layer cosine similarity statistics without writing changes.

### 2. Requantizing for Serving (INT4 / compressed-tensors)
```bash
python3 scripts/quantize_abliterated.py \
  --source-int4 <PATH_TO_CALIBRATED_BASE_INT4> \
  --source-ablit ./abliterated-model \
  --output-int4 ./abliterated-model-int4
```

---

## 6. Supported Models & Case Studies

| Model Architecture | Parameters | Target Layers | Quantization Format | Target Hardware | Dedicated Runbook |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Gemma 4 (26B-A4B-it)** | 26B (Hybrid MoE, 128 experts) | 10–24 | INT4 `compressed-tensors` group-64 | AMD Strix Halo APU (ROCm `gfx1151`) | [**Gemma 4 26B Guide**](models/gemma4-26b-a4b/README.md) |

---

## 7. References & Research

* **Arditi et al. (2024):** *Refusal in Language Models Is Mediated by a Single Direction* ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717))
* **Zou et al. (2023):** *Representation Engineering: A Top-Down Approach to AI Transparency* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405))
* **NeuralMagic:** *compressed-tensors* specification for fast GEMM kernels and group-quantized inference.
