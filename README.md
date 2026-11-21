# Model Abliterate (`model-abliterate`)

> **Representation Engineering & Refusal Direction Ablation for LLMs**  
> Specifically tested, tuned, and validated on **Google Gemma 4 (26B-A4B-it)** and AMD Strix Halo APUs (ROCm / gfx1151).

This repository preserves the mathematical pipeline, scripts, and Kubernetes manifests used to isolate and eliminate refusal directions from transformer weights without retraining, fine-tuning, or causing degradation in core reasoning and multimodal capabilities.

---

## 1. Overview & Mathematical Foundation

In modern instruction-tuned transformer architectures, refusal behavior is primarily governed by a low-dimensional subspace—often a single 1D direction vector $\vec{r} \in \mathbb{R}^{d_{\text{model}}}$—in the residual stream (*Arditi et al., 2024*).

When a prompt triggers a safety refusal, activations at intermediate layers align strongly with $\vec{r}$. Downstream layers read this component and route generation toward refusal tokens (e.g., *"I cannot fulfill this request"*).

```
                      ┌────────────────────────────────────────┐
                      │  Contrastive Prompt Pairs              │
                      │  (Harmful vs. Harmless Counterparts)   │
                      └───────────────────┬────────────────────┘
                                          │ Forward pass (BF16)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Harvest Residual Activations          │
                      │  across target layers (e.g. 10–24)     │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Compute Normalized Refusal Direction  │
                      │  r̂ = (μ_harm - μ_clean) / ||...||      │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Orthogonalize Projection Weights:     │
                      │  W' = W(I - r̂r̂ᵀ)                       │
                      │                                        │
                      │  • Attention Out:  self_attn.o_proj    │
                      │  • Shared MLP:     mlp.down_proj       │
                      │  • MoE Experts:    experts.down_proj   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Export Abliterated BF16 Safetensors   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Requantize to INT4 (compressed-tensors│
                      │  group-64 symmetric) for Strix Halo    │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Deploy to vLLM (OpenAI-compatible)    │
                      └────────────────────────────────────────┘
```

By projecting out this vector from all matrices that write into or read from the residual stream:

$$W' = W (I - \hat{r} \hat{r}^T) = W - (W \hat{r}) \hat{r}^T$$

the model permanently loses the ability to trigger or sustain the refusal state, enabling uninhibited assistance for authorized cybersecurity research, vulnerability forensics, penetration testing, creative writing, and sensitive data analysis.

---

## 2. Repository Structure

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

## 3. Workflow & Usage

### A. Environment Setup

Install prerequisites:
```bash
pip install -r requirements.txt
```

### B. Running Abliteration (BF16 Orthogonalization)

Execute `scripts/abliterate.py` directly against the base model weights:

```bash
python3 scripts/abliterate.py \
  --model-path google/gemma-4-26B-A4B-it \
  --output-path ./gemma-4-26b-abliterated \
  --layers 10-24 \
  --device cuda
```

Key arguments:
* `--model-path`: Hugging Face model identifier or local snapshot directory.
* `--output-path`: Directory where modified safetensors and configs are saved.
* `--layers`: Layer range to ablate (empirically determined as `10-24` for Gemma 4 26B).
* `--device`: Execution device (`cuda` for ROCm/CUDA, or `cpu`).
* `--dry-run`: Computes refusal vectors and logs cosine similarities without altering weights.

### C. INT4 Requantization (for AMD Strix Halo / Low VRAM)

To run the abliterated model within ~15.12 GiB VRAM on AMD Strix Halo (gfx1151 APUs), requantize the abliterated layers using `compressed-tensors` with group-size 64 symmetric quantization:

```bash
python3 scripts/quantize_abliterated.py \
  --source-int4 /root/.cache/huggingface/hub/models--jcbtc--Gemma4-26B-CIRU-AMD-STRIX-vLLM/snapshots/<hash> \
  --source-ablit /root/.cache/huggingface/gemma-4-26b-abliterated \
  --output-int4 /root/.cache/huggingface/gemma-4-26b-abliterated-int4
```

This step:
1. Hardlinks or copies untouched shards (layers 0–9, 25–29, vision encoders, embeddings).
2. Quantizes modified layers 10–24 (`self_attn.o_proj`, `mlp.down_proj`, and all 128 `experts.down_proj`).
3. Preserves all multimodal configs and tokenizers.
4. Executes in under 60 seconds on Zen 5 / Strix Halo.

---

## 4. Kubernetes In-Cluster Execution

For clusters with dedicated ROCm/GPU worker nodes:

1. **Submit the Abliteration Job:**
   ```bash
   ./scripts/run-abliterate.sh
   ```
   This creates a ConfigMap containing `abliterate.py`, cleans up prior jobs, and launches `k8s/job-abliterate.yaml` on the GPU node.

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

## 5. Verification & Testing

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

## 6. Target Layer Details for Gemma 4 (26B-A4B-it)

Activation audit results across layers:
* **Layers 0–9 (Shallow):** Low refusal cosine similarity ($\sim 0.05$–$0.20$); mostly syntax, token identity, and low-level parsing.
* **Layers 10–24 (Intermediate - Core Target):** Peak refusal alignment ($\cos \theta \ge 0.70$); semantic intent classification and refusal trigger routing occur here.
* **Layers 25–29 (Deep):** Output formatting and token projection.

---

## 7. References & Research

* Arditi et al. (2024): *Refusal in Language Models Is Mediated by a Single Direction* ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717))
* Zou et al. (2023): *Representation Engineering: A Top-Down Approach to AI Transparency* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405))
* NeuralMagic: *compressed-tensors* specification for fast INT4 GEMM kernels.
