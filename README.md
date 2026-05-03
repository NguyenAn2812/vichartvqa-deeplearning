# ChartVQA — Vietnamese Chart Visual Question Answering

> **Multi-task VQA system** for Vietnamese charts, simultaneously performing chart classification and generative answer generation.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Model Architectures](#3-model-architectures)
   - [A1 — LSTM Decoder](#a1--lstm-decoder)
   - [A2 — Transformer Decoder](#a2--transformer-decoder)
   - [B1 — Zero-shot Qwen2-VL](#b1--zero-shot-qwen2-vl)
   - [B2 — QLoRA Fine-tuning (Qwen2-VL)](#b2--qlora-fine-tuning-qwen2-vl)
4. [Architecture Diagrams](#4-architecture-diagrams)
5. [Training Methods](#5-training-methods)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Results](#7-results)
8. [Environment Setup](#8-environment-setup)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Project Overview

This project tackles **Vietnamese Chart Visual Question Answering (ChartVQA)** — a challenging multimodal task requiring the model to:

- **Understand a chart image** (bar chart, line chart, pie chart, etc.)
- **Comprehend a Vietnamese question** about that chart
- **Generate a natural language answer** in Vietnamese

Two complementary approaches are explored:

| Config | Notebook | Strategy |
|--------|----------|----------|
| **A1** | `Train_A1_A2.ipynb` | Custom encoder-fusion-decoder (LSTM) trained from scratch |
| **A2** | `Train_A1_A2.ipynb` | Custom encoder-fusion-decoder (Transformer) trained from scratch |
| **B1** | `B1_Model_with_checkpoint.ipynb` | Zero-shot inference with `Qwen2-VL-2B-Instruct` — no fine-tuning |
| **B2** | `Train_B2.ipynb` | QLoRA fine-tuning of pretrained `Qwen2-VL-2B-Instruct` |

---

## 2. Dataset

**Source:** [`Zenng2812/vqa-vietnamese-charts`](https://huggingface.co/datasets/Zenng2812/vqa-vietnamese-charts) (Hugging Face)

> 📦 **This dataset was created by our team** as part of this project. It is an AI-assisted pipeline that automatically generates chart images, Vietnamese analytical questions, and trend-descriptive answers — covering 4 chart types: Bar, Line, Area, and Pie. The dataset is publicly available on HuggingFace Hub and was split 80/10/10 by chart (not by QA pair) to prevent data leakage.
>
> 📄 For full details on how the dataset was built — including the generation pipeline, LLM prompting strategy, normalization, and export steps — see the dedicated documentation **[here](dataset/README.md)**.

Each sample contains:
- `image` — A chart image (PIL format)
- `question` — A Vietnamese question about the chart
- `answer` — The ground-truth Vietnamese answer
- `chart_type` — Chart category label (`bar`, `line`, `area`, `pie`)

| Split | Ratio | Note |
|-------|-------|------|
| Train | 80% | Used for vocabulary building & training |
| Validation | 10% | Used for checkpoint selection |
| Test | 10% | Used for final evaluation |

> **Answer vocabulary (A1/A2):** Built from the top-1000 most frequent words in the training answers, plus 4 special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.

---

## 3. Model Architectures

### A1 / A2 — Custom Encoder–Fusion–Decoder

Both A1 and A2 share the same **Encoder** and **Fusion** components. They differ only in the **Decoder**.

#### Encoders

| Component | Model | Output |
|-----------|-------|--------|
| **Text Encoder** | [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base) | `(B, L, 768)` — hidden states for all tokens |
| **Image Encoder** | [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) | `(B, 197, 768)` — patch embeddings; `[CLS]` token at position 0 used for classification |

#### Classifier Head

Operates on the ViT `[CLS]` token to predict chart type:

```
Linear(768 → 256) → ReLU → Dropout(0.1) → Linear(256 → num_classes)
```

#### Fusion — CoAttention

A **scaled dot-product cross-attention** module where:
- **Query** = text features
- **Key / Value** = image features
- A **chart-type embedding** (`type_emb`) is prepended to the text sequence before fusion, conditioning the decoder on chart type context.

```
Q = W_q(text_feats)
K = W_k(img_feats)
V = W_v(img_feats)
attn  = softmax(Q·Kᵀ / √d)
fused = LayerNorm(attn·V + text_feats)   ← residual connection
```

#### Decoders

| Config | Decoder | Details |
|--------|---------|---------|
| **A1** | `nn.LSTM` | Hidden state `h₀` initialized from `mean_pool(fused_features)`; auto-regressive token generation |
| **A2** | `nn.TransformerDecoder` | 3 layers, 8 attention heads, causal mask to prevent future token leakage |

---

### B1 — Zero-shot Qwen2-VL

B1 uses the **same base model as B2** (`Qwen2-VL-2B-Instruct`) but performs **pure zero-shot inference** — no training, no fine-tuning, no LoRA. This serves as the baseline to measure how much fine-tuning (B2) actually gains over the out-of-the-box model.

#### Base Model
[`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) — loaded in `bfloat16`, `device_map="auto"`.

#### Local Caching Strategy
To avoid re-downloading the model on every run, B1 implements a simple checkpoint-to-Drive strategy:
- **First run:** downloads from HuggingFace and saves to `LOCAL_MODEL_DIR` on Google Drive.
- **Subsequent runs:** loads directly from local — no internet required.

```python
LOCAL_MODEL_DIR = "/content/drive/MyDrive/B1_checkpoint/Qwen2VL"
```

#### Prompt Format

```
User: <image>
      Dựa vào biểu đồ, hãy trả lời câu hỏi sau một cách chính xác
      và ngắn gọn nhất (dưới 10 từ) bằng tiếng Việt: {question}
```

No system prompt — single-turn, greedy decoding (`max_new_tokens=15`).

---

### B2 — QLoRA Fine-tuning (Qwen2-VL)

Rather than building a model from scratch, B2 **adapts a powerful pretrained Vision-Language Model** using parameter-efficient fine-tuning.

#### Base Model
[`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) — a 2B-parameter multimodal LLM with native vision understanding.

#### Quantization — 4-bit QLoRA
The model is loaded in **4-bit NF4 quantization** via `bitsandbytes`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

This reduces VRAM consumption dramatically, enabling fine-tuning on a **T4 GPU (16 GB)**.

#### LoRA Adapters
Low-Rank Adaptation (LoRA) is applied to attention and MLP projection layers:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

Only **LoRA adapter weights** are trained — the base model weights remain frozen.

#### Prompt Format
A system prompt is prepended to every sample to steer the model toward chart-analysis behavior:

```
System: Bạn là chuyên gia phân tích biểu đồ. Quan sát biểu đồ và trả lời câu hỏi
        về xu hướng, so sánh, hoặc nhận xét tổng quan. Trả lời bằng tiếng Việt,
        ngắn gọn, không cần trích xuất số liệu cụ thể.
User:   <image> + <question>
Assistant: <answer>
```

During training, **only the assistant tokens** contribute to the loss — prompt tokens are masked with `-100`.

---

## 4. Architecture Diagrams

### A1 / A2 Pipeline

```
Input Image ──► ViT-Base/16 ──────────────────► [CLS] token ──► Classifier ──► chart_type_logits
                     │                                                │
                     │ patch embeddings (197, 768)                   │ type_emb
                     │                                               ▼
Input Question ──► PhoBERT-base ──► text hidden states ──► [type_emb | text] ──► CoAttention ──► fused_features
                                                                                                      │
                                                           ┌──────────────────────────────────────────┘
                                                           │
                                              A1: LSTM Decoder
                                                  h₀ = mean_pool(fused_features)
                                                  Generates tokens auto-regressively
                                                           │
                                              A2: Transformer Decoder (3L, 8H)
                                                  memory = fused_features
                                                  Causal mask prevents future peeking
                                                           │
                                                           ▼
                                                   fc_out (Linear → vocab_size)
                                                           │
                                                           ▼
                                                   Generated Answer (Vietnamese)
```

### B1 Pipeline

```
Input Image ───────────────────────────────────────────┐
                                                        ▼
Zero-shot Prompt + Question ──► Tokenizer + Processor ──► Qwen2-VL-2B-Instruct
                                                           (bfloat16, no adapters)
                                                           │
                                                           ▼
                                                    Generated Answer
                                                    (greedy, max_new_tokens=15)
```

### B2 Pipeline

```
Input Image ──────────────────────────────────────────────────────────────────┐
                                                                               ▼
System Prompt + Question ──► Chat Template ──► Tokenizer + ViT Processor ──► Qwen2-VL-2B
                                                                               (4-bit NF4)
                                                                               + LoRA adapters
                                                                               (r=16, α=32)
                                                                               │
                                                                               ▼
                                                                        Generated Answer
                                                                        (auto-regressive,
                                                                         do_sample=False)
```

---

## 5. Training Methods

### A1 / A2 — Two-Phase Training

#### Phase 1: Warm-up Classifier (2 epochs)
- **Decoder is frozen** — only Encoder + Classifier Head are trained.
- Loss: `CrossEntropyLoss` on chart-type classification.
- Goal: give the image encoder a strong classification signal before the full multi-task loop.
- Checkpoints saved to `phase1_latest.pth` for resume support.

#### Phase 2: Joint Training (10 epochs)
- **All components trained simultaneously** (Encoder + Fusion + Classifier + Decoder).
- **Combined loss:**

  ```
  loss = loss_VQA + λ_cls × loss_CLS
  ```

  Where `λ_cls = 0.3` (classifier is a regularizer, not the primary objective).

- `loss_VQA` uses **Label Smoothing (ε = 0.1)** and ignores `<PAD>` tokens.
- **Teacher forcing**: ground-truth answer tokens are fed as decoder input during training.
- Optimizer: `AdamW`, lr = `2e-5`.
- Best model saved in **FP16** to halve checkpoint size.
- **Score used for model selection** (composite):

  ```
  score = 0.15×EM + 0.20×Soft + 0.15×BLEU + 0.20×ROUGE-L + 0.30×BERTScore
  ```

> **A2 note:** The Encoder/Classifier weights from Phase 1 (trained with A1) are reused via `load_state_dict(..., strict=False)`, so A2 only needs to learn the new Transformer Decoder.

---

### B2 — QLoRA Fine-tuning

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Learning rate | `1e-5` |
| LR scheduler | Cosine |
| Warmup steps | 30 |
| Optimizer | `paged_adamw_8bit` |
| Gradient checkpointing | ✅ Enabled |
| Max grad norm | 0.3 |
| Evaluation & save every | 200 steps |
| Best model selection | Lowest `eval_loss` |
| Mixed precision | Disabled (T4 + NF4 handles memory) |

Training is handled by Hugging Face `Trainer` with a custom `TrainingLogCallback` that saves logs to both CSV and JSON at every step.

---

## 6. Evaluation Metrics

All three configurations are evaluated with the same metric suite:

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1 if prediction == ground truth (after normalization), else 0 |
| **Soft Accuracy / Token F1** | Word-overlap F1 between predicted and reference tokens |
| **BLEU** | BLEU-2 with NLTK smoothing (`method1`) |
| **ROUGE-L** | Longest Common Subsequence F1 |
| **BERTScore F1** | Semantic similarity via contextual embeddings (lang=`vi` / `xlm-roberta-base`) |

Text normalization for B2 includes lowercasing, whitespace normalization, and removal of non-alphanumeric/non-Vietnamese characters.

---

## 7. Results

> ⚠️ **Results to be filled in after experiments complete.**

### Overall Test Set Performance

| Config | EM | Soft Acc | BLEU | ROUGE-L | BERTScore F1 |
|--------|----|----------|------|---------|--------------|
| A1 (LSTM) | — | — | — | — | — |
| A2 (Transformer) | — | — | — | — | — |
| B1 (Zero-shot Qwen2-VL) | — | — | — | — | — |
| B2 (QLoRA Qwen2-VL) | — | — | — | — | — |

### Per-Chart-Type Breakdown (B2)

> *(Will be added from `metrics_by_chart_type.csv`)*

### A1 vs A2 vs B2 — Comparison

> *(Comparison table and analysis will be added by the author.)*

---

## 8. Environment Setup

### Requirements

```bash
# For A1 / A2
pip install datasets evaluate bert_score rouge-score nltk
pip install torch transformers

# For B1 / B2 (additional)
pip install transformers accelerate peft bitsandbytes qwen-vl-utils
```

### Hardware

| Config | Recommended GPU | VRAM |
|--------|----------------|------|
| A1 / A2 | Any CUDA GPU | ≥ 8 GB |
| B1 | Any CUDA GPU | ≥ 8 GB (inference only) |
| B2 | NVIDIA T4 (or better) | ≥ 16 GB |

### Google Drive Setup

All notebooks assume a mounted Google Drive. Paths:

```python
# A1 / A2
PROJECT_PATH = "/content/drive/MyDrive/VQA_Chart_Project/v2"

# B1
LOCAL_MODEL_DIR = "/content/drive/MyDrive/B1_checkpoint/Qwen2VL"

# B2
DRIVE_DIR = "/content/drive/MyDrive/B2_Qwen2VL_VQA_Charts"
```

Ensure the above directories exist, or they will be created automatically on first run.

---

## 9. How to Run

### B1 (`B1_Model_with_checkpoint.ipynb`)

Open the notebook in **Google Colab** and run cells sequentially:

```
Step 0 → Mount Google Drive          (Cell 0: drive.mount)
Step 1 → Install libraries           (Cell 1: pip install ...)
Step 2 → Imports & path config       (Cell 2: LOCAL_MODEL_DIR, HF_MODEL_ID)
Step 3 → Download & cache model      (Cell 3: auto-skipped if already cached)
Step 4 → Load model from local       (Cell 4: from_pretrained(LOCAL_MODEL_DIR))
Step 5 → Load dataset                (Cell 5: load_dataset)
Step 6 → Define inference function   (Cell 6: get_zero_shot_prediction)
Step 7 → Run demo on samples         (Cell 7: loop over NUM_DEMO samples)
```

> **Cell 3 is safe to re-run** — it checks `os.path.exists(LOCAL_MODEL_DIR)` before downloading and skips automatically if the model is already cached on Drive.

---

### A1 / A2 (`Train_A1_A2.ipynb`)

Open the notebook in **Google Colab** and run cells sequentially:

```
Step 1  → Install libraries          (Cell: pip install ...)
Step 2  → Mount Google Drive         (Cell: drive.mount)
Step 3  → Check GPU                  (Cell: torch.cuda.get_device_name)
Step 4  → Import libraries           (Cell: import ...)
Step 5  → Configure paths & params   (Cell: class Config)
Step 6  → Load dataset               (Cell: load_dataset)
Step 7  → Build answer vocabulary    (Cell: AnswerVocab)
Step 8  → Preprocess & build loaders (Cell: DataLoader)
Step 9  → Define model               (Cell: ChartVQAModel)
Step 10 → Define loss & optimizer    (Cell: criterion_cls / criterion_vqa)
Step 11 → Run Phase 1 warm-up        (Cell: train_phase1)

# --- Choose config ---

# To train A1 (LSTM decoder):
Step 12a → Run Cell: "▶️ Chạy Cấu hình A1"
           model_a1 → train_phase1 → train_joint(tag="A1")

# To train A2 (Transformer decoder):
Step 12b → Run Cell: "▶️ Chạy Cấu hình A2"
           model_a2 → loads Phase 1 weights → train_joint(tag="A2")

Step 13 → Evaluate on test set       (Cell: evaluate_soft_metrics)
Step 14 → Visualize predictions      (Cell: predict_and_show)
```

**Resume from checkpoint:** The training functions automatically detect and resume from the latest saved checkpoint — just re-run the training cell.

---

### B2 (`Train_B2.ipynb`)

Open the notebook in **Google Colab (T4 GPU)** and run cells sequentially:

```
Step 1 → Install libraries           (Cell 1: pip install ...)
Step 2 → Mount Drive & set paths     (Cell 2: drive.mount + constants)
Step 3 → Load dataset & EDA          (Cell 3: load_dataset + plots)
Step 4 → Load Qwen2-VL + LoRA        (Cell 4: BitsAndBytesConfig + LoraConfig)
Step 5 → Build data collator         (Cell 5: QwenVQACollator)
Step 6 → Configure training args     (Cell 6: TrainingArguments)
Step 7 → Train                       (Cell 7: trainer.train())
          ↳ Auto-resumes from latest checkpoint if present
Step 8 → Run inference samples       (Cell 8: show_prediction)
Step 9 → Full evaluation             (Cell 9: evaluate_model(test_ds))
Step 10 → Per-chart-type breakdown   (Cell 10: chart_report)
Step 11 → Save summary               (Cell 11: summary.json)
```

**Outputs saved to Google Drive:**

```
B2_Qwen2VL_VQA_Charts/
├── checkpoints/          # Intermediate checkpoints (keep last 3)
├── logs/
│   ├── training_log.csv  # Step-by-step loss & lr log
│   └── training_log.json
├── final_model/          # Saved model + processor weights
└── results/
    ├── predictions.csv            # Per-sample predictions
    ├── metrics.json               # Aggregate metrics
    ├── metrics_by_chart_type.csv  # Per-chart-type breakdown
    └── metrics_by_chart_type.png  # Bar chart visualization
```

---

## 10. Project Structure

```
.
├── Train_A1_A2.ipynb              # Custom encoder–fusion–decoder (A1: LSTM, A2: Transformer)
├── B1_Model_with_checkpoint.ipynb # Zero-shot inference with Qwen2-VL-2B-Instruct
├── Train_B2.ipynb                 # QLoRA fine-tuning of Qwen2-VL-2B-Instruct (B2)
└── README.md                      # This file

# Generated at runtime (Google Drive):
VQA_Chart_Project/v2/
├── checkpoints/
│   ├── phase1_latest.pth
│   ├── phase1_epoch_1.pth
│   ├── a1_epoch_N.pth
│   ├── a2_epoch_N.pth
│   ├── history_a1.json
│   └── history_a2.json
├── best_model_a1.pth     # Best A1 weights (FP16)
├── best_model_a2.pth     # Best A2 weights (FP16)
├── last_checkpoint_a1.pth
└── last_checkpoint_a2.pth
```

---

## 11. Configuration Reference

### A1 / A2 (`Config` class)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEED` | 42 | Random seed |
| `BATCH_SIZE` | 16 | Training batch size |
| `MAX_LENGTH` | 64 | Max question token length |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `LAMBDA_CLS` | 0.3 | Weight of classification loss in joint training |
| `LABEL_SMOOTHING` | 0.1 | Smoothing for VQA cross-entropy loss |
| `NUM_EPOCHS_WARMUP` | 2 | Phase 1 (classifier warm-up) epochs |
| `NUM_EPOCHS_JOINT` | 10 | Phase 2 (joint training) epochs |
| `MODEL_NAME_TEXT` | `vinai/phobert-base` | Text encoder |
| `MODEL_NAME_IMG` | `google/vit-base-patch16-224-in21k` | Image encoder |

### B2 (`TrainingArguments`)

| Parameter | Value |
|-----------|-------|
| `num_train_epochs` | 2 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | 1e-5 |
| `lr_scheduler_type` | cosine |
| `warmup_steps` | 30 |
| `optim` | paged_adamw_8bit |
| `max_grad_norm` | 0.3 |
| `save_total_limit` | 3 |
| `metric_for_best_model` | eval_loss |

---

## Notes

- **B1** is a pure **zero-shot baseline** — the model receives no task-specific training. It uses a single-turn prompt with `max_new_tokens=15` and greedy decoding. Comparing B1 vs B2 directly shows the gain from QLoRA fine-tuning on this dataset.
- **A1 / A2** use **teacher forcing** during training but **auto-regressive greedy decoding** during inference.
- **B2** uses **greedy decoding** (`do_sample=False`) at inference for reproducibility.
- For A2, the Encoder/Classifier weights are **transferred from A1's Phase 1** checkpoint (`strict=False`), so only the Transformer Decoder learns from scratch in Phase 2.
- BERTScore for B2 uses `xlm-roberta-base` instead of a Vietnamese-specific model for better stability on T4.
- Best model weights for A1/A2 are saved in **FP16** (half precision) to reduce checkpoint file size.