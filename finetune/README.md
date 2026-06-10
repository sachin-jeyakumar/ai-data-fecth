# 🧠 Fine-Tuning Pipeline — Terminal Guide

Train Qwen2.5-14B on YOUR brochures so it extracts data perfectly every time.

---

## How It Works

```
Your PDFs + Corrected JSON labels
         │
         ▼
  prepare_dataset.py   ← converts PDF text + JSON into training format (JSONL)
         │
         ▼
      train.py         ← fine-tunes the model using LoRA (Apple Silicon optimized)
         │
         ▼
  export_to_ollama.py  ← fuses weights → GGUF → registers as Ollama model
         │
         ▼
  ollama run ai-data-fetcher-v1   ← your custom trained model!
```

---

## Prerequisites

```bash
# 1. Ollama (for base model + serving fine-tuned model)
brew install ollama
ollama pull qwen2.5:14b

# 2. Tesseract OCR (for scanned PDFs)
brew install tesseract

# 3. Git (for llama.cpp clone during export)
brew install git

# 4. CMake (for building llama.cpp quantizer)
brew install cmake

# 5. Python dependencies
cd finetune/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Add Your Training Data

### Put PDFs in:
```
finetune/training_data/pdfs/
```

### Put matching JSON labels in:
```
finetune/training_data/labels/
```

> ⚠️ **IMPORTANT**: The label filename MUST match the PDF filename.
> Example: `samsung_catalog.pdf` → `samsung_catalog.json`

### Label JSON format (what your data analyst says the correct output should be):

```json
[
  {
    "name": "Samsung Galaxy S24",
    "model": "SM-S921B",
    "price": "₹74,999",
    "display": "6.2 inch Dynamic AMOLED 2X",
    "processor": "Snapdragon 8 Gen 3",
    "ram": "8GB",
    "storage": "256GB",
    "battery": "4000mAh",
    "camera": "50MP + 10MP + 12MP",
    "warranty": "1 year"
  },
  {
    "name": "Samsung Galaxy S24+",
    "model": "SM-S926B",
    "price": "₹99,999",
    "display": "6.7 inch Dynamic AMOLED 2X",
    ...
  }
]
```

> 💡 **Tip**: Use the base AI system (main app) to extract data first,
> then CORRECT any mistakes manually. Save that corrected JSON as the label.
> You only need 50–200 examples for great results!

---

## Step 2 — Prepare Dataset

```bash
cd finetune/
source venv/bin/activate
python prepare_dataset.py
```

**Output:**
```
✅ Found 75 PDF+label pairs

  Processing samsung_catalog.pdf...
    ✓ 12 products, 4823 chars
  Processing lg_brochure.pdf...
    ✓ 8 products, 3201 chars
  ...

┌──────────┬──────────┬──────────────────────────────────────────────┐
│ Split    │ Examples │ Path                                         │
├──────────┼──────────┼──────────────────────────────────────────────┤
│ Train    │ 64       │ training_data/processed/train.jsonl          │
│ Valid    │ 11       │ training_data/processed/valid.jsonl          │
└──────────┴──────────┴──────────────────────────────────────────────┘

✅ Dataset ready! Now run: python train.py
```

---

## Step 3 — Train

```bash
python train.py
```

The script **automatically detects your hardware**:
- 🍎 Apple Silicon (M1/M2/M3/M4) → uses **MLX-LM** (fastest)
- 🟢 NVIDIA GPU (CUDA)            → uses **HuggingFace + QLoRA**
- 🔵 CPU only                     → uses **HuggingFace** (slow)

**Example output on Apple M2/M3:**
```
🔍 Hardware detected: MLX

╭─────────────────── 🍎 MLX-LM Training ───────────────────╮
│ Model:      Qwen/Qwen2.5-14B-Instruct                     │
│ Data:       training_data/processed                       │
│ Iterations: 500                                           │
│ Batch size: 2                                             │
│ LoRA rank:  16                                            │
╰───────────────────────────────────────────────────────────╯

Starting training...

Iter   1: Train loss 2.843, Learning Rate 2.000e-06, It/Sec 0.412
Iter  10: Train loss 2.201, Learning Rate 2.000e-05, It/Sec 0.438
Iter  50: Train loss 1.104, Learning Rate 1.980e-05, It/Sec 0.451
Iter 100: Train loss 0.712, Validation loss 0.681, It/Sec 0.447
Iter 200: Train loss 0.441, Validation loss 0.423, It/Sec 0.449
Iter 500: Train loss 0.198, Validation loss 0.201, It/Sec 0.452

✅ Training complete!
```

**Time estimates:**
| Hardware | 500 iterations | 1000 iterations |
|---|---|---|
| M3 Max (32GB) | ~20 min | ~40 min |
| M2 Pro (32GB) | ~35 min | ~70 min |
| NVIDIA RTX 4090 | ~15 min | ~30 min |

---

## Step 4 — Export to Ollama

```bash
python export_to_ollama.py
```

**What it does:**
```
Step 1 — Fuse LoRA adapters into base model weights
Step 2 — Clone llama.cpp (for GGUF conversion)
Step 3 — Convert model to GGUF (f16)
Step 4 — Quantize to Q4_K_M (50% smaller, minimal quality loss)
Step 5 — Register as 'ai-data-fetcher-v1' in Ollama
```

**Output:**
```
✅ Fused model saved to: output/fused_model
✅ llama.cpp ready
✅ GGUF saved: output/gguf/model-f16.gguf
✅ Quantized model: output/gguf/model-q4_k_m.gguf (8.2 GB)

╭──────────────────────────────────────────────╮
│  🎉 Your custom model is ready!              │
│                                              │
│  Test: ollama run ai-data-fetcher-v1         │
│                                              │
│  In app, edit backend/config.py:             │
│    OLLAMA_LLM_MODEL = "ai-data-fetcher-v1"   │
╰──────────────────────────────────────────────╯
```

---

## Step 5 — Test Your Model

```bash
# Chat test
ollama run ai-data-fetcher-v1

# Or quick extraction test
echo "Extract product data from: Samsung Galaxy S24, Price ₹74999, 6.2 inch display" | \
  ollama run ai-data-fetcher-v1
```

---

## Tuning for Better Results

Edit `config.yaml`:

| Setting | Default | Higher = |
|---|---|---|
| `iterations` | 500 | Better accuracy, more time |
| `lora.rank` | 16 | More capacity, more memory |
| `training.batch_size` | 2 | Faster, more memory |
| `training.learning_rate` | 2e-5 | Adjust if loss plateaus |

**Rule of thumb:**
- 50 examples  → 300 iterations
- 100 examples → 500 iterations
- 200 examples → 800–1000 iterations

---

## Folder Structure

```
finetune/
├── config.yaml              ← Edit training hyperparameters here
├── prepare_dataset.py       ← Step 2: Prepare training data
├── train.py                 ← Step 3: Run fine-tuning
├── export_to_ollama.py      ← Step 4: Export to Ollama
├── requirements.txt
├── training_data/
│   ├── pdfs/                ← Put your PDF brochures HERE
│   ├── labels/              ← Put your corrected JSON labels HERE
│   └── processed/           ← Auto-generated (train.jsonl, valid.jsonl)
├── output/
│   ├── adapters/            ← LoRA adapter weights (saved during training)
│   ├── fused_model/         ← Merged model (after export step)
│   └── gguf/                ← Final GGUF files for Ollama
└── llama.cpp/               ← Auto-cloned during export
```
