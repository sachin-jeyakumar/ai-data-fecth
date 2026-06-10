"""
train.py
════════
Fine-tune Qwen2.5-14B on your product extraction dataset using LoRA.

Auto-detects hardware:
  ✅ Apple Silicon (M1/M2/M3/M4) → uses MLX-LM (fastest on Mac)
  ✅ CUDA GPU                     → uses HuggingFace + PEFT + QLoRA
  ✅ CPU only                     → uses HuggingFace + PEFT (slow, but works)

Usage:
  python train.py

After training, run:
  python export_to_ollama.py
"""

import platform
import subprocess
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()
ROOT = Path(__file__).parent

# ── Load config ─────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)


def detect_hardware() -> str:
    """Returns 'mlx', 'cuda', or 'cpu'."""
    if platform.machine() == "x86_64":
        return "cpu"
    if platform.system() == "Darwin":
        # Check for Apple Silicon
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if "Apple" in result.stdout:
                return "mlx"
        except Exception:
            pass
        # Older Mac with Intel → CPU
        return "cpu"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ════════════════════════════════════════════════════════════
# MLX Training (Apple Silicon — RECOMMENDED)
# ════════════════════════════════════════════════════════════

def train_mlx():
    """Run LoRA fine-tuning via MLX-LM on Apple Silicon."""
    try:
        import mlx_lm
    except ImportError:
        console.print("[yellow]Installing mlx-lm...[/yellow]")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-lm"], check=True)

    model_id    = cfg["model"]["hf_id"]
    data_dir    = ROOT / cfg["data"]["processed_dir"]
    adapters_dir = ROOT / cfg["output"]["adapters_dir"]
    adapters_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg  = cfg["lora"]
    train_cfg = cfg["training"]

    console.print(Panel(
        f"[bold]Model:[/bold]      {model_id}\n"
        f"[bold]Data:[/bold]       {data_dir}\n"
        f"[bold]Adapters:[/bold]   {adapters_dir}\n"
        f"[bold]Iterations:[/bold] {train_cfg['iterations']}\n"
        f"[bold]Batch size:[/bold] {train_cfg['batch_size']}\n"
        f"[bold]LoRA rank:[/bold]  {lora_cfg['rank']}",
        title="🍎 MLX-LM Training (Apple Silicon)",
        border_style="green"
    ))

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",          model_id,
        "--train",
        "--data",           str(data_dir),
        "--adapter-path",   str(adapters_dir),
        "--iters",          str(train_cfg["iterations"]),
        "--batch-size",     str(train_cfg["batch_size"]),
        "--learning-rate",  str(train_cfg["learning_rate"]),
        "--lora-layers",    str(lora_cfg["rank"]),
        "--val-batches",    str(train_cfg["val_batches"]),
        "--save-every",     str(train_cfg["save_every"]),
        "--steps-per-eval", str(train_cfg["eval_every"]),
        "--mask-prompt",    # Only compute loss on assistant responses
        "--max-seq-length", str(train_cfg["max_seq_length"]),
    ]

    console.print("\n[bold cyan]Starting training...[/bold cyan]\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        console.print("\n[bold green]✅ Training complete![/bold green]")
        console.print("Run [cyan]python export_to_ollama.py[/cyan] to export to Ollama.\n")
    else:
        console.print("\n[red]❌ Training failed. Check output above.[/red]")
        sys.exit(1)


# ════════════════════════════════════════════════════════════
# HuggingFace + PEFT Training (CUDA / CPU fallback)
# ════════════════════════════════════════════════════════════

def train_hf(device: str):
    """Run QLoRA fine-tuning via HuggingFace Transformers + PEFT."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
        import torch
    except ImportError:
        console.print("[yellow]Installing HuggingFace dependencies...[/yellow]")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "transformers", "peft", "datasets", "accelerate", "bitsandbytes"
        ], check=True)
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
        import torch

    model_id     = cfg["model"]["hf_id"]
    data_dir     = ROOT / cfg["data"]["processed_dir"]
    adapters_dir = ROOT / cfg["output"]["adapters_dir"]
    adapters_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg  = cfg["lora"]
    train_cfg = cfg["training"]

    console.print(Panel(
        f"[bold]Model:[/bold]   {model_id}\n"
        f"[bold]Device:[/bold]  {device.upper()}\n"
        f"[bold]Data:[/bold]    {data_dir}",
        title=f"🤗 HuggingFace Training ({device.upper()})",
        border_style="blue"
    ))

    # Tokenizer
    console.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model with 4-bit quantization (QLoRA) for CUDA / full for CPU
    if device == "cuda":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        console.print("Loading model with 4-bit QLoRA (CUDA)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True
        )
    else:
        console.print("[yellow]CPU mode — this will be very slow for 14B model.[/yellow]")
        console.print("[yellow]Consider using a smaller model like Qwen2.5-3B-Instruct[/yellow]")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )

    # LoRA config
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Dataset
    def _format(example):
        """Format chat messages into a single training string."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(
            text, truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length"
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    valid_ds = load_dataset("json", data_files=str(data_dir / "valid.jsonl"), split="train")
    train_ds = train_ds.map(_format, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(_format, remove_columns=valid_ds.column_names)

    # Training arguments
    args = TrainingArguments(
        output_dir=str(adapters_dir),
        num_train_epochs=3,
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=1,
        warmup_steps=train_cfg["warmup_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=10,
        save_steps=train_cfg["save_every"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_every"],
        gradient_checkpointing=train_cfg["grad_checkpoint"],
        fp16=(device == "cuda"),
        report_to="none",
        load_best_model_at_end=True,
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    console.print("\n[bold cyan]Starting training...[/bold cyan]\n")
    trainer.train()
    trainer.save_model(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))

    console.print("\n[bold green]✅ Training complete![/bold green]")
    console.print("Run [cyan]python export_to_ollama.py[/cyan] to export to Ollama.\n")


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

def main():
    # Check dataset exists
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    if not (processed_dir / "train.jsonl").exists():
        console.print(
            "[red]❌ Training data not found![/red]\n"
            "Run [cyan]python prepare_dataset.py[/cyan] first."
        )
        sys.exit(1)

    hw = detect_hardware()
    console.print(f"\n[bold]🔍 Hardware detected:[/bold] [green]{hw.upper()}[/green]\n")

    if hw == "mlx":
        train_mlx()
    else:
        train_hf(hw)


if __name__ == "__main__":
    main()
