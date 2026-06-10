"""
export_to_ollama.py
═══════════════════
After training completes, this script:

  Step 1 — Fuse LoRA adapters back into the base model
  Step 2 — Convert fused model to GGUF format (via llama.cpp)
  Step 3 — Quantize to Q4_K_M (smaller + faster, barely any quality loss)
  Step 4 — Create Ollama Modelfile
  Step 5 — Register model into Ollama

Run:
  python export_to_ollama.py

After it finishes, your custom model is ready:
  ollama run ai-data-fetcher-v1
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
ROOT = Path(__file__).parent

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_ID      = cfg["model"]["hf_id"]
OLLAMA_NAME   = cfg["model"]["ollama_name"]
ADAPTERS_DIR  = ROOT / cfg["output"]["adapters_dir"]
FUSED_DIR     = ROOT / cfg["output"]["fused_dir"]
GGUF_DIR      = ROOT / cfg["output"]["gguf_dir"]
SYSTEM_PROMPT = cfg["system_prompt"].strip()

LLAMA_CPP_DIR = ROOT / "llama.cpp"   # cloned here automatically


# ── Helpers ──────────────────────────────────────────────────

def run(cmd: list, cwd=None, desc=""):
    console.print(f"\n[dim]$ {' '.join(str(c) for c in cmd)}[/dim]")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        console.print(f"[red]❌ Failed: {desc}[/red]")
        sys.exit(1)


def check_adapters():
    """Make sure training has been run."""
    # MLX saves adapters.npz; HF saves adapter_model.bin or adapter_model.safetensors
    mlx_adapter = ADAPTERS_DIR / "adapters.npz"
    hf_adapter  = ADAPTERS_DIR / "adapter_model.safetensors"
    hf_adapter2 = ADAPTERS_DIR / "adapter_model.bin"

    if not any([mlx_adapter.exists(), hf_adapter.exists(), hf_adapter2.exists()]):
        console.print(
            "[red]❌ No adapter files found in:[/red] "
            f"[cyan]{ADAPTERS_DIR}[/cyan]\n"
            "Run [cyan]python train.py[/cyan] first."
        )
        sys.exit(1)

    is_mlx = mlx_adapter.exists()
    console.print(f"[green]✅ Adapters found[/green] ({'MLX' if is_mlx else 'HuggingFace'} format)")
    return is_mlx


# ════════════════════════════════════════════════════════════
# Step 1 — Fuse adapters into base model
# ════════════════════════════════════════════════════════════

def fuse_mlx():
    """Merge MLX LoRA adapters into the base model weights."""
    console.print(Panel("Fusing LoRA adapters into base model...", title="Step 1", border_style="cyan"))
    FUSED_DIR.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        MODEL_ID,
        "--adapter-path", str(ADAPTERS_DIR),
        "--save-path",    str(FUSED_DIR),
        "--de-quantize",  # Required for GGUF conversion
    ], desc="mlx_lm.fuse")

    console.print(f"[green]✅ Fused model saved to:[/green] {FUSED_DIR}")


def fuse_hf():
    """Merge HuggingFace PEFT adapters into the base model."""
    console.print(Panel("Fusing HuggingFace LoRA adapters...", title="Step 1", border_style="cyan"))
    FUSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "transformers", "peft", "torch"])
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

    console.print("Loading base model (this takes a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    )

    console.print("Merging LoRA adapters...")
    model = PeftModel.from_pretrained(base, str(ADAPTERS_DIR))
    model = model.merge_and_unload()

    console.print(f"Saving fused model to {FUSED_DIR}...")
    model.save_pretrained(str(FUSED_DIR))
    tokenizer.save_pretrained(str(FUSED_DIR))
    console.print(f"[green]✅ Fused model saved[/green]")


# ════════════════════════════════════════════════════════════
# Step 2 — Clone / build llama.cpp
# ════════════════════════════════════════════════════════════

def ensure_llama_cpp():
    console.print(Panel("Setting up llama.cpp for GGUF conversion...", title="Step 2", border_style="cyan"))

    if not LLAMA_CPP_DIR.exists():
        console.print("Cloning llama.cpp...")
        run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)])

    # Install Python deps for conversion script explicitly
    run([
        sys.executable, "-m", "pip", "install", "-q", "gguf", "protobuf", "sentencepiece"
    ], desc="llama.cpp requirements")

    # Build quantize binary
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        console.print("Building llama.cpp (cmake)...")
        build_dir = LLAMA_CPP_DIR / "build"
        build_dir.mkdir(exist_ok=True)
        run(["cmake", "..", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF"], cwd=build_dir)
        run(["cmake", "--build", ".", "--config", "Release", "-j4"], cwd=build_dir)

    console.print("[green]✅ llama.cpp ready[/green]")
    return quantize_bin


# ════════════════════════════════════════════════════════════
# Step 3 — Convert fused model → GGUF
# ════════════════════════════════════════════════════════════

def convert_to_gguf():
    console.print(Panel("Converting to GGUF format...", title="Step 3", border_style="cyan"))
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    f16_path = GGUF_DIR / "model-f16.gguf"

    run([
        sys.executable,
        str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
        str(FUSED_DIR),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ], desc="GGUF conversion")

    console.print(f"[green]✅ GGUF saved:[/green] {f16_path}")
    return f16_path


# ════════════════════════════════════════════════════════════
# Step 4 — Quantize to Q4_K_M (smaller + faster)
# ════════════════════════════════════════════════════════════

def quantize_gguf(f16_path: Path, quantize_bin: Path) -> Path:
    console.print(Panel("Quantizing to Q4_K_M (50% smaller, minimal quality loss)...", title="Step 4", border_style="cyan"))

    q4_path = GGUF_DIR / "model-q4_k_m.gguf"
    run([str(quantize_bin), str(f16_path), str(q4_path), "Q4_K_M"])

    size_gb = q4_path.stat().st_size / 1e9
    console.print(f"[green]✅ Quantized model:[/green] {q4_path} ({size_gb:.1f} GB)")
    return q4_path


# ════════════════════════════════════════════════════════════
# Step 5 — Register in Ollama
# ════════════════════════════════════════════════════════════

def register_ollama(gguf_path: Path):
    console.print(Panel(f"Registering '{OLLAMA_NAME}' in Ollama...", title="Step 5", border_style="cyan"))

    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_content = f"""FROM {gguf_path}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    modelfile_path.write_text(modelfile_content)
    console.print(f"[dim]Modelfile written to {modelfile_path}[/dim]")

    run(["ollama", "create", OLLAMA_NAME, "-f", str(modelfile_path)], desc="ollama create")

    console.print(Panel(
        f"[bold green]🎉 Your custom model is ready![/bold green]\n\n"
        f"Test it:  [cyan]ollama run {OLLAMA_NAME}[/cyan]\n\n"
        f"To use it in the app, edit [cyan]backend/config.py[/cyan]:\n"
        f'  [dim]OLLAMA_LLM_MODEL: str = "{OLLAMA_NAME}"[/dim]',
        border_style="green"
    ))


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    console.print(Panel(
        "[bold]Export Fine-tuned Model → Ollama[/bold]\n"
        "Fuse adapters → GGUF → Quantize → Register",
        border_style="magenta"
    ))

    is_mlx = check_adapters()

    # Step 1: Fuse
    if is_mlx:
        fuse_mlx()
    else:
        fuse_hf()

    # Steps 2-4: llama.cpp + GGUF + quantize
    quantize_bin = ensure_llama_cpp()
    f16_path     = convert_to_gguf()
    q4_path      = quantize_gguf(f16_path, quantize_bin)

    # Step 5: Register in Ollama
    register_ollama(q4_path)


if __name__ == "__main__":
    main()
