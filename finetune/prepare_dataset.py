"""
prepare_dataset.py
══════════════════
Converts PDF + JSON label pairs into an MLX-LM compatible JSONL training file.

Directory structure expected:
  training_data/
  ├── pdfs/         ← your brochure PDFs (e.g. product_A.pdf)
  └── labels/       ← matching JSON files (e.g. product_A.json)

Each label JSON should be an array of extracted products:
  [
    {"name": "...", "model": "...", "price": "...", "specs": "..."},
    ...
  ]

Run:
  python prepare_dataset.py

Output:
  training_data/processed/train.jsonl
  training_data/processed/valid.jsonl
"""

import json
import random
import sys
from pathlib import Path

import pdfplumber
import fitz
from PIL import Image
import pytesseract
import yaml
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Load config ─────────────────────────────────────────────
ROOT = Path(__file__).parent
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PDFS_DIR      = ROOT / cfg["data"]["pdfs_dir"]
LABELS_DIR    = ROOT / cfg["data"]["labels_dir"]
PROCESSED_DIR = ROOT / cfg["data"]["processed_dir"]
TRAIN_SPLIT   = cfg["data"]["train_split"]
SYSTEM_PROMPT = cfg["system_prompt"].strip()


# ── PDF/TXT text extraction ────────
def extract_pdf_text(doc_path: Path) -> str:
    if doc_path.suffix.lower() == ".txt":
        return doc_path.read_text(errors="ignore")
    pages = []
    try:
        with pdfplumber.open(doc_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 50:
                    text = _ocr_page(doc_path, i)
                pages.append(text)
    except Exception:
        doc = fitz.open(str(doc_path))
        for i in range(len(doc)):
            pages.append(_ocr_page(doc_path, i))
    return "\n\n".join(pages)


def _ocr_page(pdf_path: Path, page_idx: int) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_idx]
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang="eng")
    except Exception:
        return ""


# ── Build one training example ──────────────────────────────
def build_example(pdf_text: str, label_json: list) -> dict:
    """
    Returns a single training example in MLX-LM chat format.
    """
    user_content = (
        "Extract ALL product information from the following document text.\n"
        "Return a JSON array where each element is a product with all its details.\n\n"
        f"DOCUMENT:\n{pdf_text[:3500]}"   # cap at 3500 chars to fit context
    )
    assistant_content = json.dumps(label_json, indent=2, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Find matched PDF/TXT ↔ label pairs
    pairs = []
    for doc_path in list(PDFS_DIR.glob("*.pdf")) + list(PDFS_DIR.glob("*.txt")):
        label_path = LABELS_DIR / (doc_path.stem.replace("_SAMPLE", "") + ".json")
        if label_path.exists():
            pairs.append((doc_path, label_path))
        else:
            console.print(f"[yellow]⚠️  No label found for {doc_path.name} — skipping[/yellow]")

    if not pairs:
        console.print(
            "[red]❌ No matched PDF+JSON pairs found![/red]\n"
            f"Put PDFs in:    [cyan]{PDFS_DIR}[/cyan]\n"
            f"Put labels in:  [cyan]{LABELS_DIR}[/cyan]\n"
            "Label filename must match PDF filename (e.g. product_A.pdf → product_A.json)"
        )
        sys.exit(1)

    console.print(f"\n[bold green]✅ Found {len(pairs)} PDF+label pairs[/bold green]\n")

    # Build examples
    examples = []
    for pdf_path, label_path in pairs:
        console.print(f"  Processing [cyan]{pdf_path.name}[/cyan]...")
        try:
            pdf_text  = extract_pdf_text(pdf_path)
            with open(label_path) as f:
                label_data = json.load(f)

            # If label is a dict with a key, unwrap it
            if isinstance(label_data, dict):
                for k in ("products", "items", "data"):
                    if k in label_data:
                        label_data = label_data[k]
                        break

            example = build_example(pdf_text, label_data)
            examples.append(example)
            console.print(f"    [green]✓[/green] {len(label_data)} products, {len(pdf_text)} chars")
        except Exception as exc:
            console.print(f"    [red]✗ Error: {exc}[/red]")

    if not examples:
        console.print("[red]No valid examples generated.[/red]")
        sys.exit(1)

    # Shuffle and split
    random.shuffle(examples)
    split_idx   = max(1, int(len(examples) * TRAIN_SPLIT))
    train_data  = examples[:split_idx]
    valid_data  = examples[split_idx:] or examples[-1:]  # at least 1 val example

    # Write JSONL files
    train_path = PROCESSED_DIR / "train.jsonl"
    valid_path = PROCESSED_DIR / "valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary table
    table = Table(title="Dataset Summary", style="bold")
    table.add_column("Split",    style="cyan")
    table.add_column("Examples", style="green")
    table.add_column("Path",     style="dim")
    table.add_row("Train", str(len(train_data)), str(train_path))
    table.add_row("Valid", str(len(valid_data)), str(valid_path))
    console.print(table)

    console.print(
        "\n[bold green]✅ Dataset ready![/bold green] "
        "Now run: [cyan]python train.py[/cyan]\n"
    )


if __name__ == "__main__":
    main()
