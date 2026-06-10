# 🧠 AI Data Fetcher — Document Intelligence System

An **in-house RAG-based AI system** that reads PDFs/brochures and automatically extracts structured product data into Excel — no cloud APIs, no recurring costs.

---

## Architecture

```
Frontend (React)  ←→  FastAPI Backend  ←→  Ollama (Local LLM)
                           ↕
                       ChromaDB (Vector Store)
```

| Component | Technology |
|---|---|
| Chat UI | React + Vite |
| Backend API | FastAPI (Python) |
| Local AI Model | Ollama · qwen2.5:14b |
| Embeddings | nomic-embed-text |
| PDF Parsing | pdfplumber + pytesseract OCR |
| Vector Database | ChromaDB (local) |
| Excel Export | pandas + openpyxl |

---

## Prerequisites

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

### 2. Pull required models
```bash
# Main LLM (best for 32GB+ RAM)
ollama pull qwen2.5:14b

# Embedding model
ollama pull nomic-embed-text

# Start Ollama server (if not auto-started)
ollama serve
```

### 3. Install Tesseract OCR (for scanned PDFs)
```bash
# macOS
brew install tesseract

# Verify
tesseract --version
```

### 4. Python 3.11+
```bash
python3 --version   # Should be 3.11 or higher
```

---

## Setup & Run

### Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Start backend server
python main.py
# → Running on http://localhost:8000
# → API docs: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
# → Running on http://localhost:5173
```

---

## Usage

1. **Open** `http://localhost:5173` in your browser
2. **Upload** a PDF brochure using the 📎 button or drag-and-drop
3. The AI automatically **extracts product data** and shows it in a table
4. Click **⬇️ Download Excel** to save the structured data
5. You can also **chat** with the document: ask specific questions like:
   - *"What are all the available models?"*
   - *"List the prices for each product"*
   - *"What are the technical specifications?"*

---

## Project Structure

```
phase-1 (AI data fetcher)/
├── backend/
│   ├── main.py                  FastAPI app (all endpoints)
│   ├── config.py                Settings & paths
│   ├── requirements.txt
│   ├── services/
│   │   ├── pdf_service.py       PDF text extraction + OCR
│   │   ├── embedding_service.py ChromaDB + embeddings
│   │   ├── llm_service.py       Ollama LLM calls
│   │   └── excel_service.py     Excel export
│   ├── models/
│   │   └── schemas.py           Pydantic models
│   └── uploads/                 Uploaded files (auto-created)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              Main app
│   │   ├── components/          UI components
│   │   ├── hooks/               React hooks
│   │   └── services/api.js      Backend API client
│   └── index.html
│
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check Ollama + system status |
| POST | `/upload` | Upload & index a PDF |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document |
| POST | `/chat` | Stream RAG chat response (SSE) |
| POST | `/extract/json` | Extract structured product data |
| POST | `/export/excel` | Download as .xlsx file |

---

## Model Selection Guide

| RAM | Recommended Model | Quality |
|---|---|---|
| 8 GB | `llama3.2:3b` | Good |
| 16 GB | `llama3.1:8b` | Better |
| **32 GB+** | **`qwen2.5:14b`** | **Best** |

To change the model, edit `backend/config.py`:
```python
OLLAMA_LLM_MODEL: str = "qwen2.5:14b"
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| "Ollama offline" in sidebar | Run `ollama serve` in terminal |
| "Model not found" | Run `ollama pull qwen2.5:14b` |
| OCR not working | Install Tesseract: `brew install tesseract` |
| Slow extraction | Normal for 14B model — first run downloads model weights |
| CORS error | Ensure backend is on port 8000 |
# ai-data-fecth
