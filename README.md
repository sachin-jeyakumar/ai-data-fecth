# AI Data Fetcher

A full-stack intelligent application that extracts product data from complex PDFs (such as tool catalogs or visual brochures) using advanced AI models, OCR, and PDF parsing, and exports the structured data to Excel.

## Features
- **Intelligent RAG Extraction**: Automatically uses Groq (Llama-3.1-8b) or OpenRouter to extract tables and text blocks into structured JSON.
- **Offline Fallback**: Uses a local Ollama model (`qwen2.5:7b`) if internet or API keys fail.
- **OCR Integration**: Reads scanned documents automatically via Tesseract.
- **Excel Export**: Download styled `.xlsx` sheets natively from the browser.

## Tech Stack
- **Frontend**: React + Vite + TailwindCSS
- **Backend**: Python + FastAPI + Uvicorn + pdfplumber + Tesseract OCR
- **Database**: ChromaDB (Vector store for embeddings)

---

## 🚀 Deployment (Production)

The easiest way to deploy this application on any server is using Docker. It handles installing all system dependencies (like Tesseract and Poppler) automatically.

### Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).

### 1. Setup API Keys
Inside the `backend/` directory, ensure you have a `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Run with Docker Compose
Navigate to the root directory containing `docker-compose.yml` and run:
```bash
docker-compose up -d --build
```

- The **Frontend** will be available at: `http://localhost:80` (or your server's IP address)
- The **Backend API** will be available at: `http://localhost:8000`

### 3. Updating the API URL (If deploying to a VPS/Cloud)
If you deploy this to a remote server, the frontend running in the user's browser needs to know how to talk to the backend. Open `docker-compose.yml` and uncomment the `VITE_API_URL` arg under the `frontend` service, setting it to your server's public IP address or domain:
```yaml
      args:
        VITE_API_URL: "http://YOUR-SERVER-IP:8000"
```
Then rebuild: `docker-compose up -d --build`

---

## 🛠️ Local Development

### 1. Backend Setup
Make sure you have `tesseract-ocr` and `poppler-utils` installed on your machine.
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
