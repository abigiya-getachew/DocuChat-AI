# DocuChat AI — Local RAG Document Intelligence

Upload PDFs, ask questions, and get precise answers with **exact page citations** — powered entirely by local tools.
No cloud fees, no API keys, no data leaving your machine.

---

## How It Works (Step by Step)

```
  Your PDF
     │
     ▼  Step 1: Load & Split
  [ document_loader.py ]
  Reads the PDF page-by-page using PyPDF.
  Splits each page into overlapping chunks (~1,000 chars each).
     │
     ▼  Step 2: Embed
  [ embedding_engine.py ]
  Sends each text chunk to Ollama's `nomic-embed-text` model.
  Converts chunk meaning into a 768-number vector.
     │
     ▼  Step 3: Store
  [ vector_store.py ]
  Saves all vectors + text + metadata (source file, page number)
  into a local ChromaDB database.
     │
     ▼  Step 4: Query
  You ask a question → it is also embedded into a vector.
  ChromaDB searches all stored chunks by vector similarity.
  Returns the top 5 most relevant chunks.
     │
     ▼  Step 5: Generate
  [ llm_responder.py ]
  Builds a context from the retrieved chunks.
  Sends it to `llama3.2` via Ollama with the instruction:
  "Use ONLY the provided context and cite source + page."
  Streams back a grounded, cited answer.
```

---

## Project Structure

```
DocuChat_AI/
├── src/
│   ├── config.py            # All settings (model, chunk size, paths)
│   ├── document_loader.py   # PDF reading and text chunking
│   ├── embedding_engine.py  # Converts text chunks to vectors
│   ├── vector_store.py      # ChromaDB: add, search, delete chunks
│   └── llm_responder.py     # RAG retrieval + Ollama LLM generation
├── data/
│   └── uploaded_docs/       # Drop PDFs here (auto-created)
├── chroma_db/               # Local vector database (auto-created)
├── examples/
│   └── quick_start.py       # CLI demo: index + ask in the terminal
├── web_app.py               # Flask web interface (main entry point)
├── .env                     # Your local configuration overrides
├── .env.example             # Template for .env
└── requirements.txt
```

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running locally

---

## Setup (First Time)

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd DocuChat_AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install flask               # Required for the web app
```

### 4. Pull the required Ollama models

```bash
ollama pull llama3.2            # LLM for generating answers
ollama pull nomic-embed-text    # Embedding model for vectorizing text
```

> Make sure Ollama is running before you start (`ollama serve`).
> If you see `address already in use`, Ollama is already active — that's fine.

### 5. Configure environment (optional)

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

```dotenv
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DATA_FOLDER=./data/uploaded_docs
```

---

## Running DocuChat AI

### Option A — Web Interface (Recommended)

```bash
python3 web_app.py
```

Open **http://localhost:5000** in your browser.

**What you can do in the web UI:**
1. **Upload a PDF** — drag & drop or click to select; hit "Index Document"
2. **Watch it embed** — the progress bar animates while Ollama vectorizes your chunks
3. **Ask any question** — type in the chat box and hit Send
4. **Read cited answers** — the AI replies with exact `[Source: file.pdf, Page N]` references
5. **Delete a document** — click the 🗑️ trash icon in the Document Library to remove both the file and its ChromaDB vectors
6. **Toggle themes** — switch between Light and Dark mode using the header button

### Option B — Command Line

```bash
# Make sure your PDF is in data/uploaded_docs/ first
python3 examples/quick_start.py
```

This will:
1. Find all PDFs in `data/uploaded_docs/`
2. Chunk, embed, and index them into ChromaDB
3. Open an interactive Q&A loop in your terminal

---

## 🚀 Deploy on Render (100% Free Tier)

Render's free tier provides 512MB RAM, which cannot run local Ollama. We configured DocuChat AI to run seamlessly on Render using **in-process embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) and the free **Groq Cloud API** for ultra-fast Llama 3 answers.

### Step 1: Get a Free Groq API Key
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up and click **Create API Key** (completely free, no credit card required)

### Step 2: Push Your Code to GitHub
```bash
git add .
git commit -m "chore: prepare for Render deployment"
git push origin main
```

### Step 3: Deploy on Render
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**
2. Connect your **DocuChat_AI** repository
3. Configure settings:
   - **Name**: `docuchat-ai`
   - **Environment**: `Python 3`
   - **Region**: Any (e.g. Frankfurt or Oregon)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT web_app:app`
   - **Instance Type**: **Free** ($0/month)
4. Add **Environment Variables** in the Environment tab:
   - `GROQ_API_KEY` = `your_groq_api_key_here`
   - `DEFAULT_MODEL` = `llama-3.3-70b-versatile`
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**! Render will build and deploy your app.

---

## Multi-Document Support

DocuChat AI supports **multiple documents simultaneously**.

- Every chunk is tagged with its source filename and page number.
- When you ask a question, ChromaDB searches across **all indexed documents at once**.
- The LLM cites each fact with the exact file and page it came from.
- Deleting a document removes **only its chunks** from ChromaDB, leaving other documents intact.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `llama3.2` | LLM used for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model used to embed text chunks |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap characters between chunks |
| `DATA_FOLDER` | `./data/uploaded_docs` | Where uploaded PDFs are stored |

---

## API Endpoints (Web App)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web interface |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question (JSON or form) |
| `POST` | `/delete_document` | Remove a document and its vectors |
| `GET` | `/api/status` | Current chunk count, documents, models |

---

## Tech Stack

| Tool | Role |
|---|---|
| **Ollama** | Local LLM runtime (llama3.2 + nomic-embed-text) |
| **ChromaDB** | Local vector database for semantic search |
| **Flask** | Lightweight web server |
| **PyPDF** | PDF text extraction |
| **python-dotenv** | Environment variable management |

---

## Troubleshooting

**"0 Chunks" showing but file is listed:**
The PDF is saved to disk but not yet embedded. Click "Index Document" again or use the CLI.

**Embedding is slow:**
Normal — `nomic-embed-text` runs locally on CPU. A 100-page PDF (~400 chunks) takes ~2–3 minutes.
The 1,178-chunk `javascript.pdf` takes about 20 minutes to fully index.

**LLM says "I don't see any context":**
ChromaDB is empty. You need to index the document first (not just upload it).

**`ollama serve` says "address already in use":**
Ollama is already running in the background. This is fine — proceed normally.