# DocuChat AI — Intelligent Document Q&A System

Upload PDFs, contracts, reports, or manuals — ask questions and get precise answers
with direct citations from your files. Built entirely on local tools — no cloud fees,
no data leaves your machine.

## What it solves
- Scrolling through hundreds of pages to find one detail
- Cross-referencing clauses across long documents
- Slow manual summarization and fact-checking

## Tech stack
- Ollama → runs local LLM
- Python → core logic
- Chroma → local vector search
- PyPDF → read documents

## Quick start
1. `python -m venv .venv`
2. Activate `.venv`
3. `pip install -r requirements.txt`
4. `ollama pull llama3.2 nomic-embed-text`
5. Run `examples/quick_start.py`