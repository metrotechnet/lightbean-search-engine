# Lightbean Search Engine

This project evaluates product discovery and retrieval using Meilisearch and a chat-based workflow for Lightbeans materials.

## Contents
- `data/` — datasets and prompt configuration
- `scripts/` — conversion, search, chat, and report generation scripts
- `results/` — generated outputs from search and chat runs
- `start-search.ps1` and `start-chat.ps1` — launcher scripts for local execution

## Setup
1. Copy `.env.example` if present or create a local `.env` file with your Meilisearch credentials.
2. Install Python dependencies:
   `pip install python-dotenv python-docx`
3. Run the scripts from the project root.

## Notes
- `.env` is intentionally ignored and should never be committed to source control.
- The project is designed for local experimentation and evaluation of search relevance and chat responses.
