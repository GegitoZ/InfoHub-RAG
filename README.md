# InfoHub RAG Assistant (Georgian)

A Retrieval-Augmented Generation (RAG) system that answers customs-related questions in Georgian and cites official sources from https://infohub.rs.ge.

## Features
- Scrapes InfoHub documents
- Builds a vector index (FAISS)
- Hybrid retrieval (semantic + keyword)
- Generates answers in Georgian using OpenAI
- Always cites InfoHub sources
- FastAPI backend + Streamlit web interface

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m playwright install
