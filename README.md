# Esports Intelligence

Esports Intelligence is a Python project for building an AI-powered knowledge assistant focused on BGMI esports. The project will eventually organize authoritative esports information so it can be searched and used to answer questions.

## Current phase

**Phase 0: Project scaffold.** This phase provides a minimal FastAPI service, environment-based configuration, and the initial project layout. The RAG pipeline and BGMI source data have not been added.

## Technology stack

- Python
- FastAPI
- Uvicorn
- Pydantic Settings

## Set up locally

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to customize local settings. `.env` is ignored by Git; do not put secrets in committed files.

## Start the API

With the virtual environment active, run:

```bash
uvicorn app.main:app --reload
```

Uvicorn starts the development server at `http://127.0.0.1:8000`.

## Check the health endpoint

Open `http://127.0.0.1:8000/api/health` in a browser, or run this in another terminal:

```bash
curl http://127.0.0.1:8000/api/health
```

The endpoint should return:

```json
{"status":"healthy","project":"Esports Intelligence"}
```

Downloaded rulebooks and other source documents are not included in this scaffold. Respect their copyright and keep local source files out of version control.