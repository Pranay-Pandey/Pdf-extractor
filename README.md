# PDF Extractor

A legal-document extraction service and a standalone PDF downloader. The FastAPI service extracts text from PDFs with Gemini-powered OCR, classifies and structures legal documents, stores extracted records in MongoDB, and indexes page text in Pinecone. The downloader discovers accessible PDFs on webpages and saves them locally.

## Features

### Extraction API
- Extract a document using the existing document-service integration (`POST /extract`).
- Extract from a direct PDF URL or a webpage exposing exactly one PDF (`POST /extract_url`).
- Reuse the same OCR, Gemini extraction, MongoDB storage, and Pinecone indexing pipeline for both routes.
- Search indexed chunks globally or within a workspace (`/search`, `/workspace_search`, `/multi_retrieve`).

### PDF downloader CLI
- Discover PDF URLs in links, iframes, embeds, objects, source elements, metadata, and inline scripts.
- Resolve relative URLs and validate downloaded content using the PDF signature.
- Stream downloads with a 200 MiB default size limit and avoid overwriting existing files.
- Optionally inspect JavaScript-driven viewers using Playwright and installed Chrome.

## Requirements

- Python 3.12 or newer for the downloader package.
- The extraction API additionally requires the dependencies in `requirements.txt` and working MongoDB, Pinecone, and Gemini credentials.
- Browser-based downloader inspection requires Google Chrome or a compatible Chromium executable.

## Installation

```bash
git clone https://github.com/Pranay-Pandey/pdf-extractor.git
cd pdf-extractor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

For optional browser support:

```bash
python -m pip install -e '.[browser,dev]'
```

The browser integration uses `/usr/bin/google-chrome` by default. Override it with `--chrome` if Chrome is installed elsewhere. Playwright's browser bundle is not downloaded automatically.

## Extraction API setup

Set these environment variables before starting the API:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MONGO_URI` | MongoDB connection URI | — |
| `MONGO_DB` | MongoDB database | `pdfextractor` |
| `MONGO_COLLECTION` | MongoDB collection | `documents` |
| `DOC_ENDPOINT` | Existing document-service base URL used by `/extract` | — |
| `PINECONE_API_KEY` | Pinecone API key | — |
| `PINECONE_INDEX` | Pinecone index name | `doc-chunks` |
| `GEMINI_API_KEY` | Gemini API key | — |

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The health check is available at `GET /health`. The existing Streamlit interface can be started with:

```bash
streamlit run app.py
```

### Existing document-service extraction

`POST /extract` accepts a workspace ID, document ID, and optional prompt. It fetches the document using `DOC_ENDPOINT` and runs the shared extraction pipeline.

```json
{
  "workspace_id": "workspace-123",
  "doc_id": "document-456",
  "prompt": "Extract the parties and relevant dates"
}
```

### Extract from a URL

`POST /extract_url` accepts a direct PDF URL or a webpage URL. The page must expose exactly one downloadable PDF; ambiguous pages are rejected rather than selecting an arbitrary document.

```json
{
  "workspace_id": "workspace-123",
  "doc_id": "document-789",
  "url": "https://example.com/document.pdf",
  "prompt": "Extract the parties and relevant dates"
}
```

The URL route uses HTTP discovery only. It does not invoke the optional Chrome browser fallback. Both API routes pass the downloaded PDF through the same OCR, Gemini, MongoDB, and Pinecone processing function.

**Deployment note:** Treat URL extraction as a privileged operation. Restrict access to the API and apply outbound network/host controls at the deployment boundary; do not expose an unrestricted URL-fetching endpoint to untrusted callers.

## Standalone downloader CLI

Download PDFs discovered on a webpage or a direct PDF URL:

```bash
pdf-extractor "https://example.com/document-page"
pdf-extractor "https://example.com/document.pdf" --output-dir ./downloads
```

Inspect a JavaScript-driven viewer using Chrome:

```bash
pdf-extractor "https://example.com/document-page" --browser
```

| Option | Description |
| --- | --- |
| `--output-dir PATH` | Download directory (default: `./output`) |
| `--browser` | Always inspect the page using Chrome |
| `--no-browser` | Disable automatic browser fallback |
| `--chrome PATH` | Chrome executable path (default: `/usr/bin/google-chrome`) |
| `--timeout SECONDS` | HTTP and browser timeout (default: 30 seconds) |

`--browser` and `--no-browser` cannot be used together.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src/pdf_extractor tests/test_pdf_downloader.py
```

## Scope and limitations

Use this tool only for PDFs that are publicly accessible or that you are authorized to access. It does not bypass subscriptions, login requirements, DRM, encryption, or other access controls. Some sites generate document URLs dynamically, require authenticated browser sessions, block automated requests, or render pages as images or canvas elements. The HTTP API route does not reconstruct image-only viewers or use browser-based capture.
