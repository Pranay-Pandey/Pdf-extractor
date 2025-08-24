import os
from pinecone import Pinecone
import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from enum import Enum
import json
import logging
import numpy as np
import OCR as ocr_module

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[os.getenv("MONGO_DB", "pdfextractor")]
collection = db[os.getenv("MONGO_COLLECTION", "documents")]


# DOC_ENDPOINT setup
# DOC_ENDPOINT setup
DOC_ENDPOINT = os.getenv('DOC_ENDPOINT')

# Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'doc-chunks')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Gemini setup
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# DocumentType Enum
class DocumentType(str, Enum):
    FIR = "FIR"
    Criminal_Complaint = "Criminal_Complaint"
    Criminal_Affidavit = "Criminal_Affidavit"
    Bail_Application = "Bail_Application"
    Vakalatnama = "Vakalatnama"
    Civil_Plaint = "Civil_Plaint"
    Written_Statement = "Written_Statement"
    Civil_Affidavit = "Civil_Affidavit"
    Document_Index = "Document_Index"
    Petitions_Interlocutory = "Petitions_Interlocutory"
    Summons_Notice = "Summons_Notice"
    Judgment_Decree = "Judgment_Decree"
    Contracts_Agreements = "Contracts_Agreements"
    Personal_ID = "Personal_ID"
    Medical_Photos_Evidence = "Medical_Photos_Evidence"
    Writ_Petition = "Writ_Petition"

# FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PROMPT = (
    "You are a legal document parser. "
    "Classify the document into one of the following types: "
    f"{[e.value for e in DocumentType]}. "
    "Extract the document type as 'document_type' (from the enum), and extract all relevant metadata as JSON. Do not make any nested JSON objects, all fields should be at the top level. "
    "If possible, extract fields like parties, dates, relief, etc. "
    "Respond in only in JSON. No thinking response or explanations."
)


# Helper function: extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    output_folder = os.path.join(tempfile.gettempdir(), "pdf_images")
    output_file = os.path.join(tempfile.gettempdir(), "extracted_text.txt")
    # Run OCR
    content = ocr_module.process_large_pdf(pdf_path, output_folder, output_file)
    # Clean up temp images
    image_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
    ocr_module.delete_temp_images(image_paths, output_folder)
    # Remove output file
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        logging.warning(f"Failed to delete temp output file: {e}")
    return content

# Helper functions
def gemini_extract(content, prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([prompt, content])
    return response.text

# API endpoint

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_chunk_embedding(pc, chunk: str):
    if not chunk.lower().startswith("passage: "):
        chunk = "passage: " + chunk
    return pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[chunk],
        parameters={"input_type": "passage"},
    )[0]["values"]

def get_query_embedding(pc, query: str):
    if not query.lower().startswith("query: "):
        query = "query: " + query
    return pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"},
    )[0]["values"]

class ExtractRequest(BaseModel):
    workspace_id: str
    doc_id: str
    prompt: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_chunks(req: SearchRequest):
    embedding = get_query_embedding(pc, req.query)
    results = index.query(vector=embedding, top_k=req.top_k, include_metadata=True)
    matches = []
    for match in results.get('matches', []):
        meta = match.get('metadata', {})
        matches.append({
            "fileId": meta.get('fileId'),
            "workspace_id": meta.get('workspace_id'),
            "chunkId": meta.get('chunkId'),
            "text": meta.get('text'),
            "score": match.get('score'),
            "pageId": meta.get('pageId', None),
        })
    return {"matches": matches}


class WorkspaceSearchRequest(BaseModel):
    query: str
    workspace_id: str
    top_k: int = 5


@app.post("/workspace_search")
def workspace_search(req: WorkspaceSearchRequest):
    """Search only documents that belong to the provided workspace_id.
    This endpoint filters Pinecone results by workspace_id.
    """
    if not req.query or not req.workspace_id:
        raise HTTPException(status_code=400, detail="query and workspace_id are required")

    embedding = get_query_embedding(pc, req.query)
    try:
        results = index.query(
            vector=embedding,
            top_k=req.top_k,
            include_metadata=True,
            filter={"workspace_id": req.workspace_id},
        )
    except Exception as e:
        logging.error("Error querying Pinecone for workspace %s: %s", req.workspace_id, e)
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    matches = []
    for match in results.get('matches', []):
        meta = match.get('metadata', {})
        matches.append({
            "fileId": meta.get('fileId'),
            "workspace_id": meta.get('workspace_id'),
            "chunkId": meta.get('chunkId'),
            "text": meta.get('text'),
            "score": match.get('score'),
            "pageId": meta.get('pageId', None),
        })

    return {"matches": matches}


class MultiSearchRequest(BaseModel):
    query: str
    workspace_id: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 5


@app.post("/multi_retrieve")
def multi_retrieve(req: MultiSearchRequest):
    """Retrieve matches iteratively for each doc_id (or across workspace if none provided),
    then re-rank all retrieved matches by score and return the top_k results.
    """
    if not req.query or not req.workspace_id:
        raise HTTPException(status_code=400, detail="query and workspace_id are required")

    embedding = get_query_embedding(pc, req.query)
    collected = []

    # If doc_ids provided, query per doc_id; otherwise query across workspace
    if req.doc_ids:
        for doc_id in req.doc_ids:
            try:
                results = index.query(
                    vector=embedding,
                    top_k=req.top_k,
                    include_metadata=True,
                    filter={"fileId": doc_id, "workspace_id": req.workspace_id},
                )
            except Exception as e:
                logging.error("Error querying Pinecone for doc %s: %s", doc_id, e)
                continue

            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                collected.append({
                    "fileId": meta.get("fileId"),
                    "workspace_id": meta.get("workspace_id"),
                    "chunkId": meta.get("chunkId"),
                    "text": meta.get("text"),
                    "score": match.get("score"),
                    "pageId": meta.get("pageId", None),
                })
    else:
        # Search across the workspace
        try:
            results = index.query(
                vector=embedding,
                top_k=req.top_k,
                include_metadata=True,
                filter={"workspace_id": req.workspace_id},
            )
        except Exception as e:
            logging.error("Error querying Pinecone for workspace %s: %s", req.workspace_id, e)
            raise HTTPException(status_code=500, detail=f"Search error: {e}")

        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            collected.append({
                "fileId": meta.get("fileId"),
                "workspace_id": meta.get("workspace_id"),
                "chunkId": meta.get("chunkId"),
                "text": meta.get("text"),
                "score": match.get("score"),
                "pageId": meta.get("pageId", None),
            })

    # Re-rank collected results by score (descending) and return top_k overall
    collected_sorted = sorted(collected, key=lambda x: x.get("score", 0), reverse=True)
    top_results = collected_sorted[:req.top_k]

    return {"matches": top_results, "total_found": len(collected)}



@app.post("/extract")
def extract_from_doc_endpoint(req: ExtractRequest):
    # Download PDF from DOC_ENDPOINT
    if not DOC_ENDPOINT:
        raise HTTPException(status_code=500, detail="DOC_ENDPOINT environment variable not set.")
    if not req.workspace_id or not req.doc_id:
        raise HTTPException(status_code=400, detail="workspace_id and doc_id are required.")
    url = f"{DOC_ENDPOINT.rstrip('/')}/{req.workspace_id}/vault/{req.doc_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdfURL = response.json().get('url', url)  # Use the URL from the response if available
        response = requests.get(pdfURL)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
            logging.info("Downloaded PDF to temporary file: %s", tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from DOC_ENDPOINT: {e}")

    page_texts = extract_text_from_pdf(tmp_path)  # dict: {page_num: text}
    logging.info("Extracted content from PDF: %s", page_texts)

    if not page_texts or len(page_texts) == 0:
        logging.warning("No content extracted from PDF.")
        return {"status_code": 400, "status": "error", "message": "No content extracted from PDF."}

    # Join all page texts to get full document text
    full_text = "\n\n".join([str(t) for t in page_texts.values()])

    # Run Gemini extraction on the full document
    prompt = DEFAULT_PROMPT
    if req.prompt:
        prompt += " " + req.prompt

    try:
        extracted_json = gemini_extract(full_text, prompt)
        jsonStartIndex = extracted_json.find("{")
        jsonEndIndex = extracted_json.rfind("}")
        if jsonStartIndex == -1 or jsonEndIndex == -1 or jsonEndIndex < jsonStartIndex:
            logging.error("Failed to find valid JSON in Gemini response., response: %s", extracted_json)
            raise HTTPException(status_code=500, detail="Invalid JSON response from Gemini.")
        extracted_json = extracted_json[jsonStartIndex:jsonEndIndex + 1]
        data = json.loads(extracted_json)
        document_type = data.get("document_type")
    except Exception as e:
        data = {
            "error": "Failed to parse extracted JSON",
            "details": str(e)
        }
        logging.info("Failed to parse extracted JSON: %s", extracted_json)
        logging.error("Error parsing extracted JSON: %s", e)
        return {"status_code": 500, "status": "error", "message": str(e)}
        document_type = None

    # Store in MongoDB
    page_texts_str = {str(k): v for k, v in page_texts.items()}
    record = {
        "_id": req.doc_id,
        "document_type": document_type,
        "extracted_json": data,
        "raw_text": page_texts_str,
        "workspace_id": req.workspace_id,
    }
    collection.replace_one({"_id": req.doc_id, "workspace_id": req.workspace_id}, record, upsert=True)

    # Chunk and upload to Pinecone (one chunk per page)
    logging.info("Preparing Pinecone vectors...")
    pinecone_vectors = []
    logging.info("Generating embeddings for Pinecone... Total pages: %d", len(page_texts))
    for page_id, chunk in page_texts.items():
        embedding = get_chunk_embedding(pc, chunk)
        pinecone_vectors.append({
            "id": f"{req.doc_id}_page_{page_id}",
            "values": embedding,
            "metadata": {
                "fileId": req.doc_id,
                "workspace_id": req.workspace_id,
                "chunkId": page_id,
                "pageId": page_id,
                "text": chunk
            }
        })
    if pinecone_vectors:
        index.upsert(vectors=pinecone_vectors)
        print("Pinecone vectors upserted.")

    return {"status": "success", "document_type": document_type, "data": data}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")