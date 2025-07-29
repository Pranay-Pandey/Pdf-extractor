import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import requests
from dotenv import load_dotenv
import fitz  # PyMuPDF
import google.generativeai as genai
from pymongo import MongoClient
from enum import Enum
import json

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[os.getenv("MONGO_DB", "pdfextractor")]
collection = db[os.getenv("MONGO_COLLECTION", "documents")]


# DOC_ENDPOINT setup
DOC_ENDPOINT = os.getenv('DOC_ENDPOINT')

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
    "Respond in JSON. "
)


# Helper function: extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def gemini_extract(content, prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([prompt, content])
    return response.text

# API endpoint
class ExtractRequest(BaseModel):
    doc_id: str
    prompt: Optional[str] = None



@app.post("/extract")
def extract_from_doc_endpoint(req: ExtractRequest):
    # Download PDF from DOC_ENDPOINT
    if not DOC_ENDPOINT:
        raise HTTPException(status_code=500, detail="DOC_ENDPOINT environment variable not set.")
    url = f"{DOC_ENDPOINT.rstrip('/')}/{req.doc_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdfURL = response.json().get('url', url)  # Use the URL from the response if available
        response = requests.get(pdfURL)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from DOC_ENDPOINT: {e}")

    # Extract text (using PyMuPDF for all PDFs)
    content = extract_text_from_pdf(tmp_path)

    # Build prompt
    prompt = DEFAULT_PROMPT
    if req.prompt:
        prompt += " " + req.prompt

    # Gemini extraction
    try:
        extracted_json = gemini_extract(content, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini extraction failed: {e}")

    jsonStartIndex = extracted_json.find("{")
    jsonEndIndex = extracted_json.rfind("}")
    if jsonStartIndex == -1 or jsonEndIndex == -1 or jsonEndIndex < jsonStartIndex:
        raise HTTPException(status_code=500, detail="Invalid JSON response from Gemini.")
    extracted_json = extracted_json[jsonStartIndex:jsonEndIndex + 1]  # Ensure we only parse the JSON part
    print("Extracted JSON:", extracted_json)
    try:
        data = json.loads(extracted_json)
        document_type = data.get("document_type")
        print("Extracted document type:", document_type)
        print("Extracted data:", data)
    except Exception as e:
        data = {
            "error": "Failed to parse extracted JSON",
            "details": str(e)
        }
        print(f"Error parsing extracted JSON: {e}")
        return {"status_code": 500, "status": "error", "message": str(e)}
        document_type = None

    # Store in MongoDB
    record = {
        "_id": req.doc_id,
        "document_type": document_type,
        "extracted_json": data if 'data' in locals() else extracted_json,
        "raw_text": content,
        **{key: value for key, value in data.items() if key not in ["_id", "document_type", "extracted_json", "raw_text"]}
    }
    collection.replace_one({"_id": req.doc_id}, record, upsert=True)

    return {"status": "success", "document_type": document_type, "data": data if 'data' in locals() else extracted_json}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)