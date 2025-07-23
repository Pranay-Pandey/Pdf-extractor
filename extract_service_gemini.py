# Add Gemini API client
# You need to set your Gemini API key in the environment variable GEMINI_API_KEY
# or replace os.getenv('GEMINI_API_KEY') with your key string directly.
# Install: pip install google-generativeai
import os
from dotenv import load_dotenv
import sys
import time
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
from datetime import datetime
import google.generativeai as genai

PROMPT = "Here is a pdf which will have some legal kind of document and you have to extract the information like name, address, concern, etc. Please use JSON as the default format for output."

# Try to import markitdown, if available
try:
    from markitdown import extract
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

def is_pdf_text_based(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False

def extract_text_from_pdf(pdf_path):
    if MARKITDOWN_AVAILABLE:
        try:
            return extract(pdf_path)
        except Exception as e:
            print(f"markitdown failed: {e}, falling back to PyPDF2.")
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_image_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for i, image in enumerate(images):
        text += pytesseract.image_to_string(image) + "\n"
    return text

def gemini_extract(content, prompt):
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Use the appropriate model
    response = model.generate_content([prompt, content])
    return response.text

def append_to_output_file(data, output_file="extracted_data.txt"):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n---\n{datetime.now().isoformat()}\n{data}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_service.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    print(f"Processing: {pdf_path}")
    if is_pdf_text_based(pdf_path):
        print("Detected text-based PDF.")
        content = extract_text_from_pdf(pdf_path)
    else:
        print("Detected image-based PDF. Running OCR...")
        content = extract_text_from_image_pdf(pdf_path)
    print("Sending content to Gemini for extraction...")
    try:
        extracted_json = gemini_extract(content, PROMPT)
    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        sys.exit(1)
    append_to_output_file(extracted_json)
    print("Extraction complete. Data appended to extracted_data.txt.")

if __name__ == "__main__":
    main()
