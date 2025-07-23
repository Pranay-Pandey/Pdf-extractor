import os
import sys
import time
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import subprocess
from datetime import datetime

# Try to import markitdown, if available
try:
    from markitdown import extract
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

def is_pdf_text_based(pdf_path):
    """Check if the PDF has extractable text."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False

def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using markitdown or PyPDF2."""
    if MARKITDOWN_AVAILABLE:
        try:
            return extract(pdf_path)
        except Exception as e:
            print(f"markitdown failed: {e}, falling back to PyPDF2.")
    # Fallback to PyPDF2
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_image_pdf(pdf_path):
    """Extract text from an image-based PDF using OCR."""
    images = convert_from_path(pdf_path)
    text = ""
    for i, image in enumerate(images):
        text += pytesseract.image_to_string(image) + "\n"
    return text

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
        data = extract_text_from_pdf(pdf_path)
    else:
        print("Detected image-based PDF. Running OCR...")
        data = extract_text_from_image_pdf(pdf_path)
    append_to_output_file(data)
    print("Extraction complete. Data appended to extracted_data.txt.")

if __name__ == "__main__":
    main()
