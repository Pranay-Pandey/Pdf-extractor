import os
import tempfile
import fitz
import pytest
from main import extract_text_from_pdf
import os

def test_extract_text_from_pdf_text_pdf():
    filePath = r"1682147509.pdf"
    if not os.path.exists(filePath):
        pytest.skip(f"File not found: {filePath}")
    text = extract_text_from_pdf(filePath)
    with open("1682147509.txt", "w", encoding="utf-8") as f:
        f.write(text)
    assert len(text) == 0

if __name__ == "__main__":
    pytest.main([__file__])
