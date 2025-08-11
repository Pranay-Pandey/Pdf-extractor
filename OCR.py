import os
from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Initialize Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i+1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    return image_paths

def batch_images(image_paths, batch_size=30):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def ocr_with_gemini(image_path, instruction):
    image = Image.open(image_path)
    prompt = f"""
    {instruction}
    This is a page from a PDF document. Extract all text content while preserving the structure.
    Pay special attention to tables, columns, headers, and any structured content.
    Maintain paragraph breaks and formatting.
    """
    response = model.generate_content([prompt, image])
    return response.text

def process_large_pdf(pdf_path, output_folder, output_file):
    image_paths = convert_pdf_to_images(pdf_path, output_folder)
    page_texts = {}
    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i+1}...")
        page_text = ocr_with_gemini(image_path, "Extract all text, maintaining document structure")
        page_texts[i+1] = page_text
    # Optionally write all text to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for page_num, text in page_texts.items():
            f.write(f"--- PAGE {page_num} ---\n{text}\n")
    return page_texts

def delete_temp_images(image_paths, output_folder="pdf_images"):
    for path in image_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    if os.path.exists(output_folder) and not os.listdir(output_folder):
        os.rmdir(output_folder)

if __name__ == "__main__":
    # Example usage
    pdf_path = "1682147509.pdf"  # Change to your PDF file
    output_folder = "pdf_images"
    output_file = "extracted_text.txt"
    process_large_pdf(pdf_path, output_folder, output_file)
    print(f"OCR complete. Extracted text saved to {output_file}")