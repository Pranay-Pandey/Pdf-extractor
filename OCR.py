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

def ocr_with_gemini(image_paths, instruction):
    images = [Image.open(path) for path in image_paths]
    prompt = f"""
    {instruction}
    These are pages from a PDF document. Extract all text content while preserving the structure.
    Pay special attention to tables, columns, headers, and any structured content.
    Maintain paragraph breaks and formatting.
    """
    response = model.generate_content([prompt, *images])
    return response.text

def process_large_pdf(pdf_path, output_folder, output_file):
    image_paths = convert_pdf_to_images(pdf_path, output_folder)
    batches = batch_images(image_paths, 30)
    full_text = ""
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}...")
        batch_text = ocr_with_gemini(batch, "Extract all text, maintaining document structure")
        full_text += f"\n\n--- BATCH {i+1} ---\n\n{batch_text}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    return full_text

def delete_temp_images(image_paths):
    for path in image_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    if not os.listdir(output_folder):
        os.rmdir(output_folder)

if __name__ == "__main__":
    # Example usage
    pdf_path = "1682147509.pdf"  # Change to your PDF file
    output_folder = "pdf_images"
    output_file = "extracted_text.txt"
    process_large_pdf(pdf_path, output_folder, output_file)
    print(f"OCR complete. Extracted text saved to {output_file}")