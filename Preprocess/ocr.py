from pathlib import Path
import pdfplumber
import subprocess
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm
from multiprocessing import Pool

def read_pdf(pdf_loc, page_infos: list = None):
    try:
        pdf = pdfplumber.open(pdf_loc)
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for page in pages:
            text = page.extract_text()
            if text:
                pdf_text += text
        pdf.close()
        return pdf_text
    except Exception as e:
        print(f"Error reading {pdf_loc}: {e}")
        return ""

def detect_language(text):
    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    confidence_values = detector.compute_language_confidence_values(text)

    for confidence in confidence_values:
        if confidence.language == Language.CHINESE and confidence.value >= 0.95:
            return True  # Detected Chinese with high confidence
    return False

def run_ocr(input_file):
    if input_file.stem.endswith('-output'):
        return
    output_file = input_file.stem + "-output.pdf"  # Generate output file name
    output_file_path = input_file.parent / output_file

    try:
        # Run the ocrmypdf command
        subprocess.run(
            ["ocrmypdf", "-l", "chi_tra", "--pdf-renderer=sandwich", "--force-ocr", "-q", str(input_file), str(output_file_path)],
            check=True
        )
        print(f"OCR completed for {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during OCR for {input_file}: {e}")

def process_file(pdf_file):
    content = read_pdf(pdf_file)

    if not content.strip():
        print(f"{pdf_file} is empty")
        run_ocr(pdf_file)
    elif not detect_language(content):
        print(f"{pdf_file} is not Chinese")
        run_ocr(pdf_file)

def main(directory_path):
    path = Path(directory_path)
    pdf_files = list(path.rglob("*.pdf"))  # Use rglob to include subdirectories

    # Use multiprocessing to process files concurrently
    with Pool(4) as pool:
        list(tqdm(pool.imap(process_file, pdf_files), total=len(pdf_files), desc="Processing PDF files"))


directory_path = Path("./final_project_dataset/reference")
# directory_path = Path(".")
main(directory_path)
