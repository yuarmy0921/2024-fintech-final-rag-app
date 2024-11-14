import google.generativeai as genai
from pathlib import Path
from time import sleep
from tqdm import tqdm
import json
import os, sys
import pdfplumber

BASE = Path("../final_project_dataset")
PROMPT2 = """請再重新想一次。如果你認同之前的判斷，請告訴我最能回答此問題的文本編號。
如果你不認同上述判斷，請再次判斷後，選出能回答此問題的文本編號。
記得，只會有正好一個答案。如果有多個可能的答案，選一個最合適的。回應編號即可，不要有其他文字。"""

API_KEY = ""
genai.configure(api_key = API_KEY)

faq_data = json.loads((BASE / "reference" / "faq" / "pid_map_content.json").read_text())
question_data = json.loads(Path("../questions_preliminary.json").read_text())

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-1.5-flash')

def read_pdf(pdf_loc, page_infos: list = None):
    """
    Reads text from a PDF file between specified page indices.

    Args:
        pdf_loc (Path): Path to the PDF file to be read.
        page_infos (list, optional): List containing start and end page indices to read.
                                     If None, reads all pages.

    Returns:
        str: Extracted text from the specified PDF pages.
    """
    pdf = pdfplumber.open(pdf_loc)

    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):
        text = page.extract_text()
        if text:
            pdf_text += text
            pdf_text += '\n'
    pdf.close()

    return pdf_text

# Function to generate a prompt for multiple contents
def generate_prompt_1(question, contents):
    """
    Generates a prompt for language model evaluation by consolidating
    question and content information from various text sources.

    Args:
        question (str): The question to be answered.
        contents (dict): Dictionary of content keyed by content ID.

    Returns:
        str: Formatted prompt string for language model processing.
    """
    prompt = f"這是一個問題：「{question}」。\n以下是幾個檔案中的文本，請逐一判斷每個文本是否包含可以回答這個問題的資訊，並回答「是」或「否」。如果回答「是」，請提取相關的句子並解釋為什麼這個文本可以回應問題。\n\n"
    for key, content in contents.items():
        prompt += f"文本 {key}：\n```\n{content}\n```\n\n\n"

    document_list_str = ','.join([f"文本{key}" for key in contents.keys()])
    prompt += f"請詳細閱讀上列文本（{document_list_str}），並討論每個文本是否可以回答問題：「{question}」，以及為什麼。"
    return prompt

# Function to analyze files in batch
def analyze_files_in_batch(question, file_ids, category, ground_truth=None):
    """
    Analyzes multiple files to find content most relevant to answering the question.

    Args:
        question (str): The question being evaluated.
        file_ids (list): List of file IDs to analyze.
        category (str): The category of files being processed (e.g., "faq").
        ground_truth (int, optional): Expected file ID containing the answer (for validation).

    Returns:
        int: File ID identified as containing the best answer, or zero if not found.
    """
    contents = {}

    # Load content from all files
    for file_id in file_ids:
        if category == "faq":
            contents[str(file_id)] = ""
            for item in faq_data[str(file_id)]:
                contents[str(file_id)] += f"問：{item['question']}\n"
                contents[str(file_id)] += f"答：{' '.join(item['answers'])}\n\n"
        else:
            filepath = BASE / "reference" / category / f"{file_id}.pdf"
            filepath_ocr = BASE / "reference" / category / f"{file_id}-output.pdf"

            try:
                contents[file_id] = read_pdf(filepath_ocr) if filepath_ocr.exists() else read_pdf(filepath)
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
                continue

    prompt1 = generate_prompt_1(question, contents)

    chat = model.start_chat()
    response = chat.send_message(prompt1)
    response_text1 = response.text

    response = chat.send_message(PROMPT2)
    response_text2 = response.text

    number = 0
    try:
        number = int(response_text2)
    except ValueError:
        print(f"'{response_text2}' is not an int。")

    if (ground_truth is not None) and number != ground_truth:
        print("Wrong answer detected.")
        print(f"{question=}")
        print("prompt1")
        print(prompt1)
        print("response_text1")
        print(response_text1)
        print("response_text2")
        print(response_text2)
        print(f"{number=}")
        print(f"{ground_truth=}")
        print("=========================================================")

    return number

def calculate_accuracy(results, ground_truths):
    """
    Calculates accuracy of predicted results against known ground truths.

    Args:
        results (dict): Dictionary of results keyed by question ID.
        ground_truths (dict): Dictionary of ground truths keyed by question ID.

    Returns:
        float: Accuracy of predictions as a percentage.
    """
    correct = 0
    total = 0

    for qid, truth in ground_truths.items():
        if qid not in results:
            continue

        if results[qid] == truth:
            correct += 1

        total += 1

    return correct / total if total != 0 else 0

def load_progress(filename):
    """
    Loads progress from a JSON file if it exists.

    Args:
        filename (str): Path to the progress file.

    Returns:
        dict: Dictionary containing the question IDs and retrieved answers.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return {entry["qid"]: entry["retrieve"] for entry in data["answers"]}
    return {}

def save_progress(filename, data):
    """
    Saves progress to a JSON file.

    Args:
        filename (str): Path to save the progress file.
        data (dict): Dictionary containing question IDs and retrieved answers.
    """
    answers = [{"qid": qid, "retrieve": retrieve} for qid, retrieve in data.items()]
    with open(filename, 'w') as f:
        json.dump({"answers": answers}, f, indent=4)

progress_file = 'progress.json'
results = load_progress(progress_file)

# Filter out questions for specific conditions
question_data["questions"] = [ question for question in question_data["questions"] if question['category'] != 'faq' ]

for question in tqdm(question_data["questions"]):
    qid = question["qid"]
    if qid in results:  # Skip if already processed
        continue
    question_text = question["query"]
    file_ids = question["source"]

    result = analyze_files_in_batch(question_text, file_ids, question['category'], None)
    results[qid] = result
    save_progress(progress_file, results)

    sleep(8)

