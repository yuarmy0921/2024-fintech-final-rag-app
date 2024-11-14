import google.generativeai as genai
from pathlib import Path
from time import sleep
from tqdm import tqdm
import json
import os, sys
import pdfplumber

BASE = Path("./final_project_dataset")
PROMPT2 = """請再重新想一次。如果你認同之前的判斷，請告訴我最能回答此問題的文本編號。
如果你不認同上述判斷，請再次判斷後，選出能回答此問題的文本編號。
記得，只會有正好一個答案。如果有多個可能的答案，選一個最合適的。回應編號即可，不要有其他文字。"""
API_KEY = ""
genai.configure(api_key = API_KEY)

faq_data = json.loads((BASE / "reference" / "faq" / "pid_map_content.json").read_text())
question_data = json.loads((BASE / "dataset" / "preliminary" / "questions_example.json").read_text())
ground_truths = json.loads((BASE / "dataset" / "preliminary" / "ground_truths_example.json").read_text())["ground_truths"]
ground_truths = {item['qid']: item['retrieve'] for item in ground_truths}

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-1.5-flash')
generation_config = genai.GenerationConfig(temperature=0.0, top_p=0.5, max_output_tokens=1024)

def read_pdf(pdf_loc, page_infos: list = None):
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
    prompt = f"這是一個問題：「{question}」。\n以下是幾個檔案中的文本，請逐一判斷每個文本是否包含可以回答這個問題的資訊，並回答「是」或「否」。如果回答「是」，請提取相關的句子並解釋為什麼這個文本可以回應問題。\n\n"
    for key, content in contents.items():
        prompt += f"文本 {key}：\n```\n{content}\n```\n\n\n"

    document_list_str = ','.join([f"文本{key}" for key in contents.keys()])
    prompt += f"請詳細閱讀上列文本（{document_list_str}），並討論每個文本是否可以回答問題：「{question}」，以及為什麼。"
    return prompt

# def generate_prompt_2(question, contents, last_result):
#     prompt = f"這是一個問題：「{question}」。\n以下是幾個檔案中的文本，請逐一判斷每個文本是否包含可以回答這個問題的資訊。\n\n"
#     for key, content in contents.items():
#         prompt += f"文本 {key}：\n```\n{content}\n```\n\n\n"

#     prompt += f"""這是之前我收到的回應：
# ```
# {last_result}
# ```

# 如果你認同上面的判斷，請告訴我最能回答此問題的文本編號。
# 回應編號即可，不要有其他文字。
# """

#     return prompt

# Function to analyze files in batch
def analyze_files_in_batch(question, file_ids, category, ground_truth=None):
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

    if number != ground_truth:
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
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
            results = {int(qid): result for qid, result in results.items()}
    return {}

def save_progress(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

progress_file = 'progress.json'
results = load_progress(progress_file)


for question in tqdm(question_data["questions"]):
    qid = question["qid"]
    if qid in results:  # Skip if already processed
        continue
    question_text = question["query"]
    file_ids = question["source"]

    result = analyze_files_in_batch(question_text, file_ids, question['category'], ground_truths[qid])
    results[qid] = result
    save_progress(progress_file, results)

    # Calculate accuracy
    accuracy = calculate_accuracy(results, ground_truths)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    sleep(8)
