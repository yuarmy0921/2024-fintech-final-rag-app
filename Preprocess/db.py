import json
import sys

sys.path.append("..")
from module.util import *
sys.path.remove("..")

from langchain.text_splitter import RecursiveCharacterTextSplitter

processor = PDFProcessor()
vertex_ai = VertexAIAPI()
# gemini = GeminiAPI()
neo4j = Neo4jAPI()

chunk_size = 20000
chunk_overlap = 10000

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = len,
    separators = ["\n\n", "\r\n", "\n", "。", "，", ""],
    is_separator_regex = False,
)

neo4j.set_chunk_size(chunk_size)
neo4j.set_chunk_overlap(chunk_overlap)

def create_doc(category: str, no: int):
    """Add the document information to the database."""
    print(f'Document {no}:')
    print(' extract content...')
    text = processor.extract_content(category, f'{no}.pdf')

    print(' create chunks...')
    chunks = text_splitter.split_text(text)

    print(' get text embedding ...')
    # embeddings = [gemini.get_text_embedding({'title': category, 'content': chunk}) for chunk in chunks]
    embeddings = vertex_ai.get_text_embedding_batch(category, chunks)

    print(' create doc and links...')
    neo4j.merge_doc(category, no)

    for j, chunk in enumerate(chunks):
        neo4j.merge_chunk(category, no, j, chunk, embeddings[j])

    neo4j.link_chunk_batch(category, no, len(chunks))

def create_files_from_json(path='data/raw/faq/pid_map_content.json'):
    """Convert .txt files from the given .json file"""
    with open(path, 'r') as f:
        struct = json.load(f)

    cleaned_path = 'data/cleaned/faq'

    for no, content in struct.items():
        data = ''
        for qa in content:
            data = data + 'Q: ' + qa['question'] + '\n'

            for answer in qa['answers']:
                data = data + 'A: ' + answer + '\n'
        
            data += '\n'

        data = data[:-2]
        with open(f'{cleaned_path}/{no}.txt', 'w') as f:
            f.write(data)

# def update_embeddings(category: str, no: int):
#     print(f'Document {no}')
#     text = neo4j.get_doc_text(category, no)
#     embeddings = [gemini.get_text_embedding({'title': category, 'content': chunk}) for chunk in text]
#     neo4j.set_doc_embeddings(category, no, embeddings)


if __name__ == '__main__':
    create_files_from_json()
    category = 'finance'
    print('================ Finance ==================')
    for i in range(1035):
        # update_embeddings(category, i)
        create_doc(category, i)

    category = 'insurance'
    print('================ Insurance ==================')
    for i in range(1, 644):
        # update_embeddings(category, i)
        create_doc(category, i)

    category = 'faq'
    cleaned_path = 'data/cleaned/faq'
    print('================ FAQ ==================')
    for no in range(617):
        # update_embeddings(category, i)
        print(f'Document {no}:')
        with open(f'{cleaned_path}/{no}.txt', 'r') as f:
            text = f.read()

        print(' create chunks...')
        chunks = text_splitter.split_text(text)

        print(' get text embedding ...')
        embeddings = vertex_ai.get_text_embedding_batch(category, chunks)

        print(' create doc and links...')
        neo4j.merge_doc(category, no)

        for j, chunk in enumerate(chunks):
            neo4j.merge_chunk(category, no, j, chunk, embeddings[j])

        neo4j.link_chunk_batch(category, no, len(chunks))

    