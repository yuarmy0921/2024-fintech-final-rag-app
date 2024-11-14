import json
import argparse, sys

sys.path.append("..")
from module.util import *
sys.path.remove("..")

vertex_ai = VertexAIAPI()
chunk_size = 20000
chunk_overlap = 10000

neo4j = Neo4jAPI()
neo4j.set_chunk_size(chunk_size)
neo4j.set_chunk_overlap(chunk_overlap)

reranker = Reranker()

def find_best_document(query: list[float], embeddings: list[list[float]]) -> int:
        dot_products = np.dot(np.stack(embeddings), query)
        idx = np.argmax(dot_products)

        return idx

def find_best_document_top_k(query: list[float], embeddings: list[list[float]], k=3) -> int:
        k = min(k, len(embeddings))
        dot_products = np.dot(np.stack(embeddings), query)
        topk = np.argpartition(dot_products, -k)[-k:]

        return topk

def get_ans(query: list[float], source: list[int]) -> int:
    num_of_chunks = []
    embed_doc_search = []

    for s in source:
        embeddings = neo4j.get_doc_embeddings(question['category'], s)
        num_of_chunks.append(len(embeddings))
        embed_doc_search += embeddings

    idx = find_best_document(query, embed_doc_search)
    idx += 1

    for i, num in enumerate(num_of_chunks):
        idx -= num

        if idx <= 0:
            return question['source'][i]
        
def get_ans_rerank(category: str, query_text: str, query_embedding: list[float], source: list[int]) -> int:
    num_of_chunks = []
    embed_doc_search = []

    for s in source:
        embeddings = neo4j.get_doc_embeddings(question['category'], s)
        num_of_chunks.append(len(embeddings))
        embed_doc_search += embeddings

    indices = find_best_document_top_k(query_embedding, embed_doc_search)
    source_candidate = []

    for idx in indices:
        idx += 1
        i = 0
        while i < len(num_of_chunks):
            num = num_of_chunks[i]

            if idx-num <= 0:
                source_candidate.append([question['source'][i], idx-1])
                break
            else:
                idx -= num
                i += 1
                
    # source_candidate
    records = []

    for candidate in source_candidate:
        text = neo4j.get_chunk_text(category, candidate[0], candidate[1])
        record = {'id': f'{candidate[0]} {candidate[1]}', 'title': category, 'content': text}
        records.append(record)

    res = reranker.rerank(category, query_text, records)
    score = 0
    doc_no = -1

    for record in res:
        if record.score > score:
            score = record.score
            doc_no = record.id.split()[0]

    return int(doc_no)
        

parser = argparse.ArgumentParser(description='Retrieve answer')
parser.add_argument('--question', type=str, help='Your question json file')
parser.add_argument('--pred', type=str, help='Your output prediction json file')
args = parser.parse_args()

with open(args.question, 'r') as f:
    questions = json.load(f)['questions']

queries = [question['query'] for question in questions]
# with open('query_embeddings_preliminary.json', 'r') as f:
#     query_embeddings = json.load(f)

query_embeddings = []
num_queries = len(queries)
max_batch_size = 250
num_batches = num_queries // max_batch_size

for i in range(num_batches):
    query_embeddings += vertex_ai.get_query_embedding_batch(queries[i*max_batch_size:(i+1)*max_batch_size])
query_embeddings += vertex_ai.get_query_embedding_batch(queries[num_batches*max_batch_size:])

# with open('data/preliminary/questions_example.json', 'r') as f:
#     questions = json.load(f)['questions']
# with open('query_embeddings.json', 'r') as f:
#     query_embeddings = json.load(f)

answers = []
for i, question in enumerate(questions):
    print('qid:', question['qid'], end=', ')
    ans = get_ans(query_embeddings[i], question['source'])
    # ans = get_ans_rerank(
    #     category=question['category'], 
    #     query_text=question['query'],
    #     query_embedding=query_embeddings[i],
    #     source=question['source']
    # )
    print('ans:', ans)
    answers.append({'qid': question['qid'], 'retrieve': ans})
    print('------------------------------------')

with open(args.pred, 'w') as f:
    json.dump({'answers': answers}, f)

# with open('data/preliminary/vector-rerank-20000-10000.json', 'w') as f:
#     json.dump({'answers': answers}, f)

