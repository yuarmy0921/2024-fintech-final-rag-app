import pdf2docx 
import pdfplumber

import subprocess

# extract table
import pandas as pd

# extract image
import queue
from spire.doc import *
from spire.doc.common import *
from io import BytesIO
from PIL import Image

# image to text
import cv2
import pytesseract
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
import vertexai
from vertexai.language_models import TextEmbeddingInput,TextEmbeddingModel

from google.cloud import discoveryengine_v1 as discoveryengine

from langchain_community.graphs import Neo4jGraph

import numpy as np

from pathlib import Path

path = Path(os.path.realpath(__file__))
load_dotenv(str(path.parent.absolute()) + '/.env', override=True)

class PDFProcessor:
    def __init__(self):
        self.raw_dir = 'data/raw'
        self.tmp_dir = 'data/tmp'
        self.cleaned_dir = 'data/cleaned'

    def _extract_text(self, file: str) -> str:
        pdf = pdfplumber.open(file)
        pages = pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  
            text = page.extract_text()
    
            if text:
                pdf_text += text
        pdf.close()

        return pdf_text
    
    # def _extract_text(self, doc: Document) -> str:
    #     return doc.GetText()

    def _extract_img(self, category: str, major_no: str, doc: Document) -> list:
        # Create a Queue object
        nodes = queue.Queue()
        nodes.put(doc)

        # Create a list
        images = []

        while nodes.qsize() > 0:
            node = nodes.get()

            # Loop through the child objects in the document
            for i in range(node.ChildObjects.Count):
                child = node.ChildObjects.get_Item(i)

                # Determine if a child object is a picture
                if child.DocumentObjectType == DocumentObjectType.Picture:
                    picture = child if isinstance(child, DocPicture) else None
                    dataBytes = picture.ImageBytes

                    # Add the image data to the list 
                    images.append(dataBytes)
                
                elif isinstance(child, ICompositeObject):
                    nodes.put(child if isinstance(child, ICompositeObject) else None)

        for i, image in enumerate(images):
            img = Image.open(BytesIO(image))
            ext = img.format.lower()
            path = f'{self.cleaned_dir}/{category}/{major_no}-{i}.{ext}'
            img.save(path)

            images[i] = path

        return images

    def _extract_text_from_img(self):
        file = f'{self.cleaned_dir}/finance/1-2.png'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

        img = Image.open(file)
        print(pytesseract.image_to_string(img, lang="chi_tra+eng"))

    def _extract_table(self):
        pass

    def _apply_ocr(self, src: str, dst: str):
        try:
            # Run the ocrmypdf command
            subprocess.run(
                ["ocrmypdf", "-l", "chi_tra", "--pdf-renderer=sandwich", "--force-ocr", "-q", src, dst],
                check=True
            )
            print(f"OCR completed for {src}")
        except subprocess.CalledProcessError as e:
            print(f"Error during OCR for {src}: {e}")

    def extract_content(self, category: str, filename: str):
        major_no = filename[:-4]
        path_to_raw = f'{self.raw_dir}/{category}/{filename}'
        path_to_cleaned = f'{self.cleaned_dir}/{category}/{filename}'
        # path_to_docx = f'{self.tmp_dir}/{category}/{major_no}.docx'

        if os.path.isfile(path_to_cleaned):
            text = self._extract_text(path_to_cleaned)
            return text

        text = self._extract_text(path_to_raw)

        if not text.strip():
            self._apply_ocr(path_to_raw, path_to_cleaned)
            text = self._extract_text(path_to_cleaned)

        return text  # 返回萃取出的文本

        # convert pdf -> docx
        '''
        pdf2docx.parse(path_to_file, path_to_docx)

        doc = Document()

        doc.LoadFromFile(path_to_docx)
        text = self._extract_text(doc)
        images = self._extract_img(doc=doc, category=category, major_no=major_no)
        '''

class GeminiAPI:
    def __init__(self, model='models/embedding-004'):
        key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=key)

        self.model = model

    def get_text_embedding(self, args: dict, task='retrieval_document'):
        if task == 'retrieval_document':
            embedding = genai.embed_content(
                model=self.model, 
                content=args['content'], 
                task_type='retrieval_document',
                title=args['title']
            )['embedding']
            
        return embedding

class VertexAIAPI:
    def __init__(self):
        PROJECT_ID = os.getenv('PROJECT_ID')
        REGION = os.getenv('REGIOM')
        MODEL_ID = 'text-multilingual-embedding-002'

        vertexai.init(project=PROJECT_ID, location=REGION)

        self.model = TextEmbeddingModel.from_pretrained(MODEL_ID)

    def get_text_embedding_batch(self, title: str, text_list: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_DOCUMENT'
        inputs = [TextEmbeddingInput(text, task, title) for text in text_list]
        length = len(inputs)
        embeddings = []

        for text_input in inputs:
            embed = self.model.get_embeddings([text_input])
            embeddings += embed

        # for i in range(length // 10):
        #     inputs_seg = inputs[i*10:(i+1)*10]
        #     embeds = self.model.get_embeddings(inputs_seg)
        #     embeddings += embeds

        # inputs_seg = inputs[(length//10)*10:]

        # if inputs_seg != []:
        #     embeds = self.model.get_embeddings(inputs_seg)
        #     embeddings += embeds

        return [embedding.values for embedding in embeddings]
    
    def get_query_embedding_batch(self, query_list: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_QUERY'
        inputs = [TextEmbeddingInput(query, task) for query in query_list]
        length = len(inputs)
        embeddings = self.model.get_embeddings(inputs)

        return [embedding.values for embedding in embeddings]
    
    def find_best_document(self, query: list[float], embeddings: list[list[float]]) -> int:
        dot_products = np.dot(np.stack(embeddings), query)
        # print('dot:', dot_products)
        idx = np.argmax(dot_products)

        return idx
    
class Reranker:
    def __init__(self):
        PROJECT_ID = os.getenv('PROJECT_ID')
        self.client = discoveryengine.RankServiceClient()
        self.ranking_config = self.client.ranking_config_path(
            project=PROJECT_ID,
            location="global",
            ranking_config="default_ranking_config",
        )

    def rerank(self, category: str, query: str, records: list):
        ranking_records = [
            discoveryengine.RankingRecord(
                id=record['id'],
                title=category,
                content=record['content']
            ) for record in records
        ]
        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model="semantic-ranker-512@latest",
            top_n=10,
            query=query,
            records=ranking_records,
            ignore_record_details_in_response=True
        )

        # print(type(self.client.rank(request=request)))
        response = self.client.rank(request=request)

        # Handle the response
        return response.records

class Neo4jAPI:
    def __init__(self):
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
        self.kg = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD, 
            database=NEO4J_DATABASE
        )

        self.chunk_size = 20000
        self.chunk_overlap = 10000

        # self.kg.query("CREATE CONSTRAINT unique_doc_finance IF NOT EXISTS FOR (n:Document:finance) REQUIRE (n.no) IS UNIQUE")
        # self.kg.query("CREATE CONSTRAINT unique_doc_insurance IF NOT EXISTS FOR (n:Document:insurance) REQUIRE (n.no) IS UNIQUE")
        # self.kg.query("CREATE CONSTRAINT unique_doc_faq IF NOT EXISTS FOR (n:Document:faq) REQUIRE (n.no) IS UNIQUE")

        # self.kg.query("CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (n:Chunk:finance) REQUIRE (n.seq_no) IS UNIQUE")
        # self.kg.query("CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (n:Chunk:insurance) REQUIRE (n.seq_no) IS UNIQUE")
        # self.kg.query("CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (n:Chunk:faq) REQUIRE (n.seq_no) IS UNIQUE")
        # self.kg.query("CREATE CONSTRAINT unique_next IF NOT EXISTS FOR (n:Chunk)-[r:next_chunk]->(m:Chunk) REQUIRE (r) IS UNIQUE")

    def _get_chunk_seq_no(self, doc_no: int, seq: int):
        return f'{doc_no}-{seq}_{self.chunk_size}-{self.chunk_overlap}'
    
    def set_chunk_size(self, chunk_size: int):
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int):
        self.chunk_overlap = chunk_overlap
    
    def query(self, query, params):
        return self.kg.query(query, params)
    
    def merge_doc(self, category: str, no: int):
        query = f"""
        MERGE (:Document:{category} {{no: $no}})
        """

        self.kg.query(query, params={'no': no})

    def merge_chunk(self, category: str, doc_no: int, seq: int, text: str, embedding: list):
        seq_no = self._get_chunk_seq_no(doc_no, seq)
        query = f"""
        MERGE (chunk:Chunk:{category} {{seq_no: $seq_no}})
        ON CREATE SET
            chunk.text = $text,
            chunk.embed_retrieval_document = $embedding
        ON MATCH SET
            chunk.text = $text,
            chunk.embed_retrieval_document = $embedding
        """

        self.kg.query(query, params={'seq_no': seq_no, 'text': text, 'embedding': embedding})

    def link_chunk_batch(self, category: str, doc_no: int, chunks: int):
        query = f"""
        MATCH (doc:Document:{category} {{no: $no}})
        MATCH (chunk:Chunk:{category} {{seq_no: $seq_no}})
        MERGE (doc)-[:has_root {{chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}}}]->(chunk)
        """
        last_seq_no = self._get_chunk_seq_no(doc_no, 0)

        self.kg.query(query, params={'no': doc_no, 'seq_no': last_seq_no})

        query = f"""
        MATCH (chunk1:Chunk:{category} {{seq_no: $src}})
        MATCH (chunk2:Chunk:{category} {{seq_no: $dst}})
        MERGE (chunk1)-[:next_chunk]->(chunk2)
        """

        for i in range(1, chunks):
            seq_no = self._get_chunk_seq_no(doc_no, i)
            self.kg.query(query, params={'src': last_seq_no, 'dst': seq_no})
            last_seq_no = seq_no

    def get_doc_embeddings(self, category: str, no: int):
        query = f"""
        MATCH (:Document:{category} {{no: $no}})-[:has_root {{chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}}}]->(chunk0:Chunk:{category})
        RETURN chunk0.embed_retrieval_document as embedding
        """

        embeddings = [self.kg.query(query, params={'no': no})[0]['embedding']]

        query = f"""
        MATCH (:Document:{category} {{no: $no}})-[:has_root {{chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}}}]->(chunk0:Chunk:{category})-[:next_chunk*]->(chunk:Chunk:{category})
        RETURN chunk.embed_retrieval_document as embedding
        """
        
        embeddings += [item['embedding'] for item in self.kg.query(query, params={'no': no})]
        
        return embeddings
    
    def get_chunk_text(self, category: str, no: int, seq: int):
        seq_no = self._get_chunk_seq_no(no, seq)
        query = f"""
        MATCH (chunk:Chunk:{category} {{seq_no: $seq_no}}) 
        RETURN chunk.text as text
        """
        text = self.kg.query(query, params={'seq_no': seq_no})[0]['text']

        return text
    
    def get_doc_text(self, category: str, no: int):
        query = f"""
        MATCH (:Document:{category} {{no: $no}})-[:has_root {{chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}}}]->(chunk0:Chunk:{category})
        RETURN chunk0.text as text
        """

        text = [self.kg.query(query, params={'no': no})[0]['text']]

        query = f"""
        MATCH (:Document:{category} {{no: $no}})-[:has_root {{chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}}}]->(chunk0:Chunk:{category})-[:next_chunk*]->(chunk:Chunk:{category})
        RETURN chunk.text as text
        """
        
        text += [item['text'] for item in self.kg.query(query, params={'no': no})]
        
        return text
    
    def set_doc_embeddings(self, category: str, no: int, embeddings: list):
        query = f"""
        MATCH (chunk:Chunk:{category} {{seq_no: $seq_no}})
        SET chunk.embed_retrieval_document = $embed
        """

        for i, embed in enumerate(embeddings):
            seq_no = self._get_chunk_seq_no(no, i)
            self.kg.query(query, params={'seq_no': seq_no, 'embed': embed})