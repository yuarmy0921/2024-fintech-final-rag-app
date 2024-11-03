import pdf2docx 
import pdfplumber

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

class PDFProcessor:
    def __init__(self):
        self.raw_dir = 'data/raw'
        self.tmp_dir = 'data/tmp'
        self.cleaned_dir = 'data/cleaned'

    def _extract_text(self, doc: Document) -> str:
        return doc.GetText()

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


        # # Read the image from which text needs to be extracted
        # img = cv2.imread(file)

        # # Convert the image to the grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # Performing OTSU threshold 
        # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # # dilation parameter, bigger means less rectangle
        # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        # # Applying dilation on the threshold image
        # dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

        # # Finding contours
        # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # # Creating a copy of image
        # im2 = gray.copy()


        # cnt_list=[]
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)

        #     # Drawing a rectangle on the copied image
        #     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 5)
        #     cv2.circle(im2, (x,y), 8, (255,255,0), 8)

        #     # Cropping the text block for giving input to OCR
        #     cropped = im2[y:y + h, x:x + w]

        #     # Apply OCR on the cropped image
        #     text = pytesseract.image_to_string(cropped, lang='chi_tra')

        #     cnt_list.append([x,y,text])


        # # This list sorts text with respect to their coordinates, in this way texts are in order from top to down
        # sorted_list = sorted(cnt_list, key=lambda x: x[1])

        # # A text file is created 
        # file = open("recognized.txt", "w+")
        # file.write("")
        # file.close()


        # for x,y,text in sorted_list:
        #     # Open the file in append mode
        #     file = open("recognized.txt", "a")

        #     # Appending the text into the file
        #     file.write(text)
        #     file.write("\n")

        #     # Close the file
        #     file.close()


        # # read image 
        # rgb_image = cv2.resize(im2, (0, 0), fx = 0.4, fy = 0.4)
        # dilation = cv2.resize(dilation, (0, 0), fx = 0.4, fy = 0.4)
        # #thresh1 = cv2.resize(thresh1, (0, 0), fx = 0.4, fy = 0.4)

        # # show the image, provide the window name first
        # #cv2.imshow('thresh1', thresh1)
        # cv2.imshow('dilation', dilation)
        # cv2.imshow('gray', gray)

        # # add wait key. window waits until the user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()


    def _extract_table(self):
        pass

    def extract_content(self, category, filename):
        major_no = filename[:-4]
        path_to_file = f'{self.raw_dir}/{category}/{filename}'
        path_to_docx = f'{self.tmp_dir}/{category}/{major_no}.docx'

        # convert pdf -> docx
        pdf2docx.parse(path_to_file, path_to_docx)

        doc = Document()

        doc.LoadFromFile(path_to_docx)
        text = self._extract_text(doc)
        images = self._extract_img(doc=doc, category=category, major_no=major_no)

    def load(self):
        pass

    def read_pdf(self, page_infos: list = None):
        pdf_loc = 'data/raw/finance/162.pdf'
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容
            table = page.extract_table()
            print(table)
            df = pd.DataFrame(table[1::],columns=table[0])
            print(df)
            if text:
                pdf_text += text
        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本
    
    # def test(self):
    #     import pymupdf4llm
    #     pdf = '/home/yuarmy/vaults/1131-course/fintech/final/2024-fintech-final-rag-app/data/raw/finance/13.pdf'

    #     md_text = pymupdf4llm.to_markdown(pdf)
    #     with open('test.md', 'w') as f:
    #         f.write(md_text)
    #     print(md_text)


class GeminiAPI:
    def __init__(self, model='models/embedding-001'):
        load_dotenv('.env', override=True)

        key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=key)

        self.model = model

    def get_text_embedding(self, args: dict, task='retrieval_document'):
        if task == 'retrieval_document':
            response = genai.embed_content(
            model=self.model, 
            content=args['content'], 
            task_type='retrieval_document',
            title=args['title']
        )
            
        return response['embedding']

if __name__ == '__main__':
    processor = PDFProcessor()
    # processor.read_pdf()
    # processor.extract_content('finance', '162.pdf')
    # processor._extract_text_from_img()
    processor.test()