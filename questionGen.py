from haystack.nodes import QuestionGenerator
import requests
from bs4 import BeautifulSoup
url = "https://kmit.in/" 
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
paragraphs = soup.find_all("p")
text_from_website = " ".join([para.get_text() for para in paragraphs])
qg = QuestionGenerator()
# print("RESULT: ")
result = qg.generate(text_from_website)
# print(result)
from haystack.document_stores import InMemoryDocumentStore
# from haystack.pipelines import DocumentSearchPipeline
from haystack.schema import Document
document_store = InMemoryDocumentStore(use_bm25=True)
# print("DOCUMENT: ")
document = Document(content=text_from_website)
# print(document.content)
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2", tokenizer="deepset/bert-base-cased-squad2")
print("Questions are saved!")
