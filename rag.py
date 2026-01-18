from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from get_papers import get_papers
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import fitz
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate



class MainRagAgent:
    def __init__(self):
        self.text_model = SentenceTransformer("all-mpnet-base-v2")
        self.code_model = SentenceTransformer("BAAI/bge-m3")
        self.hf_embeddings_for_text = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.hf_embeddings_for_code = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.local_vs = None
        self.chunks = []
        self.folder_path = "papers"
        self.index_for_text = None 
        self.index_for_code = None
        self.titles = []


    def get_paper_base(self):
        get_papers()

    def create_faiss_vs(self):
        splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=10,
                        separators=["\n\n", "\n", " ", ""]
                    )
        pdf_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            text = ""
            doc = fitz.open(pdf_file)
            text = doc[0].get_text()
            title = doc.metadata.get("title")
            self.titles.append(title)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                self.chunks.append({
                    "text": chunk,
                    "metadata": {"source": os.path.basename(pdf_file)},
                })

        documents = [Document(page_content=c["text"], metadata=c["metadata"]) for c in self.chunks]
        vector_store_doc = FAISS.from_documents(
                            documents,
                            self.hf_embeddings_for_text)
        self.retriever_text = vector_store_doc.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        vector_store_code = FAISS.from_documents(
                            documents,
                            self.hf_embeddings_for_code)
        self.retriever_code = vector_store_code.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def get_summary(self):
        self.llm = ChatOllama(model="llama3.1")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever_text,
            return_source_documents=False
        )
        query_string_for_summary = ". Provide a concise summary of the paper with this title. Do not use more than 100 words. Give a single paragraph."
        query_string_for_method = ". Provide the methodology used in the paper with this title. Do not use more than 100 words."
        query_string_for_results = ". What are the main theoritical results of the paper with this title. Give only the main 3 theorems."
        query_string_for_emp = ". Provide the performance with empirical results. Describe the baselines used to compare."
        return_c = []
        for t in self.titles:
            query_string_used = "Title is -" + str(t) + query_string_for_summary
            result_s = qa_chain({"query": query_string_used})
            query_string_used = "Title is -" + str(t) + query_string_for_method
            result_q = qa_chain({"query": query_string_used})
            query_string_used = "Title is -" + str(t) + query_string_for_results
            result_t = qa_chain({"query": query_string_used})
            query_string_used = "Title is -" + str(t) + query_string_for_emp
            result_e = qa_chain({"query": query_string_used})
            return_c.append({"Title" : t,
                             "Summary" : result_s['result'],
                             "Method" : result_q['result'], 
                             "Result" : result_t["result"],
                             "Performance" : result_e['result']})
        return return_c

    def get_code(self):
        self.llm = ChatOllama(model="llama3.1")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever_code,
            return_source_documents=False
        )
        query_string = ". Provide the psuedo code of the method proposed in the paper with this title. Do not use more than 100 lines. Give lines of code. Provide a paragraph on how they tuned the hyperparameters."
        return_c = []
        for t in self.titles:
            query_string_used = "Title is -" + str(t) + query_string
            result = qa_chain({"query": query_string_used})
            return_c.append({"title" : t, "Code" : result['result']})
        return return_c



