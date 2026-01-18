from rag import MainRagAgent 
from rag_with_pinecone import MainRagAgentwithPinecone
from langchain_core.documents import Document
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings




class RLRA:
    def __init__(self):
        self.agent = MainRagAgentwithPinecone("papers-to-everything")
        self.final_content  = []

    def initial_base(self):
        self.agent.get_paper_base()
        self.agent.create_faiss_vs()
        self.content = self.agent.get_summary()
        self.code_content = self.agent.get_code()
    
    def get_content(self):

        for item in self.content:
            title = item["Title"]
            summary = item["Summary"]
            code_dict = next((d for d in self.code_content if d.get("title") == title))
            code = code_dict["Code"]

            self.final_content.append(
                {
                    "Title" : title,
                    "Summary" : summary, 
                    "Code" : code,
                    "Method" : item["Method"],
                    "Result" : item["Result"],
                    "Performance" : item["Performance"]
                }
            )

    def setup_vs(self):
        self.get_content()
        documents = [Document(page_content=f"{c['Title']}\n\n{c['Summary']}", metadata={"Code" : c["Code"], "Method" : c["Method"], "Result" : c["Result"], "Performance" : c["Performance"]}) for c in self.final_content]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store =  FAISS.from_documents(
                                    documents,
                                    embeddings)

    def ask(self, looking_for, type):
        
        query_string = "Retrieve me summary of the paper with title -" + str(looking_for)
        results = self.vector_store.similarity_search(query_string, k=2)
        if type == 1 :
            return results[0].metadata['Code']
        elif type == 2:
            return results[0].metadata['Method']
        elif type == 3:
            return results[0].metadata['Result']
        elif type == 4:
            return results[0].metadata['Performance']
        

