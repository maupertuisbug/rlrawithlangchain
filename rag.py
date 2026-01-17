from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        self.code_model = SentenceTransformer("microsoft/codebert-base")
        self.hf_embeddings_for_text = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.hf_embeddings_for_code = HuggingFaceEmbeddings(model_name="microsoft/codebert-base")
        self.local_vs = None
        self.chunks = []
        self.folder_path = "papers"
        self.index_for_text = None 
        self.index_for_code = None
        self.prompt_template_for_title = PromptTemplate(
                                        input_variables=["context"],
                                        template="""
                                    You are a research assistant.

                                    Task: Extract the paper title from the text below.

                                    Rules:
                                    1. If a title is clearly present, output ONLY the title.
                                    2. If no title is present, output EXACTLY: NO_TITLE
                                    3. Do NOT output anything else (no explanation, no extra text).
                                    4. Titles should be output in Title Case if possible.
                                    Text:
                                    {context}
                                    """
                                    )


    def get_paper_base(self):
        get_papers()

    def create_faiss_vs(self):
        splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", " ", ""]
                    )
        pdf_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            text = ""
            doc = fitz.open(pdf_file)
            text = doc[0].get_text()
            chunks = splitter.split_text(text)
            for chunk in chunks:
                self.chunks.append({
                    "text": chunk,
                    "metadata": {"source": os.path.basename(pdf_file)}
                })

        texts_to_embed = [c["text"] for c in self.chunks]
        vector_store = self.hf_embeddings_for_text.embed_documents(texts_to_embed)
        vec_array = np.array(vector_store).astype("float32")     
        dimension = vec_array.shape[1]
        self.index_for_text = faiss.IndexHNSWFlat(dimension, 32)
        faiss.normalize_L2(vec_array)
        self.index_for_text.add(vec_array)          

        documents = [Document(page_content=c["text"], metadata=c["metadata"]) for c in self.chunks]
        vector_store_doc = FAISS.from_documents(
                            documents,
                            self.hf_embeddings_for_text)
        self.retriever_text = vector_store_doc.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def get_titles(self):
        self.llm = ChatOllama(model="llama3.1")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever_text,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt_template_for_title}
        )
        titles = []
        for c in self.chunks:
            result = qa_chain({"query": "Do as follow.", 
                               "context" : c})
            titles.append(result['result'])

        return titles


