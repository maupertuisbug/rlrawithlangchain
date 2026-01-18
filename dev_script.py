from rag import MainRagAgent 
from langchain_core.documents import Document
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings


mra = MainRagAgent()
mra.get_paper_base()
mra.create_faiss_vs()
t = mra.get_summary()
tc = mra.get_code()

'''
Dictionary doc wth keys - Title, Summary, Prereq, Method, Theory, Baselines, Results
'''

final_content = []

for item in t:
    title = item["Title"]
    summary = item["Summary"]
    code_dict = next((d for d in tc if d.get("title") == title))
    code = code_dict["Code"]

    final_content.append(
        {
            "Title" : title,
            "Summary" : summary, 
            "Code" : code,
            "Method" : item["Method"],
            "Result" : item["Result"],
            "Performance" : item["Performance"]
        }
    )

### Create documents 
documents = [Document(page_content=f"{c['Title']}\n\n{c['Summary']}", metadata={"Code" : c["Code"], "Method" : c["Method"], "Result" : c["Result"], "Performance" : c["Performance"]}) for c in final_content]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store =  FAISS.from_documents(
                            documents,
                            embeddings)



looking_for = "Reinforcement Learning from Passive Data via Latent Intentions"

code = 0

query_string = "Retrieve me summary of the paper with title -" + str(looking_for)

results = vector_store.similarity_search(query_string, k=2)
print(results[0].metadata['Method'])
