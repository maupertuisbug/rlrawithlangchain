from rag import MainRagAgent 



mra = MainRagAgent()
mra.get_paper_base()
mra.create_faiss_vs()
t = mra.get_titles()
print(len(t))
for i in t:
    print(i)