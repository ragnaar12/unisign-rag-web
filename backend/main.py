from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

app = FastAPI()

# CORS pour autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# Charger le contexte
with open("backend/docs/context.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Diviser le texte
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([content])

# Embeddings + vecteurs
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Ollama LLM
llm = Ollama(model="llama2:7b")

# Chaîne RAG
prompt = PromptTemplate.from_template(
    "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

@app.post("/predict")
async def predict(query: Query):
    answer = qa_chain.run(query.query)
    return {"prediction": answer}




