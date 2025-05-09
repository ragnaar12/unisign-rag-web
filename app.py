import os
os.environ['PATH'] += r';C:\Users\A0002393\AppData\Local\Programs\Ollama' # Adjust the path to put the path in you PC

!ollama --version
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_loader = PyPDFLoader("Doc/pdf/theseWBA_English_version.pdf")

# text_loader = TextLoader("path/to/your/document.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(pdf_loader.load()) #+ text_loader.load())
import os
# import torch
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Directory containing PDF files
pdf_directory = "Doc/pdf/"

# Load all PDF files from the directory
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_loader = PyPDFLoader(os.path.join(pdf_directory, filename))
        documents.extend(pdf_loader.load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2")
# BAAI/bge-large-en: Developed by the Beijing Academy of Artificial Intelligence (BAAI), this model is designed for generating high-quality embeddings for tasks like retrieval, classification, clustering, and semantic search 

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cuda"}) # to use GPU

# Extract text content from each Document object
text_contents = [doc.page_content for doc in split_documents]
document_embeddings = embeddings.embed_documents(text_contents)

# Integrate retriever
vector_store = Chroma.from_documents(split_documents, embeddings)
retriever = vector_store.as_retriever()

# import the libraries and set the RAG pipeline
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# llm = LlamaCpp(model_path=model_path)

llm = Ollama(
  model="llama2:7b",
  temperature=0.7,  # Controls the randomness of the model's predictions. Values range from 0 (deterministic) to values > 1 (creative).
  top_p=0.95,  # Implements nucleus sampling, where the model considers the smallest set of tokens whose cumulative probability is at least top_p.
  top_k=40,  # Considers the top_k tokens with the highest probabilities for sampling.
  repeat_penalty=1.2  # Penalizes the model for repeating the same tokens.
)
retriever = vector_store.as_retriever()
prompt_template = PromptTemplate(template="Answer the question based on the following context: {context}\n\nQuestion: {question}\n\nAnswer:")

#Option1: rag_pipeline.run(query) 
"""rag_pipeline.run(query) is straightforward and easy to use for simple queries where only the final result is needed.
It returns the final result directly, which is usually a single string or text output"""


rag_pipeline = RetrievalQA.from_chain_type(
	llm=llm,
	retriever=retriever,
	chain_type="stuff",  # Specify the chain type. the "stuff" chain type will combine these documents into a single input for the language model.
	chain_type_kwargs={"prompt": prompt_template}  # Pass the prompt template
)


# import the libraries and set the RAG pipeline
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# llm = LlamaCpp(model_path=model_path)

llm = Ollama(
  model="llama2:7b",
  temperature=0.7,  # Controls the randomness of the model's predictions. Values range from 0 (deterministic) to values > 1 (creative).
  top_p=0.95,  # Implements nucleus sampling, where the model considers the smallest set of tokens whose cumulative probability is at least top_p.
  top_k=40,  # Considers the top_k tokens with the highest probabilities for sampling.
  repeat_penalty=1.2  # Penalizes the model for repeating the same tokens.
)
retriever = vector_store.as_retriever()
prompt_template = PromptTemplate(template="Answer the question based on the following context: {context}\n\nQuestion: {question}\n\nAnswer:")
# Option 2: Set up QA chain
"""qa_chain  returns a dictionary with multiple keys, allowing access to detailed information such as the main result, source documents, and other metadata.
It provides more flexibility and control, enabling you to access and manipulate various parts of the output.
"""
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True  # Set to True if you want to include source documents in the response
)


#define the query
query = "answer this question beriefly: what the acceleration factors stand for?"
# query = "What are the acceleration factors?"



#Option1: simple and single answer
result = rag_pipeline.run(query)
print(result)   




#Option2: more complete and detailed output
result = qa_chain({"query": query})  # Use the __call__ method to get all outputs
print(result["result"])  # Print the main result
print(result["source_documents"])  # Print the source documents
# Initialize conversation history
conversation_history = []

def ask_question(query):
    global conversation_history
    # Combine conversation history with the new query
    full_query = "\n".join(conversation_history + [f"User: {query}"])
    
    # Get the response from the QA chain
    result = qa_chain({"query": full_query})
    
    # Extract the main result and source documents
    answer = result["result"]
    source_documents = result["source_documents"]
    
    # Update conversation history
    conversation_history.append(f"User: {query}")
    conversation_history.append(f"Assistant: {answer}")
    
    return answer, source_documents

# Example usage
query1 = "When was this thesis defended?"
answer1, sources1 = ask_question(query1)
print("Answer 1:", answer1)
print("Sources 1:", sources1)
query2 = "what the acceleration factors stand for"
answer2, sources2 = ask_question(query2)
print("Answer 2:", answer2)
print("Sources 2:", sources2)
# %pip install PyMuPDF

import os
import fitz  # PyMuPDF has to be installed before this import works
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Directory containing PDF files
pdf_directory = "Doc/pdf/"

# Load all PDF files from the directory
documents = []
metadata = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_loader = PyPDFLoader(os.path.join(pdf_directory, filename))
        loaded_docs = pdf_loader.load()
        documents.extend(loaded_docs)
        
        # Extract metadata manually
        doc = fitz.open(os.path.join(pdf_directory, filename))
        doc_metadata = {
            "title": doc.metadata["title"],
            "author": doc.metadata["author"],
            "subject": doc.metadata["subject"],
            "keywords": doc.metadata["keywords"]
        }
        metadata.append(doc_metadata)
      # Split documents using advanced text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2")
text_contents = [doc.page_content for doc in split_documents]
document_embeddings = embeddings.embed_documents(text_contents)

# Integrate retriever
vector_store = Chroma.from_documents(split_documents, embeddings)
retriever = vector_store.as_retriever()


# Set up the RAG pipeline
llm = Ollama(model="llama2:7b", temperature=0.7, top_p=0.95, top_k=40, repeat_penalty=1.2)
prompt_template = PromptTemplate(template="Answer the question based on the following context: {context}\n\nQuestion: {question}\n\nAnswer:")

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)


# Example usage
query = "What are the acceleration factors?"
result = rag_pipeline.run(query)
print(result)


import os
import fitz  # PyMuPDF
import spacy
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to split text into sentences
def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Directory containing PDF files
pdf_directory = "Doc/pdf/"

# Load all PDF files from the directory
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_loader = PyPDFLoader(os.path.join(pdf_directory, filename))
        documents.extend(pdf_loader.load())

# Extract text content from each Document object
text_contents = [doc.page_content for doc in documents]

# Split text into sentences
sentences = []
for text in text_contents:
    sentences.extend(split_into_sentences(text))

# Use RecursiveCharacterTextSplitter to chunk sentences
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(sentences)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2")
document_embeddings = embeddings.embed_documents([doc.page_content for doc in split_documents])

# Store embeddings in Chroma
vector_store = Chroma.from_documents(split_documents, embeddings)
retriever = vector_store.as_retriever()

# Load the LLM
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the RAG pipeline
prompt_template = "Answer the question based on the following context: {context}\n\nQuestion: {question}\n\nAnswer:"
rag_pipeline = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Example usage
query = "What are the acceleration factors?"
result = rag_pipeline.run(query)
print(result)


#Import the necessary module:
from transformers import BitsAndBytesConfig

#Create the quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

#Integrate the quantization configuration into the model setup
llm = Ollama(model="llama2:7b", temperature=0.7, top_p=0.95, top_k=40, repeat_penalty=1.2, quantization_config=quantization_config)



