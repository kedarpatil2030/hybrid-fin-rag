import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if not os.environ.get("OPENAI_API_KEY") or not os.environ.get(
    "PINECONE_API_KEY"
):
  raise ValueError("API Keys missing! Please check your .env file.")

# 1. Load & Split PDF
pdf_path = r"D:\Coding\AI Project\RAG\hybrid-fin-rag\pdf\eicher_motors_financial_results.pdf"
print("Loading PDF document...")
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
document_chunks = text_splitter.split_documents(raw_documents)
print(f"Split document into {len(document_chunks)} chunks.")

# 2. Save chunks locally for fast BM25 keyword retrieval later
with open("chunks.pkl", "wb") as f:
  pickle.dump(document_chunks, f)
print("Saved chunks locally to chunks.pkl.")

# 3. Setup Pinecone & Upsert Vectors
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "eicher-financial-rag"
dimension = 1536

if not pc.has_index(index_name):
  pc.create_index(
      name=index_name,
      dimension=dimension,
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
  )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Upserting document chunks to Pinecone index (one-time upload)...")
PineconeVectorStore.from_documents(
    documents=document_chunks, embedding=embeddings, index_name=index_name
)
print(
    "Ingestion complete! You can now run main.py to ask unlimited questions."
)