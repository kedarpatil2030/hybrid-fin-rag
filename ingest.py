import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# ==========================================
# STEP 0: ENVIRONMENT & CREDENTIAL SETUP
# ==========================================
# Load environment variables from the local .env file
load_dotenv()

# Verify that necessary API keys are present before running the script
if not os.environ.get("OPENAI_API_KEY") or not os.environ.get(
    "PINECONE_API_KEY"
):
  raise ValueError("API Keys missing! Please check your .env file.")

# ==========================================
# STEP 1: LOAD THE FINANCIAL PDF DOCUMENT
# ==========================================
# Define the absolute path to your target PDF report
pdf_path = r"D:\Coding\AI Project\RAG\hybrid-fin-rag\pdf\eicher_motors_financial_results.pdf"
print("Loading PDF document...")

# Initialize the PDF loader and extract all raw pages into LangChain Document objects
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

# ==========================================
# STEP 2: CHUNK THE DOCUMENT
# ==========================================
# Split the large document text into smaller, overlapping chunks suitable for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
document_chunks = text_splitter.split_documents(raw_documents)
print(f"Split document into {len(document_chunks)} chunks.")

# ==========================================
# STEP 3: SAVE LOCAL CACHE FOR BM25 KEYWORD SEARCH
# ==========================================
# Serialize and save the chunks locally to disk. 
# This file will be loaded by main.py so BM25 can build its keyword index instantly without re-reading the PDF.
with open("chunks.pkl", "wb") as f:
  pickle.dump(document_chunks, f)
print("Saved chunks locally to chunks.pkl.")

# ==========================================
# STEP 4: INITIALIZE PINECONE CONNECTION
# ==========================================
# Create a Pinecone client instance using your API key from the environment
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "eicher-financial-rag"
dimension = 1536  # Dimension size corresponding to text-embedding-3-small

# Check if the Pinecone serverless index exists; if not, create it on AWS us-east-1
if not pc.has_index(index_name):
  pc.create_index(
      name=index_name,
      dimension=dimension,
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
  )

# ==========================================
# STEP 5: EMBED AND UPSERT TO PINECONE
# ==========================================
# Initialize OpenAI's embedding model for generating vector representations
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Upserting document chunks to Pinecone index (one-time upload)...")
# Convert all document chunks into vectors and upload them to your Pinecone index
PineconeVectorStore.from_documents(
    documents=document_chunks, embedding=embeddings, index_name=index_name
)
print(
    "Ingestion complete! You can now run main.py to ask unlimited questions."
)