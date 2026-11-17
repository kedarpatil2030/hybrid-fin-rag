import os
import pickle
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==========================================
# STEP 0: ENVIRONMENT & CREDENTIAL SETUP
# ==========================================
# Load environment variables (OPENAI_API_KEY, PINECONE_API_KEY) from the local .env file
load_dotenv()

# Ensure both required API keys are available before initializing clients
if not os.environ.get("OPENAI_API_KEY") or not os.environ.get(
    "PINECONE_API_KEY"
):
  raise ValueError("API Keys missing! Please check your .env file.")

# Target Pinecone index name (must match the index created in ingest.py)
index_name = "eicher-financial-rag"

# ==========================================
# STEP 1: CONNECT TO DENSE (PINECONE) RETRIEVER
# ==========================================
print("Connecting to existing Pinecone index...")
# Initialize the exact same embedding model used during ingestion so vector spaces match
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect directly to the pre-populated Pinecone index without re-uploading documents
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

# Configure semantic similarity search to return the top 3 matching chunks
vector_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# ==========================================
# STEP 2: INITIALIZE SPARSE (BM25) RETRIEVER
# ==========================================
print("Loading local chunks for BM25...")
# Verify that ingest.py has already generated the serialized document chunks file
if not os.path.exists("chunks.pkl"):
  raise FileNotFoundError(
      "chunks.pkl not found! Run 'python ingest.py' first."
  )

# Load the saved Document chunks from disk into memory
with open("chunks.pkl", "rb") as f:
  document_chunks = pickle.load(f)

# Build an in-memory BM25 keyword index to catch exact financial terms, numbers, and names
bm25_retriever = BM25Retriever.from_documents(document_chunks)
bm25_retriever.k = 3  # Return top 3 keyword matches

# ==========================================
# STEP 3: BUILD HYBRID (ENSEMBLE) RETRIEVER
# ==========================================
# Combine sparse keyword search (BM25) and dense semantic search (Pinecone) using Reciprocal Rank Fusion
# Equal 0.5 / 0.5 weights balance exact term matching with conceptual understanding
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

# ==========================================
# STEP 4: CONFIGURE PROMPT TEMPLATE & LLM
# ==========================================
# Define a strict grounding prompt so the LLM relies solely on retrieved report context
template = """You are a financial analyst assistant. Answer the user's question accurately using ONLY the provided context from the financial results report. If you do not know the answer based on the context, state clearly that the information is missing.

Context:
{context}

Question:
{question}

Detailed Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize GPT-4o-mini with temperature=0.0 for deterministic, factual financial answers
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


# ==========================================
# STEP 5: ASSEMBLE LCEL RAG PIPELINE
# ==========================================
# Helper function to join retrieved Document objects into a single formatted context string
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


# Construct the LangChain Expression Language (LCEL) execution chain:
# 1. Fetch context via hybrid_retriever and pass the raw user question through unchanged
# 2. Inject context and question into the prompt template
# 3. Send formatted prompt to the LLM
# 4. Parse the model's response object into a clean plain-text string
rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# STEP 6: INTERACTIVE CLI Q&A LOOP
# ==========================================
print("\nReady! Type your question below (or type 'exit' to quit).\n")

while True:
  # Prompt user for a question and strip accidental leading/trailing whitespace
  user_query = input("Ask a question: ").strip()

  # Exit condition to cleanly terminate the CLI loop
  if user_query.lower() in ["exit", "quit", "q"]:
    print("Exiting...")
    break

  # Skip empty inputs without invoking the chain
  if not user_query:
    continue

  print("\nSearching and generating answer...")
  # Execute the hybrid retrieval + generation chain for the current query
  response = rag_chain.invoke(user_query)

  print("\n--- Response ---")
  print(response)
  print("-" * 50 + "\n")