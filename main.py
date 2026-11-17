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

load_dotenv()

if not os.environ.get("OPENAI_API_KEY") or not os.environ.get(
    "PINECONE_API_KEY"
):
  raise ValueError("API Keys missing! Please check your .env file.")

index_name = "eicher-financial-rag"

# 1. Connect to EXISTING Pinecone index (No PDF loading or upserting!)
print("Connecting to existing Pinecone index...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
vector_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# 2. Load saved chunks from disk for BM25 keyword retriever
print("Loading local chunks for BM25...")
if not os.path.exists("chunks.pkl"):
  raise FileNotFoundError(
      "chunks.pkl not found! Run 'python ingest.py' first."
  )

with open("chunks.pkl", "rb") as f:
  document_chunks = pickle.load(f)

bm25_retriever = BM25Retriever.from_documents(document_chunks)
bm25_retriever.k = 3

# 3. Combine into Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

# 4. Prompt & LLM
template = """You are a financial analyst assistant. Answer the user's question accurately using ONLY the provided context from the financial results report. If you do not know the answer based on the context, state clearly that the information is missing.

Context:
{context}

Question:
{question}

Detailed Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Interactive Question Loop (Ask multiple questions without restarting)
print("\nReady! Type your question below (or type 'exit' to quit).\n")

while True:
  user_query = input("Ask a question: ").strip()
  if user_query.lower() in ["exit", "quit", "q"]:
    print("Exiting...")
    break
  if not user_query:
    continue

  print("\nSearching and generating answer...")
  response = rag_chain.invoke(user_query)
  print("\n--- Response ---")
  print(response)
  print("-" * 50 + "\n")