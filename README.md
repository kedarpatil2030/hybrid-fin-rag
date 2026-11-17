# Hybrid-Fin-RAG: Enterprise Financial Document Q&A 📈🤖

An enterprise-grade, local-first **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Pinecone Serverless**, and **Hybrid Search (Dense + Sparse Retrieval)**. Designed to query dense financial earnings reports and corporate disclosures accurately (demonstrated here using Eicher Motors financial results).

---

## 🛠️ Architecture & Tech Stack

* **Orchestration:** LangChain (LCEL - LangChain Expression Language)
* **Vector Database:** Pinecone (Serverless index with Cosine similarity)
* **Hybrid Search (Ensemble):** 
  * *Dense Semantic Search:* OpenAI Embeddings (`text-embedding-3-small`) via Pinecone.
  * *Sparse Keyword Search:* BM25 (`rank-bm25`) for exact financial metrics, part codes, and numerical lookups.
* **LLM Generation:** OpenAI `gpt-4o-mini` (configured with strict grounding prompts to minimize hallucinations).
* **Document Loading:** PyPDFLoader / Recursive Character Text Splitting.

---

## 📂 Project Structure

```text
hybrid-fin-rag/
├── pdf/                      # Directory for source financial PDF reports
│   └── eicher_motors_financial_results.pdf
├── .env                      # Local environment variables (API keys - ignored in git)
├── .gitignore                # Excludes sensitive data, venv, and local caches
├── chunks.pkl                # Local serialized chunks for BM25 keyword matching
├── ingest.py                 # One-time script to parse PDF, chunk, and upsert to Pinecone
├── main.py                   # Interactive CLI chat loop leveraging hybrid search
└── README.md                 # Project documentation
