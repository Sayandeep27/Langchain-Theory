# FlashRank Reranking + LangChain Retrieval Pipeline

---

## 📌 Overview

This project demonstrates **FlashRank-based reranking** integrated with a **LangChain Retrieval Pipeline** using:

* FlashRank Cross‑Encoder reranking
* FAISS vector database
* OpenAI embeddings
* Contextual Compression Retriever
* RetrievalQA chain

The notebook shows how to:

1. Perform **passage reranking** using FlashRank
2. Build a **vector store retriever**
3. Apply **reranking as contextual compression**
4. Improve Retrieval-Augmented Generation (RAG) answers

---

# 🚀 Model Options (FlashRank)

FlashRank provides multiple lightweight reranking models.

| Model  | Size   | Speed     | Performance               | Use Case         |
| ------ | ------ | --------- | ------------------------- | ---------------- |
| Nano   | ~4MB   | ⚡ Fastest | Competitive               | Low-latency apps |
| Small  | ~34MB  | Fast      | Best ranking precision    | Balanced usage   |
| Medium | ~110MB | Slower    | Best zero-shot ranking    | Research         |
| Large  | ~150MB | Slow      | Multilingual (100+ langs) | Global search    |

---

## ⚡ FlashRank Characteristics

* Ultra‑lite CPU execution
* No heavy dependencies
* Serverless friendly
* Cross‑encoder based reranking
* Optimized for retrieval pipelines

Supported internal models include:

* `ms-marco-TinyBERT-L-2-v2` (default)
* `ms-marco-MiniLM-L-12-v2`
* `rank-T5-flan`
* `ms-marco-MultiBERT-L-12`

---

# 🧩 Step 1 — Install FlashRank

```python
!pip install flashrank
```

Installs the FlashRank library used for reranking passages.

---

# 🧩 Step 2 — Helper Function (Document Printing)

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )
```

### Purpose

* Nicely formats retrieved documents
* Shows:

  * document content
  * metadata

Useful for debugging retrieval pipelines.

---

# 🧩 Step 3 — Define Query

```python
query = "How to speedup LLMs?"
```

This query will be used for passage ranking.

---

# 🧩 Step 4 — Define Passages

```python
passages = [
   {
      "id":1,
      "text":"Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
      "meta": {"additional": "info1"}
   },
   {
      "id":2,
      "text":"LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
      "meta": {"additional": "info2"}
   },
   {
      "id":3,
      "text":"There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint...",
      "meta": {"additional": "info3"}
   },
   {
      "id":4,
      "text":"Ever want to make your LLM inference go brrrrr... Medusa framework... 2x speedup.",
      "meta": {"additional": "info4"}
   },
   {
      "id":5,
      "text":"vLLM is a fast and easy-to-use library for LLM inference and serving...",
      "meta": {"additional": "info5"}
   }
]
```

### Structure

Each passage contains:

| Field | Meaning             |
| ----- | ------------------- |
| id    | Unique identifier   |
| text  | Passage content     |
| meta  | Additional metadata |

---

# 🧩 Step 5 — Import FlashRank

```python
from flashrank.Ranker import Ranker, RerankRequest
```

### Components

* **Ranker** → Cross‑encoder scoring model
* **RerankRequest** → Input structure for reranking

---

# 🧩 Step 6 — Create Reranking Function

```python
def get_result(query,passages,choice):
  if choice == "Nano":
    ranker = Ranker()
  elif choice == "Small":
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
  elif choice == "Medium":
    ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")
  elif choice == "Large":
    ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/opt")

  rerankrequest = RerankRequest(query=query, passages=passages)
  results = ranker.rerank(rerankrequest)
  print(results)

  return results
```

### What Happens Internally

1. Model selected based on size
2. Query + passages packed into request
3. Cross‑encoder scores relevance
4. Returns ranked passages

---

# 🧩 Step 7 — Benchmark Execution Time

```python
%%time
print("sunny")
```

Used to measure execution time inside Colab.

---

## Run Nano Model

```python
%%time
get_result(query,passages,"Nano")
```

Runs fastest reranking model.

---

## Run Small Model

```python
%%time
get_result(query,passages,"Small")
```

Higher ranking precision.

---

## Run Medium Model

```python
%%time
get_result(query,passages,"Medium")
```

Best zero‑shot performance.

---

# 🧩 Step 8 — Install LangChain Dependencies

```python
!pip install langchain_community
!pip install langchain_openai
```

Installs LangChain integrations.

---

# 🧩 Step 9 — Load API Key (Colab Secrets)

```python
from google.colab import userdata
OPENAI_API_KEY=userdata.get('OPENAI_API_KEY')

import os
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
```

Loads OpenAI key securely from Colab secrets.

---

# 🧩 Step 10 — Import LangChain Modules

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
```

### Purpose

| Module       | Role                     |
| ------------ | ------------------------ |
| TextLoader   | Load documents           |
| TextSplitter | Chunk text               |
| Embeddings   | Convert text → vectors   |
| FAISS        | Vector similarity search |

---

# 🧩 Step 11 — Load Document

```python
documents = TextLoader("/content/state_of_the_union.txt").load()
```

Loads text file into LangChain Document objects.

---

# 🧩 Step 12 — Chunk Documents

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
```

### Why Chunking?

LLMs and embeddings work better with smaller semantic units.

| Parameter     | Meaning              |
| ------------- | -------------------- |
| chunk_size    | characters per chunk |
| chunk_overlap | shared context       |

---

# 🧩 Step 13 — Add Metadata IDs

```python
for id, text in enumerate(texts):
    text.metadata["id"] = id
```

Adds unique identifier for tracking retrieved chunks.

---

# 🧩 Step 14 — Create Embeddings

```python
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
```

Converts text chunks into dense vectors.

---

# 🧩 Step 15 — Install FAISS

```python
!pip install faiss-cpu
```

FAISS enables efficient vector similarity search.

---

# 🧩 Step 16 — Create Retriever

```python
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 10})
```

### Pipeline

Text → Embedding → FAISS Index → Retriever

Returns top‑10 similar chunks.

---

# 🧩 Step 17 — Query Retrieval

```python
query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
```

Retriever performs **vector similarity search**.

---

# 🧩 Step 18 — Inspect Retrieved Docs

```python
pretty_print_docs(docs)
```

Displays retrieved chunks.

---

# 🧩 Step 19 — Contextual Compression Setup

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI
```

### Key Idea

Instead of returning all retrieved docs → **compress them using reranking**.

---

# 🧩 Step 20 — Initialize LLM

```python
llm = ChatOpenAI(temperature=0)
```

Deterministic answer generation.

---

# 🧩 Step 21 — Create FlashRank Compressor

```python
compressor = FlashrankRerank()
```

Uses FlashRank as document reranker.

---

# 🧩 Step 22 — Create Compression Retriever

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Workflow

1. Retriever fetches candidates
2. FlashRank reranks them
3. Keeps only most relevant passages

---

# 🧩 Step 23 — Invoke Compression Retriever

```python
compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
```

Produces filtered high‑relevance documents.

---

# 🧩 Step 24 — Check Result Size

```python
len(compressed_docs)
```

Shows number of remaining documents after compression.

---

# 🧩 Step 25 — Inspect Metadata IDs

```python
print([doc.metadata["id"] for doc in compressed_docs])
```

Tracks which chunks survived reranking.

---

# 🧩 Step 26 — Print Compressed Documents

```python
pretty_print_docs(compressed_docs)
```

Shows reranked context.

---

# 🧩 Step 27 — Build RetrievalQA Chain

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

### Architecture

User Query → Retrieval → Reranking → LLM → Answer

---

# 🧩 Step 28 — Ask Final Question

```python
chain.invoke(query)
```

LLM generates answer using reranked context.

---

# 🧠 Full Pipeline Architecture

```
User Query
     ↓
Vector Retriever (FAISS)
     ↓
FlashRank Cross‑Encoder Reranking
     ↓
Context Compression
     ↓
LLM (ChatOpenAI)
     ↓
Final Answer
```

---

# ✅ Why FlashRank Improves RAG

| Without Reranking     | With FlashRank             |
| --------------------- | -------------------------- |
| Top‑k similarity only | Semantic relevance scoring |
| Noisy context         | Clean context              |
| More hallucination    | Reduced hallucination      |
| Token waste           | Efficient tokens           |

---

# 📊 When to Use FlashRank

* RAG pipelines
* Search engines
* QA systems
* Chatbots
* Low‑latency APIs
* Serverless deployments

---

# ⚙️ Performance Insight

FlashRank uses **Cross‑Encoders**:

```
Score(query, passage) = Transformer([query + passage])
```

Unlike bi‑encoders, query and passage interact directly.

Result → higher ranking accuracy.

---

# 🏁 Conclusion

This notebook demonstrates a **production‑grade RAG enhancement** using FlashRank reranking.

Key Learnings:

* Dense retrieval finds candidates
* Cross‑encoder reranking improves relevance
* Context compression reduces noise
* LLM answers become more accurate

---

# 📎 Requirements

```
flashrank
langchain
langchain_openai
langchain_community
faiss-cpu
```

---

# ⭐ End of README
