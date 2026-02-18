# LangChain Retrievers & MMR — Complete Guide
---

# 1. Why Retrievers are Needed

Large Language Models (LLMs):

* ❌ Do NOT know your private data
* ❌ Cannot access PDFs or databases directly
* ❌ May hallucinate

Instead of retraining models, we:

1. Store documents in a database
2. Search relevant pieces using embeddings
3. Send only relevant context to LLM

The searching step is done by a **Retriever**.

```
User Question
      ↓
Retriever (search relevant info)
      ↓
Relevant Documents
      ↓
LLM generates answer
```

---

# 2. What is a Retriever (Definition)

A **Retriever** in LangChain is:

> A component that takes a query and returns relevant documents.

### Input

```
"What is machine learning?"
```

### Output

```
[Document1, Document2, Document3]
```

Retriever returns documents — NOT answers.

---

# 3. Retriever vs Vector Store

| Component    | Role               |
| ------------ | ------------------ |
| Vector Store | Stores embeddings  |
| Retriever    | Searches documents |
| LLM          | Generates answers  |

Analogy:

```
Vector DB = Library
Retriever = Librarian
LLM = Teacher
```

---

# 4. How Retriever Works Internally

### Step 1 — Create Embeddings

```
Text → Embedding Model → Vectors
```

### Step 2 — Store Vectors

Examples:

* FAISS
* Chroma
* Pinecone

### Step 3 — Query Embedding

```
User Query → Embedding
```

### Step 4 — Similarity Search

### Step 5 — Return Documents

---

# 5. Basic Retriever Example (LangChain)

## Install

```bash
pip install langchain langchain-community faiss-cpu sentence-transformers
```

## Load Documents

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("data.txt")
docs = loader.load()
```

## Create Embeddings

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Create Vector Store

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
```

## Create Retriever

```python
retriever = vectorstore.as_retriever()
```

## Retrieve Documents

```python
results = retriever.invoke("What is AI?")
print(results[0].page_content)
```

---

# 6. Retriever Types in LangChain

## 1. Similarity Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity"
)
```

Best for normal semantic search.

---

## 2. MMR Retriever

Balances relevance and diversity.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr"
)
```

---

## 3. Similarity Score Threshold

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
```

---

## 4. k-Nearest Documents

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

---

# 7. Retriever in RAG Pipeline

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

qa_chain.invoke("Explain neural networks")
```

Pipeline:

```
Question
   ↓
Retriever → Relevant Docs
   ↓
Prompt + Docs
   ↓
LLM Answer
```

---

# 8. Advanced Retrievers

## MultiQuery Retriever

LLM generates multiple query variations.

## Contextual Compression Retriever

Removes irrelevant text before sending to LLM.

## Parent Document Retriever

Search small chunks but return larger context.

## Self‑Query Retriever

LLM converts natural language into metadata filters.

Example:

```
"Show AI papers after 2022"
```

---

# 9. Retriever Interface

All retrievers support:

```python
retriever.invoke(query)
```

or

```python
retriever.get_relevant_documents(query)
```

Returns:

```
List[Document]
```

---

# 10. Real‑World Example Mapping

| System Component      | LangChain Concept |
| --------------------- | ----------------- |
| PDFs                  | Documents         |
| sentence-transformers | Embeddings        |
| FAISS                 | Vector Store      |
| Search logic          | Retriever         |
| LLM                   | Generator         |

---

# 11. Interview Definition

> A retriever in LangChain fetches relevant documents from a knowledge source using semantic or other search strategies, enabling Retrieval‑Augmented Generation.

---

# 12. Mental Model

```
Retriever DOES NOT ANSWER
Retriever ONLY FINDS
LLM EXPLAINS
```

---

# 13. MMR (Maximum Marginal Relevance)

## Problem with Normal Similarity Search

Similarity search often returns duplicate or highly similar chunks.

Example:

```
Doc1 → ML is a subset of AI
Doc2 → Machine learning is a branch of AI
Doc3 → ML enables computers to learn
```

Issues:

* Wasted context window
* Higher token cost
* Less diversity

---

## Goal of MMR

We want documents that are:

1. Relevant to query
2. Different from each other

MMR achieves this balance.

---

# 14. What is MMR

**Maximum Marginal Relevance (MMR)** balances:

* Relevance to query
* Diversity among documents

Definition:

> MMR selects documents relevant to the query while minimizing redundancy.

---

## Human Analogy

Similarity Search → Same department experts

MMR → Experts from different domains

---

# 15. How MMR Works

Step‑wise:

1. Pick most relevant document.
2. Next selections consider:

   * similarity to query
   * difference from selected documents

Repeated content gets penalized.

---

# 16. MMR Mathematical Idea

```
MMR = λ * Similarity(Query, D)
      -
      (1 − λ) * Similarity(D, Selected Docs)
```

Where λ controls balance.

| λ Value | Behavior       |
| ------- | -------------- |
| 1       | Only relevance |
| 0       | Only diversity |
| ~0.5    | Balanced       |

---

# 17. Visual Understanding

Without MMR:

```
[Same idea repeated]
```

With MMR:

```
Definition
Applications
History
Advantages
Limitations
```

---

# 18. MMR in LangChain

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20
    }
)
```

### Parameters

**k** → final documents returned

**fetch_k** → candidates before diversity filtering

---

# 19. Example Usage

```python
docs = retriever.invoke("Explain neural networks")

for d in docs:
    print(d.page_content)
```

---

# 20. Why MMR Improves RAG

| Metric           | Effect |
| ---------------- | ------ |
| Answer quality   | ↑      |
| Hallucination    | ↓      |
| Token efficiency | ↑      |
| Context coverage | ↑      |

---

# 21. When to Use MMR

Use when:

* Long PDFs
* Chunked documents
* Knowledge bases
* Enterprise RAG

Avoid when:

* Very small datasets
* Exact lookup tasks

---

# 22. Similarity vs MMR

| Feature          | Similarity | MMR                   |
| ---------------- | ---------- | --------------------- |
| Focus            | Relevance  | Relevance + Diversity |
| Duplicate chunks | High       | Low                   |
| Context variety  | Low        | High                  |
| RAG quality      | Medium     | High                  |
| Token usage      | Wasteful   | Efficient             |

---

# 23. Production RAG Insight

Typical pipeline:

```
Similarity Search (large fetch_k)
        ↓
MMR Filtering
        ↓
Reranker (optional)
        ↓
LLM
```

MMR acts as a **middle optimization layer**.

---

# 24. Final Mental Model

```
Similarity = Best individuals
MMR = Best team
```

---

# End of README

This document contains the **complete explanation of Retrievers and MMR exactly as covered** and is ready for GitHub usage.
