# 🔎 Retrievers in RAG (Retrieval‑Augmented Generation)

---

## 📌 What is a Retriever?

A **Retriever** is a component in a RAG pipeline responsible for **fetching relevant documents or chunks** from a knowledge source based on a user query.

Instead of relying only on an LLM’s internal knowledge, retrievers allow the model to access **external data**.

### Core Responsibility

```
User Query → Retriever → Relevant Documents → LLM → Answer
```

### Why Retrievers Matter

| Problem               | Solution by Retriever        |
| --------------------- | ---------------------------- |
| LLM hallucination     | Provides grounded context    |
| Limited training data | Access external knowledge    |
| Outdated information  | Query live knowledge bases   |
| Large documents       | Retrieve only relevant parts |

---

## 🧠 Retriever Architecture in RAG

```
                User Query
                     │
                     ▼
                Query Processing
                     │
                     ▼
                 Retriever
                     │
         ┌───────────┼───────────┐
         ▼                       ▼
   Knowledge Source        Metadata Filters
         │
         ▼
   Retrieved Documents
         │
         ▼
            LLM
```

---

# 📚 Types of Retrievers (Complete Guide)

---

## 1️⃣ Vector Store Retriever

### Concept

Retrieves documents using **semantic similarity** via embeddings.

Query and documents are converted into vectors and compared using cosine similarity.

### Use Case

Best general‑purpose retriever for RAG.

### Example

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is transformers?")
```

### Key Idea

```
semantic meaning > keyword matching
```

---

## 2️⃣ Wikipedia Retriever

### Concept

Fetches information directly from Wikipedia pages.

### Use Case

Quick factual lookup without building a vector database.

### Example

```python
from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=3)
docs = retriever.invoke("Neural networks")
```

---

## 3️⃣ BM25 Retriever (Sparse Retrieval)

### Concept

Keyword‑based ranking using term frequency and inverse document frequency.

### Formula Idea

```
score ∝ TF * IDF / document_length
```

### Use Case

Exact keyword search, legal or technical documents.

### Example

```python
from langchain.retrievers import BM25Retriever

retriever = BM25Retriever.from_texts(texts)
docs = retriever.invoke("machine learning algorithm")
```

### Strength

Excellent for lexical matching.

---

## 4️⃣ MMR Retriever (Max Marginal Relevance)

### Concept

Balances **relevance + diversity**.

Avoids returning similar chunks repeatedly.

### Idea

```
Select docs maximizing:
relevance − redundancy
```

### Example

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":5, "lambda_mult":0.5}
)
```

### When to Use

Long documents with repeated content.

---

## 5️⃣ Contextual Compression Retriever

### Concept

Filters or compresses retrieved documents before sending to the LLM.

Reduces token cost.

### Architecture

```
Retriever → Compressor → Short Relevant Context → LLM
```

### Example

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)
```

---

## 6️⃣ Self‑Query Retriever

### Concept

LLM converts a user query into:

* semantic search query
* metadata filters

Automatically.

### Example Query

```
"papers about transformers after 2021"
```

Converted into:

```
query = "transformers"
filter = year > 2021
```

### Example

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description="research papers",
    metadata_field_info=metadata
)
```

---

## 7️⃣ Arxiv Retriever

### Concept

Searches academic papers from arXiv.

### Example

```python
from langchain.retrievers import ArxivRetriever

retriever = ArxivRetriever(load_max_docs=3)
docs = retriever.invoke("diffusion models")
```

### Use Case

Research assistants and academic RAG.

---

## 8️⃣ Multi‑Vector Retriever

### Concept

Stores **multiple embeddings per document**.

Examples:

* summaries
* questions
* sections

All point to the same parent doc.

### Benefit

Improves recall.

### Example

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id"
)
```

---

## 9️⃣ Parent Document Retriever

### Concept

Search small chunks but return larger parent documents.

### Why?

Small chunks → better search
Large context → better reasoning

### Example

```python
from langchain.retrievers import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore
)
```

---

## 🔟 Multi‑Query Retriever

### Concept

LLM generates multiple query variations automatically.

Example:

```
Original: "Explain RAG"
Generated:
- What is retrieval augmented generation?
- How does RAG work?
- RAG architecture
```

### Example

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

### Benefit

Higher recall and coverage.

---

# 📊 Retriever Comparison Table

| Retriever              | Type            | Best For           | Strength             |
| ---------------------- | --------------- | ------------------ | -------------------- |
| Vector Store           | Dense           | General RAG        | Semantic search      |
| BM25                   | Sparse          | Keywords           | Precision            |
| Wikipedia              | API             | Facts              | No setup             |
| Arxiv                  | API             | Research           | Academic data        |
| MMR                    | Dense Variant   | Diversity          | Less redundancy      |
| Contextual Compression | Post‑processing | Token saving       | Efficiency           |
| Self Query             | LLM‑assisted    | Metadata filtering | Smart search         |
| Multi Vector           | Dense           | Recall             | Multiple signals     |
| Parent Doc             | Hybrid          | Long docs          | Context preservation |
| Multi Query            | LLM‑expanded    | Query coverage     | Better retrieval     |

---

# 🏗️ How Retrievers Fit in Production RAG

```
User Query
   │
   ▼
Multi‑Query Retriever
   │
   ▼
Vector Search / BM25
   │
   ▼
Reranker (optional)
   │
   ▼
Contextual Compression
   │
   ▼
LLM Answer
```

---

# ✅ Best Practices

* Combine **dense + sparse** retrieval (Hybrid Search)
* Use **MMR** for diversity
* Add **reranking** after retrieval
* Compress context before LLM
* Use metadata filters via Self‑Query
* Prefer ParentDoc for long PDFs

---

# 🚀 Minimal End‑to‑End Example

```python
retriever = vectorstore.as_retriever(search_type="mmr", k=5)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

qa_chain.invoke({"query": "Explain transformers"})
```

---

# 🧩 Retriever Selection Guide

| Scenario            | Recommended Retriever |
| ------------------- | --------------------- |
| General chatbot     | Vector Store          |
| Keyword heavy docs  | BM25                  |
| Academic search     | Arxiv                 |
| Structured metadata | Self Query            |
| Large documents     | Parent Doc            |
| Low token budget    | Context Compression   |
| Better recall       | Multi Query           |
| Avoid duplicates    | MMR                   |

---

# 📎 Key Takeaway

> **Retriever quality = RAG quality**

A strong retriever pipeline often improves results more than upgrading the LLM itself.
