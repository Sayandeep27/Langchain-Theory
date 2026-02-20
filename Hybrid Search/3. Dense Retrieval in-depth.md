# Dense Retrieval — Complete Guide (GitHub README)

---

# 1. What is Retrieval?

**Retrieval** is the process of finding the most relevant documents from a large collection based on a user query.

### Goal

> Given a query → return the most relevant documents.

### Example

**Query:**

```
What causes global warming?
```

**Dataset:**

* Doc1: Climate change explanation
* Doc2: Football news
* Doc3: Carbon emissions article

Retriever selects **Doc1 & Doc3**.

---

# 2. Retrieval Paradigms

| Type             | Representation       | Matching Method     |
| ---------------- | -------------------- | ------------------- |
| Sparse Retrieval | Keywords             | Exact word overlap  |
| Dense Retrieval  | Embeddings (vectors) | Semantic similarity |

---

## Sparse Retrieval (Traditional)

Example: **BM25**

Query:

```
car insurance price
```

Matches documents containing the same words.

### Problem

* Cannot understand meaning
* Synonyms fail

Example:

Query: *"automobile coverage cost"*

BM25 may fail because wording differs.

---

## Dense Retrieval (Modern Approach)

Dense retrieval uses **semantic understanding** via embeddings.

---

# 3. What is Dense Retrieval?

### Definition

Dense Retrieval converts queries and documents into dense vector embeddings and retrieves documents using vector similarity.

```
Query → Embedding Vector
Document → Embedding Vector
Similarity(query, document) → ranking
```

### Example

**Query:**

```
How to lose weight fast?
```

| Document                | Meaning    |
| ----------------------- | ---------- |
| Diet and exercise guide | Relevant   |
| Gym workout plan        | Relevant   |
| Stock market tips       | Irrelevant |

Embeddings capture meaning even if wording differs.

---

# 4. Core Idea

Neural networks place semantically similar texts close together in vector space.

```
          fitness
             ●
           ● query
        ● diet
                     ● finance
```

Similar meaning → closer vectors.

---

# 5. Dense Retrieval Pipeline

## Step 1 — Document Encoding

```
Document → Encoder → Vector
```

Example vector:

```
[0.21, -0.44, 0.91, ...]
```

Stored in:

* FAISS
* Pinecone
* Weaviate
* Chroma

---

## Step 2 — Query Encoding

```
Query → Encoder → Vector
```

---

## Step 3 — Similarity Search

Common metrics:

* Cosine similarity (most used)
* Dot product
* Euclidean distance

---

## Step 4 — Retrieve Top‑K

Return most similar documents.

---

# 6. Architecture of Dense Retrieval

## Dual Encoder (Bi‑Encoder)

```
Query Encoder      Document Encoder
      ↓                    ↓
   Query Vec          Doc Vec
            Similarity
```

Examples:

* DPR
* Sentence Transformers
* BGE
* E5 models

Advantages:

* Fast retrieval
* Pre‑computable document embeddings

---

## Cross Encoder (Re‑ranking)

```
[Query + Document] → Model → Score
```

More accurate but slower.

Used as:

Retriever + Reranker pipeline

---

# 7. Types of Dense Retrieval

## 1. DPR (Dense Passage Retrieval)

Training idea:

* Positive passages close
* Negative passages far

Used heavily in QA systems.

---

## 2. Sentence Transformer Retrieval

Common models:

* all-MiniLM-L6-v2
* bge-small
* e5-base

### Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

query_emb = model.encode("What is AI?")
doc_emb = model.encode("Artificial intelligence explained")
```

---

## 3. Multi‑Vector Retrieval (ColBERT)

```
Document → multiple token embeddings
```

Better semantic matching.

---

## 4. Late Interaction Retrieval

Example: **ColBERT**

Interaction happens after token encoding.

Benefits:

* High recall
* Fine‑grained matching

---

## 5. Hybrid Dense Retrieval

```
Dense Retrieval + BM25
```

Why?

* Dense = semantics
* Sparse = exact keywords

Used in:

* Pinecone Hybrid Search
* Weaviate Hybrid
* Elasticsearch Hybrid

---

# 8. Training Dense Retrieval Models

Training uses **contrastive learning**.

### Data Format

```
(Query, Positive Doc, Negative Doc)
```

### Objective

```
Similarity(query, positive) ↑
Similarity(query, negative) ↓
```

Loss Functions:

* Contrastive loss
* Triplet loss
* InfoNCE loss

---

# 9. Important Concepts

## Embedding Model Choice

| Model      | Use Case          |
| ---------- | ----------------- |
| MiniLM     | Fast              |
| BGE        | High quality      |
| E5         | Query‑focused     |
| Instructor | Instruction‑aware |

---

## Chunking Strategy

Best practice:

```
Chunk size: 300–800 tokens
Overlap: 50–150 tokens
```

---

## Vector Dimension

Typical sizes:

* 384
* 768
* 1024

Higher dimension → better representation but slower search.

---

## ANN Search (Approximate Nearest Neighbor)

Libraries:

* FAISS (IVF, HNSW)
* ScaNN
* Annoy

---

## Recall vs Precision Tradeoff

Dense retrieval:

* High semantic recall
* Sometimes lower precision

Solution → Add reranker.

---

# 10. Dense Retrieval in RAG

Pipeline:

```
User Query
     ↓
Embedding Model
     ↓
Vector DB Search
     ↓
Top Documents
     ↓
LLM Generation
```

### Example Code

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

db = FAISS.from_texts(docs, embeddings)

results = db.similarity_search("Explain transformers")
```

---

# 11. Advantages

* Understands meaning
* Works with synonyms
* Language flexible
* Great for QA
* Ideal for LLMs

---

# 12. Limitations

### Keyword Miss

Example:

```
Error code: XG-4451
```

Sparse retrieval works better.

### Requires GPU for training

### Embedding Drift

Model updates require re‑embedding documents.

---

# 13. Dense vs Sparse vs Hybrid

| Feature                | Sparse | Dense | Hybrid    |
| ---------------------- | ------ | ----- | --------- |
| Semantic understanding | ❌      | ✅     | ✅         |
| Keyword matching       | ✅      | ❌     | ✅         |
| Speed                  | Fast   | Fast  | Medium    |
| Accuracy               | Medium | High  | Very High |
| RAG usage              | Low    | High  | Best      |

---

# 14. Real‑World Systems

* ChatGPT RAG systems
* Google semantic search
* GitHub Copilot search
* Enterprise document search
* Customer support bots

---

# 15. Advanced Concepts

* Query Expansion
* Reranking
* Contextual Compression
* Multi‑hop Retrieval
* Adaptive Retrieval

---

# 16. Mental Model

```
Sparse Retrieval = WORD MATCHING
Dense Retrieval  = MEANING MATCHING
```

---

# 17. When to Use Dense Retrieval

Use when:

* Semantic search needed
* RAG applications
* Question answering
* Chatbots
* Knowledge bases

Avoid alone when:

* IDs or codes dominate queries.

---

# 18. Real‑Life Analogy

Sparse retrieval:

> Searching dictionary alphabetically.

Dense retrieval:

> Asking a knowledgeable human who understands meaning.

---

# Hard Negative Mining (Short Explanation)

## Definition

Hard Negative Mining is a training technique used in Dense Retrieval models to improve accuracy by teaching the model to distinguish between very similar but incorrect documents.

---

## Training Structure

```
(Query, Positive Document, Negative Document)
```

* Positive → correct relevant document
* Negative → irrelevant document

---

## Easy Negative Example

Query:

```
What is machine learning?
```

Negative:

```
Football match results
```

Too easy for the model.

---

## Hard Negative Example

Query:

```
What is machine learning?
```

Positive:

```
Machine learning is a subset of AI...
```

Hard Negative:

```
Deep learning neural network architecture overview
```

Semantically similar but incorrect.

---

## Why Hard Negatives Matter

Without hard negatives:

* Model becomes lazy
* Learns obvious differences only

With hard negatives:

* Learns fine semantic distinctions
* Improves ranking accuracy
* Better RAG retrieval

---

## How Hard Negatives Are Generated

1. BM25 retrieval negatives
2. Dense model mining
3. In‑batch negatives

---

## One‑Line Definition

> Hard negative mining trains a dense retriever using confusing but incorrect documents so it learns precise semantic matching.

---

## Impact in RAG

Improves:

* Retrieval precision
* Reduced hallucinations
* Better context selection


