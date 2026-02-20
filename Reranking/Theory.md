# Reranking in RAG (Retrieval‑Augmented Generation)

---

Reranking in RAG (Retrieval-Augmented Generation) is an advanced retrieval optimization technique used to improve the quality of retrieved documents before sending them to the LLM.

It solves one of the biggest problems in RAG systems:

**Vector search retrieves similar chunks — not necessarily the most relevant ones.**

Let’s go step-by-step from **intuition → architecture → algorithms → implementation → best practices**.

---

## 1. Why Reranking is Needed in RAG

### Basic RAG Flow

```
User Query
     ↓
Embedding Model
     ↓
Vector Database (Top-K similarity search)
     ↓
Retrieved Chunks
     ↓
LLM → Answer
```

### Problem

Vector search uses embedding similarity (cosine distance).

But embeddings optimize for semantic similarity, not answer relevance.

### Example

**Query:**

```
"What are side effects of aspirin?"
```

Vector DB may retrieve:

* History of aspirin discovery
* Chemical composition
* Pain relief uses
* Side effects section (actual answer)

Similarity ≠ usefulness.

So the LLM receives noisy context.

### Result

* hallucinations
* incomplete answers
* wasted tokens

---

## 2. What is Reranking?

Reranking = Reordering retrieved documents using a stronger relevance model.

Instead of trusting vector similarity ranking:

```
Initial Retrieval → Reranker → Better Ranked Context
```

### Core Idea

* Retrieve many candidates quickly (cheap step)
* Use smarter model to rank relevance (expensive but accurate)
* Send only best chunks to LLM

---

## 3. RAG With Reranking (Architecture)

```
                FAST (Recall)
User Query ──► Vector Search (Top 20–50)
                        ↓
                ACCURATE (Precision)
                  Reranker Model
                        ↓
                 Top 3–5 Chunks
                        ↓
                       LLM
```

### Two-stage retrieval

| Stage     | Goal           | Method        |
| --------- | -------------- | ------------- |
| Retrieval | High recall    | Vector / BM25 |
| Reranking | High precision | Cross-encoder |

---

## 4. How Reranking Works Internally

Rerankers evaluate:

```
(Query, Document) → Relevance Score
```

Unlike embeddings:

Embeddings encode separately

Rerankers read query + document together

**This is VERY important.**

### Embedding Model (Bi-Encoder)

```
E(query)
E(document)

Similarity(Eq, Ed)
```

Fast but shallow understanding.

### Reranker (Cross-Encoder)

```
[CLS] Query + Document [SEP]
        ↓
Transformer
        ↓
Relevance Score
```

Model deeply understands interaction between query and text.

### Result

Much higher accuracy.

---

## 5. Types of Rerankers

### (1) Cross-Encoder Reranker (Most Common)

Best quality.

**Examples:**

* BAAI bge-reranker
* Cohere Rerank
* FlashRank
* monoT5

**Process**

For each retrieved chunk:

```
score = model(query, chunk)
```

Sort by score.

**Pros:**

* Extremely accurate

**Cons:**

* Slower (evaluates each pair)

---

### (2) LLM-based Reranking

LLM judges relevance.

Example prompt:

```
Query: ...
Document: ...
Rate relevance 1-10
```

**Pros:**

* Very intelligent

**Cons:**

* Expensive
* Slow

---

### (3) Hybrid Reranking

Combine:

* BM25 score
* Vector similarity
* Reranker score

Final score:

```
0.4 * vector_score +
0.3 * bm25_score +
0.3 * rerank_score
```

Used in production search engines.

---

### (4) Lightweight Rerankers

Optimized for speed.

**Example:**

* FlashRank (ultra-fast)
* MiniLM rerankers

Used in real-time apps.

---

## 6. Mathematical View

Given:

```
Query Q
Documents D1, D2, ... Dk
```

Vector search gives:

```
Sim(Q, Di)
```

Reranker computes:

```
R(Q, Di) = relevance score
```

Final ranking:

```
Sort(Di by R(Q, Di))
```

Where:

```
R(Q, Di) ≠ cosine similarity
```

It captures:

* intent
* reasoning
* context alignment

---

## 7. Reranking vs Similarity Search

| Feature       | Similarity Search | Reranking        |
| ------------- | ----------------- | ---------------- |
| Speed         | Very fast         | Slower           |
| Understanding | Surface semantic  | Deep interaction |
| Accuracy      | Medium            | High             |
| Cost          | Cheap             | Moderate         |
| Role          | Recall            | Precision        |

Best systems use BOTH.

---

## 8. Practical Example

### Without Reranking

Top-5 chunks:

* Intro paragraph
* Random mention
* Related topic
* Actual answer
* Irrelevant text

LLM confusion ↑

### With Reranking

After reranker:

* Actual answer
* Supporting paragraph
* Context explanation

LLM accuracy ↑ dramatically.

---

## 9. Implementation (LangChain Example)

### Install

```bash
pip install sentence-transformers
pip install flashrank
```

### Using BGE Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

pairs = [(query, doc.page_content) for doc in docs]

scores = reranker.predict(pairs)

# attach scores
for doc, score in zip(docs, scores):
    doc.metadata["score"] = score

reranked_docs = sorted(
    docs,
    key=lambda x: x.metadata["score"],
    reverse=True
)
```

### Using FlashRank (Fast)

```python
from flashrank import Ranker, RerankRequest

ranker = Ranker()

request = RerankRequest(
    query=query,
    passages=[d.page_content for d in docs]
)

results = ranker.rerank(request)
```

---

## 10. Where Reranking Helps MOST

Reranking is critical when:

* Large documents
* Chunk overlap issues
* Similar topics exist
* Legal / medical docs
* Research papers
* Enterprise knowledge bases
* Multi-PDF QA systems

Basically → serious RAG systems.

---

## 11. Performance Impact

Typical improvements:

| Metric                  | Improvement |
| ----------------------- | ----------- |
| Retrieval accuracy      | +20–40%     |
| Hallucination reduction | High        |
| Answer grounding        | Much better |
| Token efficiency        | Higher      |

This is why production RAG almost always uses reranking.

---

## 12. Best Practices (Industry Standard)

### Retrieve More → Rerank → Reduce

* Retrieve: Top 30–50
* Rerank: Keep Top 3–5
* Send to LLM

### Use Hybrid Retrieval + Rerank

BM25 + Vector → Reranker

Best performing architecture today.

### Cache Reranking Results

Rerankers are expensive.

### Use Smaller Rerankers for Speed

Example:

* FlashRank for real-time apps
* BGE reranker for accuracy

---

## 13. Common Mistakes

❌ Retrieving only top-3 before reranking
✔ Retrieve many candidates first.

❌ Sending all retrieved chunks to LLM
✔ Compress via reranking.

❌ Assuming embeddings are enough
✔ They are recall tools, not precision tools.

---

## 14. Real Production RAG Pipeline (Modern)

```
User Query
     ↓
Query Expansion
     ↓
Hybrid Retrieval (Vector + BM25)
     ↓
Top 40 Chunks
     ↓
Reranker (Cross Encoder)
     ↓
Top 5 Chunks
     ↓
Context Compression
     ↓
LLM
     ↓
Answer
```

This is used by:

* Perplexity
* Enterprise search copilots
* Advanced document QA systems

---

## 15. One-Line Intuition

**Vector Search finds POSSIBLE answers.**
**Reranking finds the BEST answers.**

---
