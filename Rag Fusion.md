# RAG Fusion + Reranking Architecture

---

# Table of Contents

1. Introduction
2. Problem with Basic RAG
3. What is RAG Fusion?
4. What is Reranking?
5. Why Combine Fusion + Reranking?
6. Industry Standard Architecture
7. Complete Pipeline Flow
8. Reciprocal Rank Fusion (RRF)
9. Reranking Models
10. Full Implementation (Python + LangChain)
11. Advanced Production Stack
12. Performance Comparison
13. When to Use This Architecture
14. Best Practices
15. Common Mistakes
16. Interview Definition
17. Project Structure Example
18. References

---

# 1. Introduction

**RAG Fusion + Reranking** is an advanced retrieval architecture designed to improve:

* Retrieval Recall (finding all relevant information)
* Retrieval Precision (keeping only the best information)
* LLM Answer Quality
* Hallucination Reduction

Modern AI systems do NOT rely on single vector search anymore.

Instead, they use **multi‑stage retrieval pipelines**.

---

# 2. Problem with Basic RAG

## Basic RAG Flow

```
User Query → Embedding → Vector Search → Top‑K Chunks → LLM
```

### Issues

| Problem           | Explanation                           |
| ----------------- | ------------------------------------- |
| Query ambiguity   | One query cannot capture all meanings |
| Missed context    | Relevant docs may not be retrieved    |
| Semantic mismatch | Embeddings approximate meaning        |
| Noise             | Irrelevant chunks included            |

Result → Lower answer accuracy.

---

# 3. What is RAG Fusion?

RAG Fusion improves **recall** by generating multiple search queries.

## Core Idea

Instead of searching once:

```
1 query → 1 retrieval
```

We do:

```
Multiple queries → Multiple retrievals → Fusion
```

### Example

User Question:

```
How memory works in LangChain?
```

Generated Queries:

```
- types of langchain memory
- conversation buffer memory
- entity memory langchain
- chat history storage langchain
```

Each retrieves different documents.

---

# 4. What is Reranking?

Reranking improves **precision**.

A reranker evaluates relevance by reading:

```
(Query + Document) together
```

Unlike embeddings, which compare vectors separately.

## Embedding Search vs Reranker

| Feature  | Embeddings        | Reranker        |
| -------- | ----------------- | --------------- |
| Speed    | Fast              | Slower          |
| Accuracy | Approximate       | High            |
| Method   | Vector similarity | Cross attention |
| Purpose  | Recall            | Precision       |

---

# 5. Why Combine Fusion + Reranking?

```
RAG Fusion  → Find EVERYTHING relevant
Reranker    → Keep ONLY best results
```

Analogy:

| Stage    | Role                  |
| -------- | --------------------- |
| Fusion   | HR collects resumes   |
| Reranker | Technical interview   |
| LLM      | Final hiring decision |

---

# 6. Industry Standard Architecture

```
                USER QUERY
                     │
                     ▼
            Query Expansion (LLM)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Query1       Query2       Query3
        │            │            │
        └──── Vector Retrieval ───┘
                     │
              RAG Fusion (RRF)
                     │
           Candidate Documents
                     │
                 RERANKER
                     │
             Top‑K Documents
                     │
                    LLM
                     │
               Final Answer
```

---

# 7. Complete Pipeline Flow

## Step 1 — Query Expansion

LLM generates multiple semantic variations.

## Step 2 — Parallel Retrieval

Each query retrieves documents independently.

## Step 3 — Rank Fusion

Combine ranked results using RRF.

## Step 4 — Reranking

Cross‑encoder scores true relevance.

## Step 5 — Context Selection

Select top documents only.

## Step 6 — Generation

LLM produces grounded answer.

---

# 8. Reciprocal Rank Fusion (RRF)

RRF merges multiple ranked lists.

## Formula

```
Score(d) = Σ 1 / (k + rank(d))
```

Where:

* rank(d) = document position
* k = constant (usually 60)

Documents appearing across searches receive higher scores.

## Example

| Document | Q1 Rank | Q2 Rank | Score  |
| -------- | ------- | ------- | ------ |
| Doc A    | 1       | 3       | High   |
| Doc B    | 2       | —       | Medium |
| Doc C    | —       | 1       | Medium |

---

# 9. Reranking Models

## Lightweight (Fast)

* FlashRank
* bge-reranker-base
* jina-reranker

## High Accuracy

* CrossEncoder (SentenceTransformers)
* Cohere Rerank
* OpenAI Rerank models

---

# 10. Full Implementation (Python + LangChain)

## Install Dependencies

```bash
pip install langchain sentence-transformers faiss-cpu
```

---

## Query Generation

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate 4 search queries for: {question}"
)

query_chain = LLMChain(llm=llm, prompt=prompt)
queries = query_chain.run(question).split("\n")
```

---

## Retrieval

```python
all_docs = []
for q in queries:
    docs = retriever.get_relevant_documents(q)
    all_docs.append(docs)
```

---

## Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(results, k=60):
    scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            scores[doc] = scores.get(doc, 0) + 1/(k + rank)
    return sorted(scores, key=scores.get, reverse=True)

fused_docs = reciprocal_rank_fusion(all_docs)
```

---

## Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(question, doc.page_content) for doc in fused_docs]
scores = reranker.predict(pairs)

reranked_docs = [doc for _, doc in sorted(zip(scores, fused_docs), reverse=True)]
```

---

## Final Generation

```python
context = reranked_docs[:5]
response = llm.invoke(str(context) + question)
```

---

# 11. Advanced Production Stack

Modern enterprise systems often use:

```
User Query
   │
Query Rewriting
   │
Hybrid Search (BM25 + Vector)
   │
RAG Fusion (RRF)
   │
Reranker
   │
Context Compression
   │
LLM Generation
```

---

# 12. Performance Comparison

| Metric         | Basic RAG | Fusion + Rerank  |
| -------------- | --------- | ---------------- |
| Recall         | Medium    | Very High        |
| Precision      | Medium    | High             |
| Hallucination  | Higher    | Lower            |
| Answer Quality | Good      | Production Grade |

---

# 13. When to Use This Architecture

Use when:

* Large document collections
* Enterprise knowledge bases
* Research assistants
* Legal or financial documents
* Multi-topic PDFs
* High accuracy requirements

Avoid for:

* Small FAQ datasets
* Simple chatbots

---

# 14. Best Practices

* Use 3–5 query variations
* Retrieve 30–50 candidates before reranking
* Rerank down to 5–8 chunks
* Keep chunk size ~300–600 tokens
* Combine with hybrid search for best results

---

# 15. Common Mistakes

| Mistake                  | Why Wrong       |
| ------------------------ | --------------- |
| Using only vector search | Low recall      |
| Skipping reranker        | Noisy context   |
| Too many final chunks    | Token waste     |
| Large chunk size         | Poor embeddings |

---

# 16. Interview Definition

**RAG Fusion + Reranking is a two‑stage retrieval architecture where multiple LLM‑generated query searches maximize recall and a cross‑encoder reranker refines results to maximize precision before final LLM generation.**

---

# 17. Project Structure Example

```
rag-project/
│
├── data/
├── embeddings/
├── retriever.py
├── fusion.py
├── reranker.py
├── generator.py
├── app.py
└── README.md
```

