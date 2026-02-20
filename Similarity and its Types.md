# Similarity in RAG (Retrieval‑Augmented Generation)

---

## Good question — this is **one of the most important foundations of RAG**.

If you truly understand **similarity**, you understand how retrieval actually works.

Let’s go step‑by‑step from intuition → math → types → when to use → MMR mapping.

---

# Similarity in RAG (Retrieval-Augmented Generation)

## 1. What is “Similarity” in RAG?

In **RAG**, similarity answers one core question:

> **Which stored document chunks are most related to the user query?**

Since LLMs cannot search raw text efficiently, we convert text into **vectors (embeddings)**.

```
Text → Embedding Model → Vector (numbers)
```

Example:

```
Query: "What is ANN search?"

→ [0.21, -0.44, 0.89, ...]
```

Every document chunk also becomes a vector.

Retrieval = **compare vectors and find closest ones**.

So similarity = **mathematical closeness between vectors**.

---

## 2. Why Similarity Matters in RAG

RAG pipeline:

```
User Query
     ↓
Embedding Model
     ↓
Similarity Search   ← (CORE PART)
     ↓
Top-k Documents
     ↓
LLM Answer
```

If similarity is wrong → retrieval is wrong → hallucination increases.

---

# 3. Vector Similarity — Intuition

Imagine vectors as **points in space**.

Two texts are similar if:

* They point in same direction
* They are close in space
* Their semantic meaning overlaps

Different similarity metrics measure this differently.

---

# 4. Main Types of Similarity Metrics

We divide them into **two families**:

| Family     | Measures        |
| ---------- | --------------- |
| Similarity | Higher = better |
| Distance   | Lower = better  |

---

# A. COSINE SIMILARITY (Most Important)

## Idea

Measures **angle between vectors**, not magnitude.

```
Same direction → high similarity
Different direction → low similarity
```

### Formula

```
cos(θ) = (A · B) / (||A|| ||B||)
```

Range:

```
-1 → opposite
0  → unrelated
1  → identical meaning
```

---

## Why Cosine Works Well for NLP

Embeddings encode **semantic direction**, not length.

Example:

```
"dog animal"
"puppy pet"
```

Vectors point similarly → high cosine similarity.

---

## Advantages

* Scale independent
* Stable embeddings
* Best semantic matching
* Default for most vector DBs

---

## Used In

* FAISS
* Pinecone
* Chroma
* Weaviate
* OpenAI embeddings
* Sentence Transformers

---

## Use When

✅ Semantic search
✅ RAG retrieval
✅ Question answering
✅ General NLP

👉 **Default choice for RAG**

---

# B. DOT PRODUCT (Inner Product)

## Idea

Measures alignment **and magnitude**.

```
A · B = Σ AiBi
```

Large vectors produce larger scores.

---

## Intuition

```
Cosine → direction only
Dot Product → direction + strength
```

---

## When Useful

If embedding magnitude carries meaning.

Some models intentionally encode confidence in vector length.

---

## Used In

* Maximum Inner Product Search (MIPS)
* Dense Passage Retrieval (DPR)
* Large-scale recommender systems

---

## Use When

✅ Model trained with dot-product objective
✅ Recommendation systems
✅ DPR-style retrieval

❌ Not ideal if vectors not normalized.

---

# C. DISTANCE METRICS

Distance = **how far apart vectors are**.

Lower distance → higher similarity.

---

## 1. Euclidean Distance (L2)

### Formula

```
sqrt(Σ (Ai − Bi)^2)
```

Straight-line distance.

### Intuition

Geometric closeness.

---

### Pros

* Natural geometric measure
* Works in low dimensions

### Cons

* High-dimensional embeddings suffer (curse of dimensionality)

---

### Use When

✅ Image embeddings
✅ Physical measurements
⚠️ Less common in text RAG

---

## 2. Manhattan Distance (L1)

```
Σ |Ai − Bi|
```

Grid-like movement distance.

---

### Use When

* Sparse vectors
* Feature-based ML
* Some classical IR systems

Rare in modern RAG.

---

## 3. Hamming Distance

Counts number of different bits.

```
10101
11100
↓
differences = 3
```

---

### Use When

✅ Binary embeddings
✅ Hash-based retrieval
✅ ANN indexing tricks

Not used for normal text embeddings.

---

# 5. Quick Comparison Table

| Metric      | Measures          | Best For            | RAG Usage |
| ----------- | ----------------- | ------------------- | --------- |
| Cosine      | Angle             | Semantic similarity | ⭐⭐⭐⭐⭐     |
| Dot Product | Angle + magnitude | DPR, recommenders   | ⭐⭐⭐⭐      |
| Euclidean   | Spatial distance  | vision/audio        | ⭐⭐        |
| Manhattan   | Axis distance     | sparse data         | ⭐         |
| Hamming     | Bit difference    | binary vectors      | ⭐         |

---

# 6. How Vector Databases Map Them

Important insight:

> Many systems internally convert metrics.

Example:

If vectors are **normalized**:

```
Cosine similarity ≈ Dot product
```

Because:

```
||A|| = 1
||B|| = 1
```

So:

```
A · B = cosine similarity
```

That’s why many ANN libraries use inner product internally.

---

# 7. ANN Search + Similarity

Approximate Nearest Neighbor (ANN) indexes optimize search based on metric.

| ANN Index | Works Best With |
| --------- | --------------- |
| HNSW      | Cosine / Dot    |
| IVF       | Euclidean       |
| PQ        | Euclidean       |
| ScaNN     | Dot/Cosine      |

Choosing wrong metric reduces recall.

---

# 8. MMR (Maximal Marginal Relevance)

Now comes the **advanced part**.

---

## Problem: Similarity Collapse

Top-k retrieval often returns:

```
Doc 1: ANN explanation
Doc 2: ANN explanation (same)
Doc 3: ANN explanation (same)
```

All similar → low diversity.

LLM receives redundant context.

---

## MMR Idea

Balance:

```
Relevance  +  Diversity
```

Instead of:

```
Most similar documents
```

We choose:

```
Relevant BUT different documents
```

---

## MMR Formula (Conceptual)

```
MMR = λ × similarity(query, doc)
      − (1 − λ) × similarity(doc, selected_docs)
```

Where:

* First term → relevance
* Second term → redundancy penalty

---

## Lambda Parameter

| λ   | Behavior               |
| --- | ---------------------- |
| 1.0 | Pure similarity        |
| 0.7 | Balanced (recommended) |
| 0.3 | More diversity         |

---

## MMR Retrieval Flow

```
1. Retrieve many candidates (top 20–50)
2. Pick most relevant
3. Penalize similar docs
4. Select diverse set
```

---

## Why MMR is Powerful in RAG

LLMs need **coverage**, not duplicates.

MMR gives:

* broader context
* fewer hallucinations
* better answers

---

# 9. Mapping — What to Use Where (VERY IMPORTANT)

## Practical Decision Guide

| Scenario              | Similarity  | MMR?       | Why                     |
| --------------------- | ----------- | ---------- | ----------------------- |
| Standard RAG          | Cosine      | ✅ Yes      | Best semantic retrieval |
| QA chatbot            | Cosine      | ✅ Yes      | Avoid duplicates        |
| Research assistant    | Cosine      | ✅ Strongly | Needs coverage          |
| Recommendation system | Dot Product | Optional   | magnitude useful        |
| Image search          | Euclidean   | Optional   | geometric space         |
| Binary ANN index      | Hamming     | No         | hash matching           |

---

## Industry Default Stack

```
Embedding model → cosine similarity
ANN index → HNSW
Retriever → Top-k
Post-process → MMR
LLM → Generation
```

This is used in many production RAG systems.

---

# 10. LangChain Example

### Cosine Similarity Retrieval

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

---

### MMR Retrieval

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)
```

---

# 11. Mental Model (Remember Forever)

Think:

```
Cosine → Find relevant docs
MMR → Find useful set of docs
```

Similarity answers:

> “Which documents match the query?”

MMR answers:

> “Which combination of documents helps the LLM most?”

---

# 12. Final Cheat Sheet

```
IF building RAG → use COSINE
IF duplicates appear → add MMR
IF using DPR → DOT PRODUCT
IF image/vector geometry → EUCLIDEAN
IF binary index → HAMMING
```

---

## One-Line Summary

**Similarity metrics decide WHAT you retrieve.
MMR decides WHICH COMBINATION you send to the LLM.**
