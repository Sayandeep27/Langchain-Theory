# Bi‑Encoder vs Cross‑Encoder

---

# 1. Introduction

In modern **Natural Language Processing (NLP)** and **Information Retrieval systems**, especially in **semantic search**, **Retrieval‑Augmented Generation (RAG)**, and **question answering**, two important architectures are widely used:

* **Bi‑Encoder**
* **Cross‑Encoder**

Both are based on Transformer models (like BERT, RoBERTa, etc.), but they differ fundamentally in **how inputs are processed** and **how similarity is computed**.

Understanding these architectures is critical for building scalable AI systems.

---

# 2. Why Do We Need Encoder Architectures?

Traditional keyword search fails because:

* It relies on exact word matching
* Cannot capture semantic meaning
* Fails with paraphrases

Example:

Query: "How to learn machine learning?"
Document: "Guide to studying ML"

Keyword search → low similarity
Semantic encoding → high similarity

Encoders convert text into **dense vector embeddings** that capture meaning.

---

# 3. What is an Encoder in NLP?

An **encoder** converts text into a numerical vector representation.

```
Text → Tokenization → Transformer → Embedding Vector
```

Example:

```
"Machine learning is amazing"
      ↓
[0.21, -0.88, 1.02, ...]
```

These embeddings allow similarity computation using:

* Cosine similarity
* Dot product
* Euclidean distance

---

# 4. Bi‑Encoder — Detailed Explanation

## Definition

A **Bi‑Encoder** encodes two inputs **independently** using the same encoder model.

```
Query Encoder      Document Encoder
     |                   |
   Vector Q           Vector D
          \           /
           Similarity
```

Both encoders share weights.

---

## Workflow

1. Encode query separately
2. Encode documents separately
3. Store document embeddings
4. Compare vectors using similarity metrics

---

## Key Idea

Similarity is computed **after encoding**, not during encoding.

---

## Example

```
Q = Encoder("What is AI?")
D = Encoder("Artificial Intelligence explained")

score = cosine_similarity(Q, D)
```

---

## Characteristics

* Independent encoding
* Pre‑computable embeddings
* Extremely fast retrieval
* Scalable to millions of documents

---

# 5. Cross‑Encoder — Detailed Explanation

## Definition

A **Cross‑Encoder** processes query and document **together** in a single forward pass.

```
[CLS] Query [SEP] Document [SEP]
            ↓
      Transformer Model
            ↓
      Relevance Score
```

---

## Workflow

1. Combine query and document
2. Feed into transformer jointly
3. Model learns token‑level interactions
4. Output relevance score

---

## Example

```
score = Model("What is AI?", "Artificial Intelligence explained")
```

---

## Key Idea

The model directly learns **cross‑attention interactions** between tokens.

---

## Characteristics

* Joint encoding
* High accuracy
* Expensive computation
* Cannot precompute embeddings

---

# 6. Architecture Comparison

| Component   | Bi‑Encoder     | Cross‑Encoder   |
| ----------- | -------------- | --------------- |
| Encoding    | Separate       | Joint           |
| Interaction | After encoding | During encoding |
| Speed       | Very Fast      | Slow            |
| Accuracy    | Medium‑High    | Very High       |
| Scalability | Excellent      | Poor            |

---

# 7. Working Mechanism (Step‑by‑Step)

## Bi‑Encoder

```
Step 1: Encode query
Step 2: Encode documents
Step 3: Compute similarity
Step 4: Rank results
```

## Cross‑Encoder

```
Step 1: Pair query with each document
Step 2: Joint transformer encoding
Step 3: Output relevance score
Step 4: Rank results
```

---

# 8. Mathematical Intuition

## Bi‑Encoder

Let:

```
f(q) = query embedding
f(d) = document embedding
```

Similarity:

```
s(q,d) = cosine(f(q), f(d))
```

---

## Cross‑Encoder

```
s(q,d) = Transformer([q,d])
```

Model directly predicts relevance.

---

# 9. Computational Complexity

Assume:

* N documents

### Bi‑Encoder

```
Encoding cost: N (offline)
Query time: O(N similarity)
```

### Cross‑Encoder

```
Query time: O(N forward passes)
```

Huge difference at scale.

---

# 10. Speed vs Accuracy Trade‑off

| Metric     | Bi‑Encoder | Cross‑Encoder |
| ---------- | ---------- | ------------- |
| Latency    | Low        | High          |
| Throughput | High       | Low           |
| Precision  | Good       | Excellent     |

---

# 11. Training Differences

## Bi‑Encoder Training

Uses contrastive learning:

```
(anchor, positive, negative)
```

Objective:

* Pull similar texts closer
* Push dissimilar apart

Loss examples:

* Triplet loss
* Contrastive loss
* Multiple negatives ranking loss

---

## Cross‑Encoder Training

Supervised classification/regression:

```
(query, document, label)
```

Output:

```
relevance score
```

---

# 12. Inference Pipeline Differences

## Bi‑Encoder

```
Offline:
  Encode all documents
  Store in vector DB

Online:
  Encode query
  Vector search
```

## Cross‑Encoder

```
For each candidate:
  Joint inference
```

---

# 13. Use Cases

## Bi‑Encoder

* Semantic search
* Vector databases
* RAG retrieval
* Recommendation systems

## Cross‑Encoder

* Re‑ranking
* High precision search
* Answer validation

---

# 14. Real‑World RAG Pipeline Usage

Typical production pipeline:

```
User Query
    ↓
Bi‑Encoder Retrieval (Fast)
    ↓
Top‑K Documents
    ↓
Cross‑Encoder Re‑Ranking (Accurate)
    ↓
LLM Generation
```

---

# 15. Retrieval + Re‑Ranking Strategy

This hybrid approach combines:

* Speed of Bi‑Encoder
* Accuracy of Cross‑Encoder

Industry standard architecture.

---

# 16. Advantages and Disadvantages

## Bi‑Encoder

### Advantages

* Scalable
* Fast inference
* Embedding reuse

### Disadvantages

* Limited interaction modeling

---

## Cross‑Encoder

### Advantages

* Deep token interaction
* Highest ranking accuracy

### Disadvantages

* Computationally expensive
* Not scalable for large corpora

---

# 17. Comparison Table (Complete)

| Feature           | Bi‑Encoder  | Cross‑Encoder |
| ----------------- | ----------- | ------------- |
| Input Processing  | Separate    | Joint         |
| Embedding Storage | Yes         | No            |
| Precomputation    | Possible    | Impossible    |
| ANN Search        | Supported   | Not Supported |
| Latency           | Low         | High          |
| Accuracy          | Medium‑High | Very High     |
| Best Role         | Retrieval   | Re‑ranking    |

---

# 18. Example Implementations (Python)

## Bi‑Encoder Example (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What is artificial intelligence?"
docs = [
    "AI is simulation of human intelligence",
    "Cooking recipes for beginners"
]

q_emb = model.encode(query)
d_emb = model.encode(docs)

scores = util.cos_sim(q_emb, d_emb)
print(scores)
```

---

## Cross‑Encoder Example

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
    ("What is artificial intelligence?",
     "AI is simulation of human intelligence")
]

scores = model.predict(pairs)
print(scores)
```

---

# 19. When to Use What?

| Scenario                 | Recommended Model |
| ------------------------ | ----------------- |
| Million documents search | Bi‑Encoder        |
| Final ranking            | Cross‑Encoder     |
| Real‑time search         | Bi‑Encoder        |
| High precision QA        | Cross‑Encoder     |

---

# 20. Interview‑Level Explanation

**Bi‑Encoder:** Encodes texts separately and compares embeddings using similarity.

**Cross‑Encoder:** Encodes query and document together to directly predict relevance.

---

# 21. Common Mistakes

* Using Cross‑Encoder for full corpus search
* Expecting Bi‑Encoder accuracy equal to Cross‑Encoder
* Skipping re‑ranking in RAG pipelines

---

# 22. Advanced Concepts

* Hard negative mining
* Late interaction models (ColBERT)
* Hybrid retrieval systems
* Multi‑stage ranking pipelines

---

# 23. Summary

Bi‑Encoders prioritize **speed and scalability**, while Cross‑Encoders prioritize **accuracy and deep interaction modeling**.

Modern AI systems use **both together**.

---

# 24. Key Takeaways

* Bi‑Encoder → Fast retrieval
* Cross‑Encoder → Accurate ranking
* Hybrid pipeline → Best performance
* Industry standard for RAG systems

---

**End of Document**
