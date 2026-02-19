# Sparse vs Dense Retrieval in GenAI

---

## 1. Why Retrieval Exists in GenAI

Large Language Models (LLMs):

* do **not remember your documents**
* cannot access private knowledge
* have limited context window

So we use **Retrieval Systems**.

### Goal of Retrieval

Given a query:

```
"What are the side effects of aspirin?"
```

Find **most relevant documents/chunks** from a knowledge base.

This process is called:

> **Information Retrieval (IR)**

In GenAI pipelines (RAG):

```
User Query
     ↓
Retriever  ← (Sparse or Dense)
     ↓
Relevant Documents
     ↓
LLM generates answer
```

---

# 2. What Does "Sparse" and "Dense" Mean?

It refers to **how text is represented mathematically**.

## Representation Space

| Type             | Representation                      |
| ---------------- | ----------------------------------- |
| Sparse Retrieval | Large vector with mostly zeros      |
| Dense Retrieval  | Small vector with continuous values |

---

# 3. Sparse Retrieval

## Definition

Sparse retrieval represents text using **explicit word occurrences**.

Each dimension = a word/token.

Example vocabulary:

```
["apple", "doctor", "health", "eat"]
```

Sentence:

```
"eat apple"
```

Vector:

```
[1, 0, 0, 1]
```

Most positions = 0 → **Sparse vector**.

---

## Intuition

Sparse retrieval answers:

> "Do query and document share the SAME words?"

It relies on **keyword matching**.

---

## Where Used

* Traditional search engines
* Elasticsearch
* BM25 ranking
* Legal search
* Keyword-heavy domains

---

## Main Algorithm: TF-IDF

### Step 1 — Term Frequency (TF)

Measures how often a word appears.

```
TF(t,d) = count of term t in document / total terms in document
```

---

### Step 2 — Inverse Document Frequency (IDF)

Penalizes common words.

```
IDF(t) = log(N / df(t))
```

Where:

* N = total documents
* df(t) = documents containing term t

---

### Step 3 — TF-IDF Score

```
TFIDF(t,d) = TF(t,d) × IDF(t)
```

---

## Similarity Calculation

Usually:

### Cosine Similarity

```
sim(q,d) = (q · d) / (||q|| ||d||)
```

---

## Improved Sparse Method — BM25

BM25 is the **industry standard** sparse retriever.

Score:

```
BM25(q,d) = Σ IDF(t) * ( f(t,d)(k+1) / ( f(t,d)+k(1-b+b|d|/avgdl) ) )
```

Key ideas:

* term frequency saturation
* document length normalization

---

## Advantages

✔ Fast
✔ Interpretable
✔ No training needed
✔ Works well for exact keywords

---

## Limitations

❌ Cannot understand meaning

Example:

```
Query: automobile
Doc: car
```

Sparse → FAIL (different words)

---

# 4. Dense Retrieval

## Definition

Dense retrieval represents text using **neural embeddings**.

Instead of words → meaning vectors.

Example:

```
"car" → [0.21, -0.44, 0.91, ...]
"automobile" → [0.19, -0.40, 0.88, ...]
```

Vectors become **close in semantic space**.

---

## Intuition

Dense retrieval answers:

> "Do query and document MEAN the same thing?"

---

## How Dense Embeddings Are Created

Using Transformer models:

* BERT
* Sentence Transformers
* OpenAI embeddings
* E5
* BGE
* MiniLM

Pipeline:

```
Text
 ↓
Tokenizer
 ↓
Transformer Encoder
 ↓
Embedding Vector (768-dim etc.)
```

---

## Mathematical Representation

Each sentence:

```
x ∈ R^d
```

Example:

```
d = 384 or 768 or 1536
```

Dense vector:

```
[0.12, -0.98, 0.44, ...]
```

(No zeros → dense)

---

## Similarity Computation

### Cosine Similarity

```
sim(q,d) = (q · d) / (||q|| ||d||)
```

OR

### Dot Product

```
q^T d
```

---

## Training Objective (Core Math)

Dense retrievers learn using **contrastive learning**.

Goal:

```
Query closer to relevant docs
Query far from irrelevant docs
```

Loss:

### InfoNCE Loss

```
L = -log( exp(sim(q,d+)) / Σ exp(sim(q,di)) )
```

Where:

* d+ = positive document
* others = negatives

---

## Example

Query:

```
"how to lose weight"
```

Relevant doc:

```
"diet and exercise tips"
```

Model learns to push embeddings closer.

---

## Advantages

✔ Semantic understanding
✔ Handles synonyms
✔ Better for natural language queries
✔ Essential for RAG

---

## Limitations

❌ Requires embedding model
❌ Higher compute cost
❌ Less interpretable

---

# 5. Sparse vs Dense — Core Comparison

| Feature         | Sparse          | Dense             |
| --------------- | --------------- | ----------------- |
| Basis           | Keywords        | Meaning           |
| Representation  | High-dim sparse | Low-dim dense     |
| Training        | No              | Yes               |
| Semantic search | Poor            | Excellent         |
| Speed           | Very fast       | Fast (ANN needed) |
| Used in RAG     | Sometimes       | Mostly            |
| Example         | BM25            | Embeddings        |

---

# 6. Where Used in GenAI

## Sparse Retrieval Used In

* Elasticsearch BM25
* Hybrid search
* Log search
* Legal keyword lookup

---

## Dense Retrieval Used In

* RAG pipelines
* Chat with PDFs
* Semantic search
* QA systems
* Agents memory

---

## Modern Reality

Most systems use:

# Hybrid Retrieval (BEST PRACTICE)

```
Sparse + Dense together
```

Because:

* Sparse → exact match
* Dense → semantic match

---

# 7. Architecture in RAG

```
Documents
   ↓
Chunking
   ↓
Embeddings (dense)
   +
Keyword Index (sparse)
   ↓
Hybrid Retriever
   ↓
Top-k Context
   ↓
LLM
```

---

# 8. Implementation — Sparse Retrieval

## Using TF-IDF (Python)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "cars are fast"
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

query = ["neural networks"]
query_vec = vectorizer.transform(query)

scores = cosine_similarity(query_vec, doc_vectors)
print(scores)
```

---

## Using BM25

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

query = "neural networks".split()
scores = bm25.get_scores(query)
```

---

# 9. Implementation — Dense Retrieval

## Step 1: Install

```bash
pip install sentence-transformers faiss-cpu
```

---

## Step 2: Create Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = ["machine learning", "deep learning", "cars"]
doc_embeddings = model.encode(docs)
```

---

## Step 3: Vector Search (FAISS)

```python
import faiss
import numpy as np

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

index.add(np.array(doc_embeddings))

query = model.encode(["neural networks"])
D, I = index.search(query, k=2)

print(I)
```

---

# 10. Hybrid Retrieval (Industry Standard)

Combine scores:

```
Score = α × Dense + (1-α) × Sparse
```

Example tools:

* Weaviate Hybrid Search
* Pinecone hybrid
* Elasticsearch + embeddings
* LangChain EnsembleRetriever

---

## LangChain Example

```python
from langchain.retrievers import EnsembleRetriever
```

Combine BM25 + vector retriever.

---

# 11. When To Use What

| Scenario       | Best Choice    |
| -------------- | -------------- |
| Exact keywords | Sparse         |
| Semantic QA    | Dense          |
| Enterprise RAG | Hybrid         |
| Small dataset  | Dense          |
| Legal search   | Sparse + Dense |

---

# 12. Mental Model (Very Important)

Think:

```
Sparse Retrieval = Matching WORDS
Dense Retrieval  = Matching MEANING
```

or

```
Sparse → lexical similarity
Dense → semantic similarity
```

---

# 13. Real Production Stack (2025+)

Modern GenAI systems:

```
Query
 ↓
Query Embedding
 ↓
Hybrid Retriever
   ├─ BM25 Index
   └─ Vector DB (HNSW)
 ↓
Reranker (Cross Encoder)
 ↓
LLM
```

---

# Final Summary

## Sparse Retrieval

* keyword-based
* TF-IDF / BM25
* interpretable
* no deep learning

## Dense Retrieval

* embedding-based
* semantic understanding
* neural networks
* core of modern RAG

## Industry Trend

> **Hybrid Retrieval + Reranking = State of the Art**

