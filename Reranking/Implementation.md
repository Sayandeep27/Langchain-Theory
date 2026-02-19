# Semantic Search and Reranking Pipeline (Step‑by‑Step)
---

# Table of Contents

1. Dataset Creation
2. Installing Dependencies
3. Loading Sentence Transformer Model
4. Generating Document Embeddings
5. Query Embedding
6. Cosine Similarity Search
7. Ranking Documents
8. Selecting Top‑K Documents
9. BM25 Reranking (Sparse Retrieval)
10. Cross‑Encoder Reranking
11. Cohere Reranking API
12. Complete Retrieval Pipeline Summary

---

# 1. Dataset Creation

```python
documents = [
    "This is a list which containing sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings.",
    "Understanding document structure aids in keyword extraction.",
    "Efficient keyword extraction enhances search accuracy.",
    "Semantic similarity improves document retrieval performance.",
    "Machine learning algorithms can optimize keyword extraction methods."
]
```

## Explanation

We create a small corpus of **8 documents**.

Each document represents a searchable text entry.

These will later be converted into numerical vectors (embeddings).

---

# 2. Installing Dependencies

```python
!pip install sentence_transformers
```

## Explanation

Installs the **Sentence Transformers** library which provides:

* Pretrained embedding models
* Semantic similarity capabilities
* Easy vector generation

---

# 3. Loading Sentence Transformer Model

```python
from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
model = SentenceTransformer(model_name)
```

## Explanation

We load a pretrained multilingual embedding model.

### Model Used

`paraphrase-xlm-r-multilingual-v1`

Features:

| Feature             | Description                      |
| ------------------- | -------------------------------- |
| Multilingual        | Supports many languages          |
| Sentence embeddings | Converts text → vectors          |
| Semantic similarity | Similar meaning → closer vectors |

This model outputs dense vector embeddings.

---

# 4. Inspect Documents

```python
documents
len(documents)
```

## Explanation

* Displays documents
* Confirms total number of documents

Output: **8 documents**

---

# 5. Generate Document Embeddings

```python
document_embeddings = model.encode(documents)
```

## Explanation

Each document is converted into a numerical vector.

Conceptually:

```
Text → Neural Network → Vector Representation
```

Example:

```
"keyword extraction" → [0.12, -0.44, 0.98, ...]
```

These vectors capture **semantic meaning**.

---

## Embedding Dimensions

```python
len(document_embeddings)
len(document_embeddings[0])
```

Explanation:

* First line → number of embeddings (8)
* Second line → embedding size (vector dimension)

---

## View Embeddings

```python
for i, embedding in enumerate(document_embeddings):
    print(f"Document {i+1} embedding: {embedding}")
```

Displays numerical vectors for each document.

---

# 6. Define Query

```python
query = "Natural language processing techniques enhance keyword extraction efficiency."
```

## Explanation

This is the **user search query**.

We want to find the most relevant documents.

---

# 7. Query Embedding

```python
query_embedding = model.encode(query)
print("Query embedding:", query_embedding)
len(query_embedding)
```

## Explanation

The query is converted into the same vector space as documents.

Important rule:

> Query and documents must use the SAME embedding model.

---

# 8. Cosine Similarity Search

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(np.array([query_embedding]), document_embeddings)
```

## Explanation

Cosine similarity measures angle similarity between vectors.

Formula:

```
cos(θ) = (A · B) / (||A|| ||B||)
```

Range:

| Value | Meaning      |
| ----- | ------------ |
| 1     | Very similar |
| 0     | Unrelated    |
| -1    | Opposite     |

---

# 9. Find Most Similar Document

```python
most_similar_index = np.argmax(similarities)
most_similar_document = documents[most_similar_index]
similarity_score = similarities[0][most_similar_index]
```

Explanation:

* `argmax` finds highest similarity score
* Retrieves best matching document

---

# 10. Rank All Documents

```python
sorted_indices = np.argsort(similarities[0])[::-1]
ranked_documents = [(documents[i], similarities[0][i]) for i in sorted_indices]
```

## Explanation

Steps:

1. Sort similarity scores
2. Reverse order (highest first)
3. Attach document text with score

---

## Print Rankings

```python
print("Ranked Documents:")
for rank, (document, similarity) in enumerate(ranked_documents, start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")
```

Displays ranked search results.

---

# 11. Select Top‑4 Documents

```python
print("Top 4 Documents:")
for rank, (document, similarity) in enumerate(ranked_documents[:4], start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")
```

## Explanation

We keep only **Top‑K results**.

Why?

* Faster reranking
* Reduced computation
* Standard retrieval pipeline step

---

# 12. Install BM25 Library

```python
!pip install rank_bm25
```

---

# 13. BM25 Reranking (Sparse Retrieval)

```python
from rank_bm25 import BM25Okapi
```

BM25 is a **keyword-based ranking algorithm**.

Dense embeddings → semantic meaning
BM25 → keyword matching

---

## Extract Top‑4 Documents

```python
top_4_documents = [doc[0] for doc in ranked_documents[:4]]
```

---

## Tokenization

```python
tokenized_top_4_documents = [doc.split() for doc in top_4_documents]
tokenized_query = query.split()
```

Explanation:

BM25 works on tokens (words).

---

## Build BM25 Index

```python
bm25 = BM25Okapi(tokenized_top_4_documents)
```

Creates inverted index internally.

---

## Compute BM25 Scores

```python
bm25_scores = bm25.get_scores(tokenized_query)
```

Scores based on:

* Term frequency
* Inverse document frequency
* Document length normalization

---

## Rerank Documents

```python
sorted_indices2 = np.argsort(bm25_scores)[::-1]
reranked_documents = [(top_4_documents[i], bm25_scores[i]) for i in sorted_indices2]
```

BM25 produces a new ranking.

---

## Print BM25 Rerank

```python
print("Rerank of top 4 Documents:")
for rank, (document, similarity) in enumerate(reranked_documents, start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")
```

---

# 14. Cross‑Encoder Reranking

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

## What is a Cross‑Encoder?

Unlike bi‑encoder embeddings:

| Bi‑Encoder        | Cross‑Encoder   |
| ----------------- | --------------- |
| Encode separately | Encode together |
| Fast              | Slower          |
| Retrieval         | Reranking       |

Cross‑encoder reads:

```
[Query + Document] together
```

and predicts relevance directly.

---

## Create Query‑Document Pairs

```python
pairs = []
for doc in top_4_documents:
    pairs.append([query, doc])
```

---

## Predict Scores

```python
scores = cross_encoder.predict(pairs)
```

Produces relevance scores.

---

## Sort by Cross‑Encoder Score

```python
scored_docs = zip(scores, top_4_documents)
reranked_document_cross_encoder = sorted(scored_docs, reverse=True)
```

Higher score → more relevant.

---

# 15. Cohere Reranking API

```python
!pip install cohere
```

---

```python
import cohere

co = cohere.Client("jnfdonfod********************************")
```

Creates Cohere API client.

---

## Perform Reranking

```python
response = co.rerank(
    model="rerank-english-v3.0",
    query="Natural language processing techniques enhance keyword extraction efficiency.",
    documents=top_4_documents,
    return_documents=True
)
```

Explanation:

Cohere uses a large transformer reranker trained for search relevance.

---

## View Results

```python
print(response)
response.results[0].document.text
response.results[0].relevance_score
```

---

## Print All Reranked Results

```python
for i in range(4):
  print(f'text: {response.results[i].document.text} score: {response.results[i].relevance_score}')
```

---

# 16. Retrieval Pipeline Flow

```
User Query
     ↓
Dense Embedding Search (SentenceTransformer)
     ↓
Top‑K Documents
     ↓
BM25 Rerank (Keyword Matching)
     ↓
Cross‑Encoder Rerank (Deep Relevance)
     ↓
Cohere Rerank (LLM‑level Ranking)
     ↓
Final Results
```

---

# 17. Key Concepts Summary

| Stage         | Technique         | Purpose                  |
| ------------- | ----------------- | ------------------------ |
| Retrieval     | Dense Embeddings  | Semantic search          |
| Similarity    | Cosine Similarity | Initial ranking          |
| Sparse Rerank | BM25              | Keyword relevance        |
| Neural Rerank | Cross‑Encoder     | Deep matching            |
| LLM Rerank    | Cohere            | Production‑grade ranking |

---

# 18. Why Multi‑Stage Retrieval Works

Single method limitations:

* Dense → may miss keywords
* Sparse → misses semantics
* Cross‑encoder → expensive

Combined pipeline gives:

* Speed
* Accuracy
* Semantic understanding
* Keyword precision

This architecture is used in:

* RAG systems
* Search engines
* AI assistants
* Enterprise document search

---

# 19. Final Notes

This notebook demonstrates a **modern hybrid retrieval system** combining:

* Dense Retrieval
* Sparse Retrieval
* Neural Reranking
* LLM Reranking

This is the foundation of advanced Retrieval‑Augmented Generation (RAG) systems.

---

**End of README**
