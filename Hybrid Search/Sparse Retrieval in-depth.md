# Sparse Retrieval, TF‑IDF, and BM25 — Complete Guide

---

## 1. What is Sparse Retrieval?

**Sparse retrieval** represents documents and queries using **sparse vectors**, meaning:

* Vocabulary size may be **50,000+ words**
* Each document contains only a few of those words
* Most vector values = **0**

### Example Vocabulary

```
["machine", "learning", "dog", "cat", "finance","is","powerful"]
```

### Document

```
"machine learning is powerful"
```

### Vector Representation

```
[1, 1, 0, 0, 0,1,1]
```

Mostly zeros → **Sparse Vector**

---

### Why Sparse Retrieval?

**Goal:**
Find documents containing important query words.

Unlike dense embeddings:

* No semantic understanding
* Purely keyword/statistical matching

Used in:

* Google (early search engines)
* Elasticsearch
* BM25 retrievers
* Hybrid Search in RAG

---

## 2. Core Idea Behind Sparse Retrieval

Sparse retrieval answers:

> Which documents contain important words from the query, and how important are those words?

Two main concepts:

* **Term Frequency (TF)** → word importance inside document
* **Inverse Document Frequency (IDF)** → word rarity across corpus

---

## 3. TF‑IDF (Term Frequency – Inverse Document Frequency)

TF‑IDF is the classic sparse retrieval algorithm.

---

### 3.1 Term Frequency (TF)

Measures:

How often a word appears in a document.

### Formula

```
TF(t,d) = count of term t in document d / total words in document
```

### Example

Document:

```
D1: "machine learning machine"
```

Word counts:

| Word     | Count |
| -------- | ----- |
| machine  | 2     |
| learning | 1     |

Total words = 3

TF:

```
TF(machine) = 2/3 = 0.67
TF(learning) = 1/3 = 0.33
```

---

### 3.2 Inverse Document Frequency (IDF)

Measures:

How rare a word is across all documents.

Rare words are more important.

### Formula

```
IDF(t) = log(N / df(t))
```

Where:

* N = total documents
* df(t) = documents containing term t

### Example Corpus

```
D1: machine learning
D2: machine vision
D3: deep learning
```

| Word     | Appears in | df |
| -------- | ---------- | -- |
| machine  | D1, D2     | 2  |
| learning | D1, D3     | 2  |
| deep     | D3         | 1  |

If N = 3:

```
IDF(machine) = log(3/2)
IDF(deep) = log(3/1)  ← higher importance
```

Rare word = higher score.

---

### 3.3 TF‑IDF Score

```
TFIDF(t,d) = TF(t,d) × IDF(t)
```

Meaning:

Important if frequent in document **AND** rare globally.

---

### 3.4 Retrieval Using TF‑IDF

Steps:

1. Convert documents → TF‑IDF vectors
2. Convert query → TF‑IDF vector
3. Compute similarity (usually cosine similarity)
4. Return top‑k documents

### Example

Query:

```
"deep learning"
```

Document scores depend on:

* presence of "deep"
* rarity of "deep"

Document containing rare word ranks higher.

---

### Problem with TF‑IDF

TF‑IDF has weaknesses:

* Long documents get unfair advantage
* Term frequency grows linearly (not realistic)
* No normalization tuning
* Weak ranking quality

→ This led to **BM25**.

---

## 4. BM25 (Best Matching 25)

BM25 is an improved version of TF‑IDF.

It is the **industry standard sparse retriever**.

Used in:

* Elasticsearch
* OpenSearch
* Modern RAG pipelines

---

### Key Improvements Over TF‑IDF

| Problem                        | BM25 Solution        |
| ------------------------------ | -------------------- |
| Term frequency grows endlessly | Saturation effect    |
| Long documents dominate        | Length normalization |
| No tuning                      | Tunable parameters   |

---

## 5. BM25 Formula (Intuition First)

BM25 score:

```
Score(D,Q) = Σ IDF(qᵢ) · f(qᵢ,D)(k₁ + 1)
             ----------------------------------
             f(qᵢ,D) + k₁(1 − b + b |D| / avgdl)
```

Don’t worry — we simplify it.

---

### Components Explained

#### 1. IDF(qᵢ)

Same idea as TF‑IDF:

Rare terms matter more.

---

#### 2. f(qᵢ, D)

Term frequency of word in document.

BUT BM25 adds **saturation**:

* 1 occurrence → big gain
* 10 occurrences → small extra gain

Because repeating words endlessly shouldn’t dominate ranking.

---

#### 3. Document Length Normalization

```
|D| / avgdl
```

Where:

* |D| = document length
* avgdl = average document length

Prevents long documents from winning unfairly.

---

#### 4. Parameters

**k₁ (usually 1.2–2.0)**

Controls TF importance.

Higher k₁:

TF matters more.

**b (usually 0.75)**

Controls length normalization.

* b = 1 → full normalization
* b = 0 → ignore length

---

## 6. BM25 Example (Intuition)

Corpus:

```
D1: "machine learning basics"
D2: "machine machine machine learning advanced tutorial"
```

Query:

```
"machine learning"
```

### TF‑IDF behavior

D2 wins strongly because:

* machine appears 3 times

### BM25 behavior

BM25 says:

After some repetitions, extra words add little value.

So D2 gets only slightly higher score.

More realistic ranking.

---

## 7. TF‑IDF vs BM25 Comparison

| Feature              | TF‑IDF  | BM25            |
| -------------------- | ------- | --------------- |
| Era                  | Classic | Modern standard |
| Term frequency       | Linear  | Saturated       |
| Length normalization | Weak    | Strong          |
| Tunable              | No      | Yes             |
| Ranking quality      | Medium  | High            |
| Used in RAG          | Rarely  | Very common     |

---

## 8. Sparse Retrieval in RAG Systems

Typical pipeline:

```
User Query
     ↓
BM25 Retriever
     ↓
Top‑K Documents
     ↓
LLM
```

### Why BM25 works well

* Exact keyword matching
* Handles technical terms
* Works without embeddings
* Fast and cheap

---

## 9. Example in Python

### TF‑IDF Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer


docs = [
    "machine learning basics",
    "deep learning tutorial",
    "finance and banking"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

query = vectorizer.transform(["deep learning"])

scores = (X * query.T).toarray()
print(scores)
```

---

### BM25 Example

Install:

```bash
pip install rank-bm25
```

Code:

```python
from rank_bm25 import BM25Okapi


docs = [
    "machine learning basics",
    "deep learning tutorial",
    "finance and banking"
]


tokenized_docs = [doc.split() for doc in docs]

bm25 = BM25Okapi(tokenized_docs)

query = "deep learning".split()

scores = bm25.get_scores(query)
print(scores)
```

---

## 10. When to Use TF‑IDF vs BM25

### Use TF‑IDF when:

* learning concepts
* small datasets
* simple similarity tasks

### Use BM25 when:

* building search engines
* RAG retrievers
* production systems
* hybrid search

---

## 11. Sparse vs Dense Retrieval (Big Picture)

| Sparse               | Dense                |
| -------------------- | -------------------- |
| Keyword matching     | Semantic meaning     |
| Explainable          | Black‑box embeddings |
| Fast                 | Compute heavy        |
| BM25                 | Vector DB embeddings |
| Good for exact terms | Good for paraphrases |

Modern systems use:

```
Hybrid Search = BM25 + Dense Retrieval
```

---

## 12. Simple Mental Model (Very Important)

Think of:

* **TF** → How loud a word speaks inside document
* **IDF** → How rare the word is in the world
* **BM25** → Smart judge balancing both fairly

---

**End of Document**
