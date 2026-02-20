# Contextual Compression in RAG — Complete Guide

---

# Table of Contents

1. What is Contextual Compression?
2. Why Contextual Compression is Needed
3. RAG Flow Overview
4. Types of Contextual Compression
5. LLMChainExtractor
6. LLMChainFilter
7. EmbeddingsFilter (Embedding Compression)
8. DocumentCompressorPipeline (Multi‑Stage Compression)
9. Full Production RAG Flow
10. Comparison Tables
11. Best Practices

---

# 1. What is Contextual Compression?

**Contextual Compression** is a post‑retrieval step in RAG where retrieved documents are filtered or rewritten so that only query‑relevant information is passed to the LLM.

### Definition

> Contextual Compression reduces noise in retrieved documents before generation.

---

# 2. Why Contextual Compression is Needed

## Problem in Basic RAG

```
User Query
     ↓
Retriever
     ↓
Top‑K Chunks
     ↓
LLM
```

Issues:

* Retrieved chunks contain irrelevant text
* Token wastage
* Poor answers
* Context window overflow

Example:

```
Machine learning history...
CNN architectures...
Overfitting explanation...
Random unrelated information...
```

Only a small portion is useful.

---

# 3. Improved RAG Flow

```
User Query
     ↓
Retriever
     ↓
Contextual Compression
     ↓
LLM
```

Compression ensures the LLM receives **dense, relevant context**.

---

# 4. Types of Contextual Compression

| Method            | Purpose                    | Uses LLM | Edits Text |
| ----------------- | -------------------------- | -------- | ---------- |
| LLMChainExtractor | Extract relevant sentences | Yes      | Yes        |
| LLMChainFilter    | Keep or remove documents   | Yes      | No         |
| EmbeddingsFilter  | Similarity filtering       | No       | No         |
| Pipeline          | Multi‑stage compression    | Optional | Mixed      |

---

# 5. LLMChainExtractor

## Concept

Extracts only relevant parts of documents using an LLM.

### Pipeline

```
Retriever → Extractor → Compressed Docs
```

### Code Example

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
```

### What Happens Internally

1. Retriever fetches documents.
2. LLM receives `(query + document)`.
3. LLM extracts only relevant sentences.
4. Smaller documents returned.

---

# 6. LLMChainFilter

## Concept

LLM decides whether a document should be kept or discarded.

### Pipeline

```
Retriever → LLM Filter → Relevant Docs
```

### Code Example

```python
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever

filter = LLMChainFilter.from_llm(llm)

compression_retriever2 = ContextualCompressionRetriever(
    base_compressor=filter,
    base_retriever=retriever
)
```

### Behavior

* Document unchanged
* Only relevant docs remain

### Internal Logic

```
If LLM says YES → keep
If NO → discard
```

---

# 7. EmbeddingsFilter (Embedding Compression)

## Concept

Uses **cosine similarity** instead of LLM reasoning.

Fast and inexpensive.

### Pipeline

```
Retriever → Embedding Similarity Filter
```

### Code Example

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

embeddings = OpenAIEmbeddings()

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings
)

compression_retriever3 = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)
```

### Internal Steps

1. Embed query
2. Embed documents
3. Compute cosine similarity
4. Remove weak matches

---

# 8. DocumentCompressorPipeline (Multi‑Stage Compression)

## Concept

Allows multiple compression stages sequentially.

```
Split → Remove Redundancy → Keep Relevant
```

---

## Step 1 — Splitter

```python
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=". "
)
```

Breaks retrieved documents into smaller pieces.

---

## Step 2 — Redundant Filter

```python
from langchain_community.document_transformers import EmbeddingsRedundantFilter

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
```

Removes semantically duplicate chunks.

---

## Step 3 — Relevant Filter

```python
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76
)
```

Keeps only highly relevant content.

---

## Step 4 — Build Pipeline

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
```

---

## Step 5 — Wrap Retriever

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)
```

---

### Execution Flow

```
Retriever
   ↓
Split Documents
   ↓
Remove Duplicates
   ↓
Filter Relevant Chunks
   ↓
Compressed Context
```

---

# 9. Full Production RAG Flow

```
User Query
   ↓
Retriever (Vector / Hybrid)
   ↓
Reranker
   ↓
Contextual Compression Pipeline
   ↓
LLM Generation
```

---

# 10. Comparison Summary

| Feature      | Extractor | LLM Filter | Embedding Filter | Pipeline |
| ------------ | --------- | ---------- | ---------------- | -------- |
| Uses LLM     | Yes       | Yes        | No               | Optional |
| Removes Docs | No        | Yes        | Yes              | Yes      |
| Edits Text   | Yes       | No         | No               | Mixed    |
| Speed        | Slow      | Medium     | Fast             | Medium   |
| Cost         | High      | Medium     | Low              | Balanced |

---

# 11. Best Practices

## Recommended Order

```
Retrieve Broadly
Rank Carefully
Compress Intelligently
Generate Accurately
```

## Production Strategy

* Use embedding filtering for speed
* Use LLM filtering for semantic precision
* Use extractor for token optimization
* Use pipeline for enterprise RAG

---

# Key Insight

Modern RAG is not just retrieval — it is **information distillation before generation**.

---

# End of Document
