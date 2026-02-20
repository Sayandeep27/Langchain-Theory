# Self‑Query Retriever in RAG (Retrieval‑Augmented Generation)

---

## Overview

A **Self‑Query Retriever** is an advanced retrieval technique in **RAG (Retrieval‑Augmented Generation)** where the **LLM automatically converts a user question into structured search filters + semantic search queries** before retrieving documents.

Instead of only doing vector similarity search, the model learns to ask:

* **“What should I search?”**
* **“What filters should I apply?”**

---

## 1. Why Do We Need Self‑Query Retriever?

### Problem with Normal RAG (Similarity Search)

In basic RAG:

```
User Query → Embedding → Vector DB → Top‑k similar chunks
```

The retriever only looks at **semantic similarity**.

### Issue

It cannot understand **metadata constraints**.

#### Example Dataset

| Document | Topic | Year | Author |
| -------- | ----- | ---- | ------ |
| Doc1     | RAG   | 2022 | John   |
| Doc2     | RAG   | 2024 | Alice  |
| Doc3     | LLM   | 2024 | John   |

#### User Query

```
Give me RAG papers after 2023 written by Alice.
```

### Normal Vector Search Behavior

* Searches semantic meaning
* May retrieve wrong documents

Because:

* `after 2023` → numeric filter
* `Alice` → metadata filter

Vector search alone **cannot apply these rules**.

### Solution → Self‑Query Retriever

The LLM converts the question into:

```
Semantic Query: "RAG papers"
Filters:
    author = "Alice"
    year > 2023
```

Then queries the vector database properly.

---

## 2. Core Idea (Simple Definition)

**Self‑Query Retriever = LLM + Metadata‑aware Retrieval**

It allows the LLM to:

* Understand user intent
* Extract structured constraints
* Generate metadata filters
* Perform filtered vector search

---

## 3. Architecture

```
                User Question
                       │
                       ▼
              LLM Query Analyzer
        (creates structured query)
                       │
        ┌──────────────┴──────────────┐
        │                               │
Semantic Search Query            Metadata Filters
        │                               │
        └──────────────┬──────────────┘
                       ▼
                 Vector Database
                       ▼
                Relevant Documents
                       ▼
                     LLM
                       ▼
                    Answer
```

---

## 4. How Self‑Query Retriever Works (Step‑by‑Step)

### Step 1 — User asks question

```
"Show ML papers published after 2022 about transformers"
```

### Step 2 — LLM parses query

The LLM generates:

```
query = "transformers machine learning"
filter = year > 2022
```

### Step 3 — Structured query created

Internally something like:

```json
{
  "query": "transformers machine learning",
  "filter": {
    "year": { "$gt": 2022 }
  }
}
```

### Step 4 — Vector DB executes hybrid retrieval

* semantic similarity
* * metadata filtering

### Step 5 — Retrieved docs → LLM answer

---

## 5. Key Components

### (1) Documents with Metadata

Self‑query requires metadata.

Example:

```python
Document(
   page_content="RAG improves LLM accuracy",
   metadata={
       "author": "Alice",
       "year": 2024,
       "topic": "RAG"
   }
)
```

Without metadata → self‑query retriever is useless.

---

### (2) Metadata Schema Description

You must tell the LLM what metadata exists.

Example:

```python
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="Publication year",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="Author name",
        type="string",
    ),
]
```

This acts like a **database schema** for the LLM.

---

### (3) Query Constructor (LLM)

LLM translates:

```
Natural Language → Structured Query
```

This is the **“self‑query”** part.

---

## 6. LangChain Implementation (Conceptual)

### Step 1 — Install

```bash
pip install langchain langchain-community
```

### Step 2 — Create Documents

```python
from langchain.schema import Document

docs = [
    Document(
        page_content="RAG paper",
        metadata={"author": "Alice", "year": 2024}
    )
]
```

### Step 3 — Define Metadata Fields

```python
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="author",
        description="Author of the paper",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="Publication year",
        type="integer"
    ),
]
```

### Step 4 — Create Self Query Retriever

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description="Research papers",
    metadata_field_info=metadata_field_info
)
```

### Step 5 — Query

```python
docs = retriever.get_relevant_documents(
    "papers by Alice after 2023"
)
```

The LLM automatically creates filters.

---

## 7. What Makes It Powerful?

| Traditional Retriever       | Self‑Query Retriever                     |
| --------------------------- | ---------------------------------------- |
| Query → Similarity Search   | Query → Reasoning → Structured Retrieval |
| No constraint understanding | Understands constraints                  |
| No metadata filters         | Metadata filters applied                 |
| Basic retrieval             | Intelligent retrieval                    |

It adds **reasoning before retrieval**.

---

## 8. Comparison with Other Retrieval Methods

| Method               | Understand Filters | Uses Metadata | Intelligent Query |
| -------------------- | ------------------ | ------------- | ----------------- |
| Similarity Search    | No                 | No            | No                |
| BM25                 | No                 | No            | No                |
| Hybrid Search        | Partial            | Limited       | No                |
| Parent‑Child         | Context fix        | No            | No                |
| Self‑Query Retriever | Yes                | Yes           | Yes               |

---

## 9. When to Use Self‑Query Retriever

Use it when your data has:

* dates
* categories
* authors
* ratings
* prices
* tags
* structured attributes

### Examples

* Research paper search
* Product catalog RAG
* Legal documents
* Movie recommendation systems
* Enterprise knowledge bases

---

## 10. Real‑World Example

### User asks:

```
Show affordable laptops under ₹80,000 released after 2022.
```

### Self‑Query converts to:

```
semantic_query: "laptop"
filters:
    price < 80000
    year > 2022
```

Huge improvement over plain vector search.

---

## 11. Advantages

* Very accurate retrieval
* Natural language filtering
* Works like SQL + semantic search
* Reduces irrelevant chunks
* Better grounding → less hallucination

---

## 12. Limitations

* Requires good metadata
* Slightly slower (extra LLM step)
* Needs schema definition
* LLM may occasionally misinterpret filters

---

## 13. Mental Model (Very Important)

Think of Self‑Query Retriever as:

> **LLM acting like a Database Query Planner**

It converts:

```
English → Semantic Search + WHERE clause
```

Just like SQL:

```sql
SELECT *
FROM documents
WHERE author="Alice"
AND year > 2023
```

---

## One‑Line Summary

**Self‑Query Retriever is a metadata‑aware RAG retriever where an LLM automatically converts natural language into structured search queries and filters before retrieving documents.**
