# Contextual Compression Retriever in LangChain

A **complete, step‑by‑step explanation** of Contextual Compression Retrieval using LangChain.

This README explains **every line of code**, concepts, internal workflow, and reasoning behind each step.

---

# Table of Contents

1. Introduction
2. What Problem Are We Solving?
3. High‑Level Architecture
4. Environment Setup
5. Loading Documents
6. Text Chunking
7. Creating Embeddings
8. Building FAISS Vector Store
9. Retriever Basics
10. Running Basic Retrieval
11. RetrievalQA Chain
12. Contextual Compression Retriever
13. LLMChainExtractor (Extraction Compression)
14. LLMChainFilter (Filtering Compression)
15. Measuring Compression Ratio
16. EmbeddingsFilter (Embedding‑based Compression)
17. DocumentCompressorPipeline
18. Multi‑Stage Compression Pipeline
19. ChatOpenAI Integration
20. Final RetrievalQA with Compression
21. Final Output Explanation
22. Internal Flow Diagram
23. Key Advantages
24. When To Use Contextual Compression
25. Summary

---

# 1. Introduction

Contextual Compression Retrieval is an advanced retrieval technique in **Retrieval‑Augmented Generation (RAG)** that:

* Retrieves documents
* Removes irrelevant parts
* Sends only relevant context to the LLM

This improves:

* Accuracy
* Token efficiency
* Cost
* Response quality

---

# 2. What Problem Are We Solving?

Normal RAG problem:

```
User Query → Retriever → Full Documents → LLM
```

Issues:

* Documents contain irrelevant text
* Token limit waste
* Higher cost
* Hallucinations

Solution:

```
User Query → Retriever → Compressor → Relevant Context → LLM
```

---

# 3. High‑Level Architecture

```
                Query
                  │
                  ▼
            Vector Retriever
                  │
                  ▼
        Contextual Compression
          (Filter / Extract)
                  │
                  ▼
              Clean Context
                  │
                  ▼
                 LLM
```

---

# 4. Environment Setup

```python
!pip install langchain_community
!pip install langchain_openai
!pip install faiss-cpu
```

### Explanation

| Package             | Purpose                          |
| ------------------- | -------------------------------- |
| langchain_community | loaders, vectorstores, utilities |
| langchain_openai    | OpenAI LLM + embeddings          |
| faiss-cpu           | similarity search engine         |

---

# 5. Import Required Modules

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```

### What Each Does

* **TextLoader** → Loads text files
* **FAISS** → Vector database
* **OpenAIEmbeddings** → Converts text → vectors
* **CharacterTextSplitter** → Splits documents

---

# 6. Loading Documents

```python
documents = TextLoader("/content/state_of_the_union.txt").load()
```

### Explanation

Loads the State of the Union speech into LangChain Document objects.

Each document contains:

* page_content
* metadata

---

# 7. Text Chunking

```python
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
```

### Why Chunking?

LLMs cannot process very long documents efficiently.

Chunking:

* Improves retrieval precision
* Creates semantic units
* Prevents context loss

### Parameters

| Parameter     | Meaning              |
| ------------- | -------------------- |
| chunk_size    | characters per chunk |
| chunk_overlap | shared context       |

---

# 8. Setup OpenAI API Key

```python
from google.colab import userdata
OPENAI_API_KEY=userdata.get('OPENAI_API_KEY')

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```

### Purpose

Authenticates OpenAI services for:

* embeddings
* LLM calls

---

# 9. Build Vector Store (FAISS)

```python
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()
```

### Internals

1. Convert chunks → embeddings
2. Store vectors in FAISS index
3. Enable similarity search

Mathematically:

```
similarity(query, doc) = cosine(q, d)
```

---

# 10. Basic Retrieval

```python
docs = retriever.invoke("What did the president say about Ketanji Brown Jackson")
```

### What Happens

1. Query → embedding
2. FAISS similarity search
3. Top‑K documents returned

---

# Helper Function

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

Used to display retrieved chunks cleanly.

---

# 11. More Retrieval Queries

```python
docs2 = retriever.invoke("What were the top three priorities outlined in the most recent State of the Union address?")
pretty_print_docs(docs2)
```

and

```python
docs3 = retriever.invoke("How did the President propose to tackle the issue of climate change?")
pretty_print_docs(docs3)
```

---

# 12. Initialize LLM

```python
from langchain_openai import OpenAI
llm=OpenAI(temperature=0)
```

### Temperature = 0

* deterministic
* factual responses

---

# 13. RetrievalQA Chain

```python
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

### Pipeline

```
Query → Retrieve Docs → Inject Context → LLM Answer
```

---

```python
query="What were the top three priorities outlined in the most recent State of the Union address?"
chain.invoke(query)
print(chain.invoke(query)['result'])
```

---

# 14. Contextual Compression Retriever

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
```

Contextual compression reduces document size BEFORE LLM sees it.

---

# 15. LLMChainExtractor (Extraction Compression)

```python
compressor = LLMChainExtractor.from_llm(llm)
```

### Concept

Instead of returning full chunks:

* LLM extracts only relevant sentences.

Example:

```
Original: 500 tokens
Extracted: 80 tokens
```

---

```python
compression_retriever=ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Flow

```
Retriever → LLM Extract Relevant Parts → Return Compressed Docs
```

---

```python
compressed_docs = compression_retriever.invoke(
"What did the president say about Ketanji Jackson Brown")
```

---

# 16. LLMChainFilter (Filtering Compression)

```python
from langchain.retrievers.document_compressors import LLMChainFilter
filter = LLMChainFilter.from_llm(llm)
```

### Difference vs Extractor

| Extractor      | Filter                 |
| -------------- | ---------------------- |
| edits text     | keeps or removes chunk |
| sentence-level | document-level         |

---

```python
compression_retriever2 = ContextualCompressionRetriever(
    base_compressor=filter,
    base_retriever=retriever
)
```

---

# 17. Compression Ratio Measurement

```python
original_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(docs2)]))
```

Original token length.

```python
compressed_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(compressed_docs)]))
```

Compressed length.

```python
print("Original context length:", original_contexts_len)
print("Compressed context length:", compressed_contexts_len)
print("Compressed Ratio:", f"{original_contexts_len/(compressed_contexts_len + 1e-5):.2f}x")
```

Shows efficiency gain.

---

# 18. EmbeddingsFilter (Embedding Compression)

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(embeddings=embeddings)
```

### Concept

Uses cosine similarity instead of LLM reasoning.

Faster + cheaper.

---

```python
compression_retriever3 = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)
```

---

# 19. DocumentCompressorPipeline

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
```

Pipeline allows multi-stage compression.

---

```python
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
```

Splits sentences further.

---

```python
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
```

Removes duplicate semantic chunks.

---

```python
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76
)
```

Keeps only highly relevant content.

---

```python
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
```

### Pipeline Order

1. Split
2. Remove redundancy
3. Keep relevant

---

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)
```

---

# 20. ChatOpenAI LLM

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0)
```

Chat-based model for better reasoning.

---

# 21. RetrievalQA with Compression

```python
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

Now retrieval includes compression automatically.

---

```python
query="What were the top three priorities outlined in the most recent State of the Union address?"
chain.invoke(query)
print(chain.invoke(query)['result'])
```

---

# 22. Final Output Explained

The model identifies three priorities:

1. Combating opioid epidemic
2. Infrastructure & innovation
3. Strengthening domestic production

Because compression ensured only relevant context reached the LLM.

---

# 23. Internal Flow Diagram

```
User Query
    ↓
Retriever (FAISS)
    ↓
Compression Pipeline
    ↓
Filtered Context
    ↓
LLM (ChatOpenAI)
    ↓
Final Answer
```

---

# 24. Key Advantages

| Benefit      | Explanation     |
| ------------ | --------------- |
| Lower Tokens | smaller prompts |
| Faster       | less context    |
| Cheaper      | fewer tokens    |
| Accurate     | less noise      |
| Scalable     | large corpora   |

---

# 25. When To Use Contextual Compression

Use when:

* Documents are large
* Retrieval returns noisy chunks
* Token limits matter
* Production RAG systems

Avoid when:

* very small datasets
* ultra-low latency required

---

# 26. Summary

Contextual Compression Retriever adds an intelligent filtering layer between retrieval and generation.

It transforms RAG from:

```
Retrieve Everything → Hope LLM Finds Answer
```

into:

```
Retrieve → Compress → Deliver Only Signal
```

This is one of the most important optimizations in modern RAG systems.

---

**End of README**
