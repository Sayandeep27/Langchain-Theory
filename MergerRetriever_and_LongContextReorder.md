# Multi‑Document RAG Pipeline with LOTR (Lord of the Retrievers) + Contextual Compression

---

## Overview

This project demonstrates how to build an **Advanced Retrieval‑Augmented Generation (RAG)** pipeline using:

* **LangChain**
* **ChromaDB (Vector Database)**
* **HuggingFace Embeddings**
* **Merged Retrievers (LOTR)**
* **Contextual Compression**
* **Local LLM using Llama.cpp**

The system loads **multiple PDFs**, converts them into embeddings, stores them in vector databases, merges retrievers, refines retrieved context, and finally answers questions using a local LLM.

---

# Architecture Flow

```
PDFs → Loader → Chunking → Embeddings → Vector DB → Retrievers
        ↓
   LOTR (Merge Retrievers)
        ↓
Contextual Compression Pipeline
        ↓
        LLM (Zephyr‑7B)
        ↓
      Final Answer
```

---

# Step‑by‑Step Explanation

---

# 1. Environment Setup

## Install Required Libraries

```python
!pip install -qU langchain chromadb huggingface_hub sentence-transformers pypdf openai tiktoken
```

### Purpose

Installs core dependencies required for:

| Library               | Purpose                     |
| --------------------- | --------------------------- |
| langchain             | RAG orchestration framework |
| chromadb              | Vector database             |
| sentence-transformers | Embedding models            |
| pypdf                 | PDF parsing                 |
| openai                | Optional embedding provider |
| tiktoken              | Token counting              |

---

```python
!pip install -U langchain-community
```

### Why?

Community integrations (document loaders, embeddings, retrievers).

---

# 2. Load the Data (PDF Documents)

```python
from langchain.document_loaders import PyPDFLoader
```

### Purpose

Loads PDF files and converts them into LangChain **Document objects**.

---

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mounts Google Drive to access files.

---

```python
loader_harrypotter  = PyPDFLoader("/content/harry_potter_book.pdf")
```

Creates loader for Harry Potter book.

---

```python
documnet_harrypotter = loader_harrypotter.load()
```

Loads PDF into documents.

Each page becomes:

```
Document(
  page_content="text...",
  metadata={page_number, source}
)
```

---

```python
print(len(documnet_harrypotter))
```

Shows number of pages loaded.

---

```python
loader_got = PyPDFLoader("/content/got_book.pdf")
```

Loader for Game of Thrones book.

---

```python
documnet_got = loader_got.load()
```

Loads GOT documents.

---

```python
print(len(documnet_got))
```

Displays page count.

---

# 3. Text Chunking

Large documents cannot be embedded directly.
They must be split into smaller chunks.

---

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### Recursive Splitter

Splits text intelligently using:

* paragraphs
* sentences
* words

---

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

### Parameters

| Parameter     | Meaning                       |
| ------------- | ----------------------------- |
| chunk_size    | characters per chunk          |
| chunk_overlap | shared context between chunks |

Overlap prevents context loss.

---

```python
text_harrypotter = text_splitter.split_documents(documnet_harrypotter)
text_got = text_splitter.split_documents(documnet_got)
```

Creates chunked documents.

---

```python
print(len(text_harrypotter))
print(len(text_got))
```

Shows total chunks.

---

# 4. Embedding Models (Text → Vectors)

```python
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    HuggingFaceBgeEmbeddings
)
```

Embeddings convert text into numerical vectors.

---

```python
HF_TOKEN_REMOVEDembeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Lightweight embedding model.

---

```python
HF_TOKEN_REMOVEDbge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en"
)
```

High‑quality semantic embeddings.

---

```python
from google.colab import userdata
OPENAI_API_KEY=userdata.get('OPENAI_API_KEY')
```

Fetch API key securely.

---

```python
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
```

Sets environment variable.

---

```python
openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```

Optional embedding provider.

---

# 5. Create Chroma Vector Database

```python
from langchain.vectorstores import Chroma
import chromadb
```

Chroma stores vectors for similarity search.

---

```python
import os
os.getcwd()
```

Checks working directory.

---

```python
CURRENT_DIR = os.path.dirname(os.path.abspath("."))
```

Gets project directory.

---

```python
DB_DIR = os.path.join(CURRENT_DIR, "/content/db")
```

Defines database location.

---

```python
client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)
```

### Meaning

| Setting              | Purpose             |
| -------------------- | ------------------- |
| is_persistent        | Save DB permanently |
| persist_directory    | Storage path        |
| anonymized_telemetry | Disable tracking    |

---

## Harry Potter Vector Store

```python
harrypotter_vectorstore = Chroma.from_documents(
    text_harrypotter,
    HF_TOKEN_REMOVEDbge_embeddings,
    client_settings=client_settings,
    collection_name="harrypotter",
    collection_metadata={"hnsw":"cosine"},
    persist_directory="/store/harrypotter"
)
```

Creates vector index.

Uses:

* **HNSW indexing**
* **Cosine similarity**

---

## GOT Vector Store

```python
got_vectorstore = Chroma.from_documents(
    text_got,
    HF_TOKEN_REMOVEDbge_embeddings,
    client_settings=client_settings,
    collection_name="got",
    collection_metadata={"hnsw":"cosine"},
    persist_directory="/store/got"
)
```

Separate collection for GOT.

---

# 6. Create Retrievers

```python
retriever_harrypotter = harrypotter_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "include_metadata": True}
)
```

### MMR (Maximal Marginal Relevance)

Balances:

* relevance
* diversity

---

```python
retriever_got = got_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "include_metadata": True}
)
```

Retriever for GOT.

---

# 7. Merge Retrievers — LOTR (Lord of the Retrievers)

```python
from langchain.retrievers.merger_retriever import MergerRetriever
```

---

```python
lotr = MergerRetriever(
    retrievers=[retriever_harrypotter, retriever_got]
)
```

### What LOTR Does

Combines multiple knowledge sources into one retriever.

```
Query → HP Retriever
      → GOT Retriever
      → Merge Results
```

---

```python
for chunks in lotr.get_relevant_documents("Who was the jon snow?"):
    print(chunks.page_content)
```

Retrieves from both books.

---

```python
for chunks in lotr.get_relevant_documents("Who is a harry potter?"):
    print(chunks.page_content)
```

Multi‑source retrieval.

---

# Problem: Lost in the Middle

Large contexts cause LLM attention degradation.

Solution → Context Compression.

---

# 8. Contextual Compression Pipeline

```python
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import LongContextReorder
```

---

## Redundant Filter

```python
filter = EmbeddingsRedundantFilter(
    embeddings=HF_TOKEN_REMOVEDbge_embeddings
)
```

Removes duplicate semantic chunks.

---

## Reordering

```python
reordering = LongContextReorder()
```

Moves important chunks to edges.

Helps avoid **lost‑in‑the‑middle problem**.

---

## Compression Pipeline

```python
pipeline = DocumentCompressorPipeline(
    transformers=[filter, reordering]
)
```

Sequential document processing.

---

## Contextual Compression Retriever

```python
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=lotr,
    search_kwargs={"k": 3, "include_metadata": True}
)
```

Workflow:

```
Query
  ↓
LOTR Retrieval
  ↓
Redundant Removal
  ↓
Reordering
  ↓
Compressed Context
```

---

# 9. Load Local LLM (Llama.cpp)

```python
!pip install llama-cpp-python
```

Installs local inference backend.

---

```python
from langchain.llms import LlamaCpp
```

---

```python
llms = LlamaCpp(
    streaming=True,
    model_path="/content/drive/MyDrive/zephyr-7b-beta.Q4_K_M.gguf",
    max_tokens=1500,
    temperature=0.75,
    top_p=1,
    gpu_layers=0,
    stream=True,
    verbose=True,
    n_threads=int(os.cpu_count()/2),
    n_ctx=4096
)
```

### Key Parameters

| Parameter   | Meaning          |
| ----------- | ---------------- |
| model_path  | Local GGUF model |
| temperature | Creativity       |
| n_ctx       | Context window   |
| streaming   | Token streaming  |
| n_threads   | CPU usage        |

---

# 10. RetrievalQA Chain

```python
from langchain.chains import RetrievalQA
```

---

```python
qa = RetrievalQA.from_chain_type(
    llm=llms,
    chain_type="stuff",
    retriever=compression_retriever_reordered,
    return_source_documents=True
)
```

### chain_type="stuff"

All retrieved docs are stuffed into prompt.

---

# 11. Ask Questions

```python
query ="who is jon snow?"
results = qa(query)
print(results['result'])
print(results["source_documents"])
```

Produces answer + sources.

---

## Example Output

Jon Snow is a character in George R.R. Martin's *A Song of Ice and Fire* series.

---

```python
results = qa("who is a harry potter?")
print(results['result'])
print(results["source_documents"])

for source in results["source_documents"]:
    print(source.metadata)
```

Displays metadata sources.

---

## Complex Question

```python
results = qa.invoke(
"How does Jon Snow's relationship with the Stark family influence his identity and decisions throughout A Game of Thrones?"
)
```

---

```python
print(results['result'])
print(results["source_documents"])

for source in results["source_documents"]:
    print(source.metadata)
```

Shows reasoning with citations.

---

# Final Pipeline Summary

```
PDFs
 ↓
Loader
 ↓
Chunking
 ↓
Embeddings
 ↓
Chroma Vector DB
 ↓
Retrievers (MMR)
 ↓
LOTR Merge
 ↓
Compression Pipeline
 ↓
Local LLM
 ↓
Answer + Sources
```

---

# Key Concepts Learned

* Multi‑document RAG
* Vector databases
* HNSW indexing
* MMR retrieval
* Retriever merging (LOTR)
* Contextual compression
* Lost‑in‑the‑middle mitigation
* Local LLM inference

---

# Advantages of This Architecture

* Works with multiple knowledge sources
* Reduces hallucinations
* Faster retrieval
* Cleaner context for LLM
* Fully local inference possible


**End of README**
