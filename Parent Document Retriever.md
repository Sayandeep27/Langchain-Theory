# Parent Document Retriever in LangChain — Complete Guide

---

## Open In Colab

```
(Open your notebook in Google Colab before running the cells)
```

---

## Parent Document Retriever

### Which issue this parent–child retrieval solves

You may want to have **small documents**, so that their embeddings can most accurately reflect their meaning. If too long, then the embeddings can lose meaning.

You also want to have **long enough documents** so that the context of each chunk is retained.

The **ParentDocumentRetriever** strikes that balance by:

* Splitting and storing **small chunks** of data (child documents).
* During retrieval, it first fetches the small chunks.
* Then it looks up the **parent IDs** for those chunks.
* Finally, it returns the **larger parent documents**.

> Note: "Parent document" refers to the document that a small chunk originated from. This can either be:
>
> * The whole raw document, OR
> * A larger chunk created during preprocessing.

---

# Installation

```bash
!pip install langchain
```

```bash
!pip install -U langchain-community
```

```bash
!pip install sentence-transformers
```

```bash
!pip install langchain_chroma
```

---

## (Optional) Gemini Setup

If you want to use Gemini embeddings and LLM:

```bash
%pip install --upgrade --quiet google-generativeai langchain-google-genai
```

---

# Data Ingestion

## Step 1 — Load Documents

```python
from langchain_community.document_loaders import TextLoader
```

```python
loaders = [
    TextLoader("/content/data/paul_graham_essay.txt"),
    TextLoader("/content/data/state_of_the_union.txt"),
]
```

```python
docs = []
```

```python
for loader in loaders:
    docs.extend(loader.load())
```

```python
docs
```

### Explanation

* `TextLoader` loads raw text files.
* Each file becomes a LangChain **Document object**.
* Documents contain:

  * `page_content`
  * `metadata`

---

## Step 2 — Child Text Splitter

This splitter creates **child documents** (small semantic chunks).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
```

### Why small chunks?

Small chunks:

* Produce more accurate embeddings.
* Improve semantic matching.
* Reduce embedding noise.

---

## Step 3 — Storage Layers

```python
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
```

We need **two storage systems**:

| Component    | Purpose                           |
| ------------ | --------------------------------- |
| Vector Store | Stores embeddings of child chunks |
| Doc Store    | Stores parent documents           |

---

# Embedding Model Selection Concepts

### Key Factors

* **Dataset size** → Larger datasets benefit from stronger models like MPNet.
* **Computational resources** → Smaller models like MiniLM are faster.
* **Task complexity** → QA & reasoning tasks prefer MPNet.
* **Embedding dimensionality** → Affects retrieval accuracy and storage.
* **Performance vs efficiency trade‑off** → Accuracy vs speed.

### Experimentation is Key

Try multiple embedding models and evaluate performance.

---

## Important Benchmarks & Resources

* MTEB: Massive Text Embedding Benchmark
* MPNET: Masked and Permuted Pre‑training for Language Understanding
* BGE: BAAI General Embedding

Links:

* [https://huggingface.co/BAAI](https://huggingface.co/BAAI)
* [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
* [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
* [https://huggingface.co/blog/mteb](https://huggingface.co/blog/mteb)

### Model Comparison

| Model             | Speed      | Quality      |
| ----------------- | ---------- | ------------ |
| all-mpnet-base-v2 | Slower     | Best quality |
| all-MiniLM-L6-v2  | ~5x faster | Good quality |

---

## (Optional) HuggingFace Embeddings Example

```python
# specify embedding model (using huggingface sentence transformer)
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=model_kwargs
)
```

---

## Step 4 — Setup Gemini Embeddings

```python
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

### Explanation

* Loads Gemini embedding model.
* Converts text → vectors.

---

## Step 5 — Create Vector Store

```python
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=gemini_embeddings
)
```

### What Chroma Does

* Stores embeddings.
* Enables similarity search.
* Supports fast retrieval.

---

## Step 6 — Parent Document Storage

```python
store = InMemoryStore()
```

Stores parent documents separately from embeddings.

---

## Step 7 — Create ParentDocumentRetriever

```python
from langchain.retrievers import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
```

### Internal Workflow

1. Split documents into child chunks.
2. Embed child chunks.
3. Store embeddings in vector DB.
4. Store parents in docstore.
5. Link children → parents using IDs.

---

## Step 8 — Add Documents

```python
retriever.add_documents(docs, ids=None)
```

### What Happens Internally

* Documents are split.
* Child chunks embedded.
* Parent mapping created.

---

## Step 9 — Inspect Stored Parents

```python
list(store.yield_keys())
```

Returns parent document IDs.

---

## Step 10 — Retrieve Documents

```python
retrieved_docs = retriever.invoke("What did the president say about Ketanji Brown Jackson")
```

```python
print(retrieved_docs[0].page_content)
```

```python
print(len(retrieved_docs[0].page_content))
```

### Key Idea

* Search occurs on **child embeddings**.
* Returned result is **parent document**.

---

# Advanced Version — Explicit Parent + Child Splitters

## Child Splitter

```python
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
```

## Parent Splitter

```python
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
```

### Why Two Splitters?

| Splitter | Role                    |
| -------- | ----------------------- |
| Child    | Precise semantic search |
| Parent   | Rich contextual answer  |

---

## Storage Layer

```python
store1 = InMemoryStore()
```

---

## Vector Store

```python
vectorstore1 = Chroma(
    collection_name="full_documents", embedding_function=gemini_embeddings
)
```

---

## Retriever with Parent Splitter

```python
retriever2 = ParentDocumentRetriever(
    vectorstore=vectorstore1,
    docstore=store1,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

---

## Add Documents Again

```python
retriever2.add_documents(docs)
```

---

## Compare Stored Keys

```python
len(list(store1.yield_keys()))
```

```python
len(list(store.yield_keys()))
```

---

## Retrieval Using Improved Setup

```python
retrieved_docs2 = retriever2.invoke(
    "What did the president say about Ketanji Brown Jackson"
)
```

```python
retrieved_docs2
```

```python
len(retrieved_docs2[0].page_content)
```

### Result

Returned document is larger → better context.

---

# Data Generation (LLM Step)

## Step 1 — Load Gemini LLM

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

---

## Step 2 — Simple Generation Example

```python
result = llm.invoke("Write a ballad about LangChain")
print(result.content)
```

---

# Retrieval QA Chain

## Step 3 — Create RetrievalQA

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever2
)
```

### chain_type="stuff"

* Inserts retrieved documents directly into prompt.
* Simple and effective for moderate context sizes.

---

## Step 4 — Query

```python
query = "What did the president say about Ketanji Brown Jackson"
```

---

## Step 5 — Run QA

```python
qa.run(query)
```

---

# End‑to‑End Flow Summary

```
Raw Documents
      ↓
Parent Splitter
      ↓
Child Splitter
      ↓
Child Embeddings → Vector Store (Chroma)
      ↓
Parent Docs → DocStore
      ↓
User Query
      ↓
Similarity Search (Children)
      ↓
Parent Lookup
      ↓
LLM Answer Generation
```

---

# Why ParentDocumentRetriever is Powerful

| Problem                               | Solution                     |
| ------------------------------------- | ---------------------------- |
| Small chunks lose context             | Return parent docs           |
| Large chunks reduce embedding quality | Search using children        |
| Hallucinations                        | Better grounded context      |
| RAG accuracy                          | Improved retrieval relevance |

---

# When to Use ParentDocumentRetriever

Use when:

* Documents are long (PDFs, reports, essays).
* You need accurate semantic retrieval.
* Context preservation is critical.
* Building production RAG systems.

---

# Key Takeaways

* Embed **small**, return **big**.
* Parent–child linking improves RAG.
* Two‑level chunking balances precision and context.
* Works extremely well with Gemini + Chroma.

---

# Final Insight

ParentDocumentRetriever solves the **core trade‑off in RAG systems**:

> **Embedding accuracy vs Context completeness**

By searching small chunks but returning larger parents, it achieves both.

---

**End of README**
