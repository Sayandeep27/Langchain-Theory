# 🚀 Hybrid Retrieval-Augmented Generation (RAG) using **Weaviate + LangChain + HuggingFace**

---

## 📌 Overview

This project demonstrates a **complete Hybrid RAG pipeline** using:

* **Weaviate Vector Database** (Hybrid Search: Keyword + Semantic)
* **LangChain** orchestration
* **HuggingFace LLM (Zephyr‑7B)**
* **4‑bit Quantization (BitsAndBytes)**
* **PDF Knowledge Ingestion**
* **Reranking with Cohere**

The notebook builds an **end‑to‑end Retrieval‑Augmented Generation system** capable of:

* Storing documents
* Hybrid retrieval
* Context compression
* Question answering

---

# 🧱 Architecture

```
User Query
     ↓
Hybrid Retriever (BM25 + Dense Search)
     ↓
(Optional) Reranker / Compressor
     ↓
Prompt Template
     ↓
LLM (Zephyr‑7B)
     ↓
Final Answer
```

---

# ⚙️ Step 1 — Environment Setup

## Install Dependencies

```python
!pip install weaviate-client
!pip install langchain
!pip install -U langchain-community
```

### Why these packages?

| Package             | Purpose                 |
| ------------------- | ----------------------- |
| weaviate-client     | Connects to Weaviate DB |
| langchain           | RAG orchestration       |
| langchain-community | Community integrations  |

---

# 🌐 Step 2 — Import Weaviate

```python
import weaviate
```

Weaviate is a **vector database** supporting:

* Semantic search
* Keyword search
* Hybrid search
* Built‑in embedding models

---

# 🔐 Step 3 — Configure Weaviate Credentials

```python
WEAVIATE_CLUSTER="https://hybridsearch-ewd5zpr1.weaviate.network"
WEAVIATE_API_KEY="" # Replace with your Weaviate API key
```

### Explanation

* `WEAVIATE_CLUSTER` → Hosted vector database endpoint
* `WEAVIATE_API_KEY` → Authentication key

---

```python
WEAVIATE_URL = WEAVIATE_CLUSTER
WEAVIATE_API_KEY = WEAVIATE_API_KEY
```

Just assigning variables for clarity.

---

# 🤗 Step 4 — HuggingFace Token

```python
HF_TOKEN=""  # Replace with your Hugging Face API token
```

Required because Weaviate uses HuggingFace embeddings internally.

---

# 🧩 Step 5 — Create Weaviate Client

```python
import os

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={
         "X-HuggingFace-Api-Key": HF_TOKEN
    },
)
```

### What happens internally?

* Authenticates with Weaviate
* Enables HuggingFace vectorization module
* Connects LangChain ↔ Weaviate

---

## Check Connection

```python
client.is_ready()
```

Returns `True` if cluster is active.

---

# 🧬 Step 6 — Inspect Existing Schema

```python
client.schema.get()
```

Schema = database structure.

Equivalent to tables in SQL.

---

# 🧹 Step 7 — Delete Existing Schema

```python
client.schema.delete_all()
```

Removes all collections/classes.

⚠️ Useful for clean experimentation.

---

# 🏗 Step 8 — Create Vector Schema

```python
schema = {
    "classes": [
        {
            "class": "RAG",
            "description": "Documents for RAG",
            "vectorizer": "text2vec-huggingface",
            "moduleConfig": {"text2vec-huggingface": {"model": "sentence-transformers/all-MiniLM-L6-v2", "type": "text"}},
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-huggingface": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}
```

### Key Concepts

| Field      | Meaning         |
| ---------- | --------------- |
| class      | Collection name |
| vectorizer | Embedding model |
| properties | Stored fields   |
| content    | Text to embed   |

Model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

A lightweight semantic embedding model.

---

## Create Schema

```python
client.schema.create(schema)
```

---

## Verify Schema

```python
client.schema.get()
```

---

# 🔎 Step 9 — Hybrid Search Retriever

```python
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
```

Hybrid search combines:

* Sparse retrieval (BM25 keywords)
* Dense retrieval (embeddings)

---

```python
retriever = WeaviateHybridSearchRetriever(
    alpha = 0.5,
    client = client,
    index_name = "RAG",
    text_key = "content",
    attributes = [],
    create_schema_if_missing=True,
)
```

### Parameters

| Parameter  | Meaning                   |
| ---------- | ------------------------- |
| alpha      | 0 = keyword, 1 = semantic |
| index_name | Weaviate class            |
| text_key   | stored text field         |

`alpha=0.5` → equal hybrid weighting.

---

# 🤖 Step 10 — Select LLM

```python
model_name = "HuggingFaceH4/zephyr-7b-beta"
```

Zephyr‑7B = instruction‑tuned open LLM.

---

# ⚡ Step 11 — Install Quantization Dependencies

```python
!pip install bitsandbytes
!pip install accelerate
```

Allows GPU-efficient inference.

---

# 🧠 Step 12 — Import Transformers

```python
import torch
from transformers import (
 AutoModelForCausalLM,
 AutoTokenizer,
 BitsAndBytesConfig,
 pipeline,
)
from langchain import HuggingFacePipeline
```

---

# 🪶 Step 13 — Load 4‑bit Quantized Model

```python
def load_quantized_model(model_name: str):
```

### Quantization Config

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
```

### Why Quantization?

| Benefit  | Result            |
| -------- | ----------------- |
| Memory ↓ | Runs on Colab GPU |
| Speed ↑  | Faster inference  |
| Cost ↓   | Smaller footprint |

---

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
```

Loads compressed LLM.

---

# 🔤 Step 14 — Initialize Tokenizer

```python
def initialize_tokenizer(model_name: str):
```

```python
tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
tokenizer.bos_token_id = 1
```

Tokenizer converts text → tokens.

---

# 🧪 Step 15 — Create Generation Pipeline

```python
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    do_sample=True,
    top_k=5,
    max_new_tokens=100,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
```

### Important Settings

| Parameter      | Purpose             |
| -------------- | ------------------- |
| do_sample      | stochastic output   |
| top_k          | controls randomness |
| max_new_tokens | response length     |

---

## Convert to LangChain LLM

```python
llm = HuggingFacePipeline(pipeline=pipeline)
```

---

# 📄 Step 16 — Load PDF

```python
doc_path="/content/Retrieval-Augmented-Generation-for-NLP.pdf"
```

Install loader:

```python
!pip install pypdf
!pip install langchain_community
```

---

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(doc_path)
docs = loader.load()
```

Each PDF page becomes a LangChain Document.

---

```python
docs[6]
```

Shows a sample page.

---

# 🧾 Step 17 — Add Documents to Weaviate

```python
retriever.add_documents(docs)
```

Internally:

1. Text extracted
2. Embeddings created
3. Stored in vector DB

---

# 🔍 Step 18 — Test Retrieval

```python
print(retriever.invoke("what is RAG token?")[0].page_content)
```

Returns best matching document.

---

```python
retriever.invoke(
    "what is RAG token?",
    score=True
)
```

Also returns similarity scores.

---

# 🔗 Step 19 — RetrievalQA Chain

```python
from langchain.chains import RetrievalQA
```

Combines:

* Retriever
* Prompt
* LLM

---

# 🧩 Step 20 — Prompt Engineering

```python
from langchain_core.prompts import ChatPromptTemplate
```

System prompt:

```python
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
```

Defines LLM behavior.

---

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{query}"),
    ]
)
```

---

## Alternative Prompt Template

```python
from langchain.prompts import PromptTemplate
```

Custom detailed instruction prompt created.

---

# 🧠 Step 21 — Combine Documents Chain

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
```

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
```

"Stuff" = concatenate all retrieved docs.

---

# 🔄 Step 22 — Hybrid RetrievalQA

```python
hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)
```

---

## Query Example

```python
result1 = hybrid_chain.invoke("what is natural language processing?")
print(result1)
print(result1['result'])
```

---

```python
query="What is Abstractive Question Answering?"
response = hybrid_chain.invoke({"query":query})
```

---

# ⚙️ Step 23 — LCEL RAG Chain

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
```

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} |
    prompt |
    llm
)
```

### Flow

```
Query
 → Retriever
 → Prompt
 → LLM
```

---

```python
response=rag_chain.invoke("what is RAG token?")
print(response)
```

---

# 🧠 Step 24 — Context Compression (Reranking)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
```

Install:

```python
!pip install cohere
```

---

```python
compressor = CohereRerank(cohere_api_key="")
```

Cohere reranker:

* Reorders retrieved documents
* Keeps most relevant chunks

---

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

Pipeline:

```
Retriever → Reranker → LLM
```

---

```python
compressed_docs = compression_retriever.get_relevant_documents(user_query)
print(compressed_docs)
```

Returns compressed context.

---

# 🔁 Step 25 — QA with Compression

```python
hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever
)
```

---

```python
response = hybrid_chain.invoke("What is Abstractive Question Answering?")
print(response.get("result"))
```

---

# 🧩 Final Pipeline Summary

| Stage         | Component     |
| ------------- | ------------- |
| Storage       | Weaviate      |
| Embeddings    | MiniLM        |
| Retrieval     | Hybrid Search |
| Compression   | Cohere Rerank |
| LLM           | Zephyr‑7B     |
| Orchestration | LangChain     |

---

# ✅ Key Learnings

* Hybrid search improves recall
* Quantized LLMs enable local inference
* Reranking improves answer quality
* Prompt design controls hallucination
* Context compression reduces noise

---

# 🚀 End Result

You built a **production‑style Hybrid RAG System** featuring:

* Vector database
* Hybrid retrieval
* Reranking
* Quantized open LLM
* LCEL pipeline

---

# 📚 References

* [https://s4ds.org/](https://s4ds.org/)
* [https://www.icdmai.org/](https://www.icdmai.org/)


**End of README**
