# RAG Fusion with LangChain — Complete Step‑by‑Step Guide

---

## Overview

This repository demonstrates **RAG Fusion (Retrieval‑Augmented Generation Fusion)** using **LangChain**, **Gemini LLM**, **Chroma Vector DB**, and **BGE embeddings**.

The notebook builds a full pipeline that:

1. Loads documents
2. Splits text into chunks
3. Creates embeddings
4. Stores vectors in Chroma
5. Retrieves relevant documents
6. Generates answers using an LLM
7. Improves retrieval using **RAG Fusion + Reciprocal Rank Fusion (RRF)**

---

## What is RAG Fusion?

**RAG Fusion** improves traditional RAG by:

* Generating **multiple search queries** from a single user question
* Retrieving documents for each query
* Combining results using **Reciprocal Rank Fusion (RRF)**

This increases recall and reduces retrieval bias.

---

## Environment Setup

### Install Dependencies

```python
# Open In Colab
# RAG Fusion

!pip -q install langchain huggingftiktace_hub oken pypdf
!pip -q install google-generativeai chromadb
!pip -q install sentence_transformers

!pip install -U langchain-community
```

### Explanation

| Package               | Purpose                     |
| --------------------- | --------------------------- |
| langchain             | LLM orchestration framework |
| chromadb              | Vector database             |
| google-generativeai   | Gemini API                  |
| sentence_transformers | Embeddings                  |
| langchain-community   | Additional integrations     |

---

## Utility Function

### Text Wrapper

```python
import textwrap

def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text
```

### Purpose

* Formats long LLM outputs
* Improves readability in notebooks

---

## API Key Setup

```python
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

### Explanation

* Fetches stored Colab secret
* Sets environment variable for Gemini access

---

## Install Gemini Integration

```python
%pip install --upgrade --quiet langchain-google-genai
```

---

## Initialize LLM

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

result = llm.invoke("Write a ballad about LangChain")
print(result.content)
```

### What Happens Here

* Loads Gemini 1.5 Pro
* Sends prompt
* Receives generated text

---

## Imports

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import langchain
```

### Role

| Component    | Purpose         |
| ------------ | --------------- |
| TextSplitter | Chunk documents |
| Chroma       | Vector storage  |
| langchain    | Debug utilities |

---

## Load Documents

```python
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
```

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Dataset Path

```python
data_path="/content/drive/MyDrive/English"
```

### Install Loader Dependency

```python
!pip install unstructured
```

### Load Files

```python
%%time
loader = DirectoryLoader(data_path, glob="*.txt", show_progress=True)
docs = loader.load()
```

### Explanation

* Reads all `.txt` files
* Converts them into LangChain Documents

---

## Inspect Documents

```python
len(docs)

docs = docs[:50]
len(docs)

docs[0]
```

```python
print(docs[2].page_content)
print(docs[1].page_content)
```

### Purpose

* Validate data loading
* Reduce dataset size for experimentation

---

## Merge Raw Text

```python
raw_text = ''
for i, doc in enumerate(docs):
    text = doc.page_content
    if text:
        raw_text += text

print(raw_text)
```

### Why Merge?

Creates one continuous corpus for chunking.

---

## Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)
```

### Parameters

| Parameter       | Meaning              |
| --------------- | -------------------- |
| chunk_size      | characters per chunk |
| chunk_overlap   | shared context       |
| length_function | measurement function |

### Split Text

```python
texts = text_splitter.split_text(raw_text)

len(texts)
print(texts[4])
```

---

## BGE Embeddings

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
```

```python
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}
```

```python
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
)
```

### Why Normalize?

Normalization enables cosine similarity comparison.

---

## Create Vector Database

```python
%%time

db = Chroma.from_texts(
    texts,
    embedding_function,
    persist_directory="./chroma_db"
)
```

### What Happens

1. Convert chunks → embeddings
2. Store vectors in Chroma
3. Persist locally

---

## Similarity Search

```python
query = "Tell me about Universal Studios Singapore?"

db.similarity_search(query, k=5)
```

### Result

Returns top‑K similar chunks.

---

## Setup Retriever

```python
retriever = db.as_retriever()

retriever.get_relevant_documents(query)
```

### Retriever Role

Abstraction layer over vector search.

---

## Build Basic RAG Chain

```python
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
```

### Prompt Template

```python
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
```

### Chain Construction

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Execute

```python
text_reply = chain.invoke("Tell me about Universal Studio Singapore")
print(wrap_text(text_reply))
```

### Pipeline Flow

```
Question
   ↓
Retriever
   ↓
Prompt
   ↓
LLM
   ↓
Parsed Output
```

---

# RAG Fusion Section

---

## Query Generation Prompt

```python
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
```

```python
prompt = ChatPromptTemplate(
    input_variables=['original_query'],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template='You are a helpful assistant that generates multiple search queries based on a single input query.'
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['original_query'],
                template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'
            )
        )
    ]
)
```

### Goal

Generate multiple semantic variations of a query.

---

## Original Query

```python
original_query = "universal studios Singapore"
```

---

## Query Generation Chain

```python
generate_queries = (
    prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```

### Output Example

```
1. Universal Studios Singapore attractions
2. Theme parks in Singapore
3. USS rides and tickets
4. Singapore amusement parks
```

---

## Reciprocal Rank Fusion (RRF)

```python
from langchain.load import dumps, loads
```

```python
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results
```

### Mathematical Idea

RRF score:

```
Score(d) = Σ 1 / (k + rank)
```

Where:

* `rank` = position in each retrieval list
* `k` = smoothing constant

---

## RAG Fusion Retrieval Chain

```python
ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
```

### Flow

```
User Query
     ↓
Generate Multiple Queries
     ↓
Retrieve per Query
     ↓
Fuse Rankings (RRF)
```

---

## Debug Mode

```python
langchain.debug = True
```

Shows internal execution traces.

---

## Input Schema

```python
ragfusion_chain.input_schema.schema()
```

Displays expected inputs.

---

## Execute Fusion Retrieval

```python
ragfusion_chain.invoke({"question": original_query})
```

Returns reranked documents.

---

## Final RAG Fusion QA Chain

```python
from langchain.schema.runnable import RunnablePassthrough
```

```python
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
```

```python
full_rag_fusion_chain = (
    {
        "context": ragfusion_chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Inspect Schema

```python
full_rag_fusion_chain.input_schema.schema()
```

---

## Run Final Query

```python
full_rag_fusion_chain.invoke({
    "question": "Tell me about Singapore’s nightlife scene?"
})
```

### Example Output

```
Singapore’s nightlife scene is incredibly diverse, offering a blend of high-energy clubs and more relaxed options for a night out...
```

---

# Architecture Summary

```
User Question
      ↓
Query Expansion (LLM)
      ↓
Multiple Retrievals
      ↓
Reciprocal Rank Fusion
      ↓
Context Assembly
      ↓
LLM Answer Generation
```

---

# Why RAG Fusion Works Better

| Traditional RAG   | RAG Fusion       |
| ----------------- | ---------------- |
| Single query      | Multiple queries |
| Lower recall      | Higher recall    |
| Sensitive wording | Robust retrieval |
| Single ranking    | Rank fusion      |

---

# Advantages

* Improves retrieval diversity
* Reduces missing context
* Handles ambiguous questions
* Better semantic coverage

---

# Limitations

* More LLM calls
* Higher latency
* Increased compute cost

---

# When to Use RAG Fusion

Use when:

* Knowledge bases are large
* Queries are ambiguous
* High recall is required
* Enterprise search systems

---

# Key Concepts Recap

* Recursive Chunking
* Dense Embeddings (BGE)
* Vector Databases (Chroma)
* Retrievers
* Query Expansion
* Reciprocal Rank Fusion
* Runnable Pipelines

---

# Conclusion

This notebook demonstrates a **production‑grade RAG Fusion pipeline** using LangChain.

You learned:

* Standard RAG
* Multi‑query retrieval
* Rank fusion
* End‑to‑end LLM QA system

RAG Fusion significantly improves answer quality by combining multiple retrieval perspectives before generation.

