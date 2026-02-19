# Basic RAG Flow vs Self‑Query Retrieval (LangChain)

---

## Overview

This document explains **Basic RAG Flow** and **Self‑Query Retrieval** in LangChain using detailed step‑by‑step explanations, architecture understanding, and full runnable code.

This README is designed to be:

* GitHub‑ready
* Beginner → Advanced friendly
* Clean UI structure
* Fully reproducible in Google Colab

---

# Table of Contents

1. Basic RAG Flow
2. When to Use Basic RAG
3. Problems in Basic RAG
4. Complete Basic RAG Implementation
5. RAG Chain Construction (LCEL)
6. Query Examples
7. Self‑Query Retrieval
8. Why Self‑Query Retriever Exists
9. Architecture of Self‑Query Retrieval
10. Metadata Filtering Concept
11. Full Self‑Query Implementation
12. Structured Query Generation
13. Retriever Execution
14. Final RAG Chain with Self‑Query
15. Comparison: Basic RAG vs Self‑Query
16. Key Takeaways

---

# 1. Basic RAG Flow

## Concept

**Retrieval Augmented Generation (RAG)** combines:

* Vector Search (Retrieval)
* Large Language Model (Generation)

Flow:

User Query → Retrieve Similar Chunks → Send Context to LLM → Generate Answer

---

## When to Use It

This is the most basic flow but would be very effective in documents like PDFs where:

* Data has linear structure
* Sections are independent
* Minimal cross‑dependencies between chunks

Examples:

* Manuals
* Research papers
* Policies
* Documentation

---

## Issue in Basic RAG

Similarity Search will filter out only **top‑k similar chunks** which are similar to the user query but:

* It might not be relevant chunk.
* Retrieval depends only on semantic similarity.
* No understanding of logical dependency between chunks.
* Important information may exist in non‑retrieved chunks.

### Result

➡️ **Information Loss**

---

# 2. Install Dependencies

```python
!pip -q install langchain openai tiktoken PyPDF2 faiss-cpu
!pip install langchain_openai
!pip install -U langchain-community
!pip install langchain_chroma
```

---

## Optional: Gemini Setup

```python
%pip install --upgrade --quiet google-generativeai langchain-google-genai

import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

result = llm.invoke("Write a ballad about LangChain")
print(result.content)
```

---

# 3. OpenAI Key Setup

```python
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```

---

# 4. Create Embeddings + Vector Store

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```

---

## Documents

```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="A hacker discovers reality is a simulation and leads a rebellion against the machines controlling it.",
        metadata={"year": 1999, "director": "Lana Wachowski, Lilly Wachowski", "rating": 8.7, "genre": "science fiction"},
    ),
    Document(
        page_content="A young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery.",
        metadata={"year": 1994, "rating": 8.5, "genre": "animated"},
    ),
    Document(
        page_content="Batman faces off against the Joker, a criminal mastermind who plunges Gotham into chaos.",
        metadata={"year": 2008, "director": "Christopher Nolan", "rating": 9.0, "genre": "action"},
    ),
    Document(
        page_content="A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        metadata={"year": 2014, "director": "Christopher Nolan", "rating": 8.6, "genre": "science fiction"},
    )
]
```

---

## Build Vector Store

```python
vectorstore = Chroma.from_documents(docs, embedding)
```

---

# 5. Similarity Search

```python
question1 = "Which 1994 animated movie has a rating of 8.5?"
question2 = "Which movie features Batman facing off against the Joker and who directed it?"
question3 = "What genre is the movie 'The Matrix' and who directed it?"

vectorstore.similarity_search(question1)
vectorstore.similarity_search(question2)
```

---

# 6. Create Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

### Meaning

| Parameter  | Purpose                  |
| ---------- | ------------------------ |
| similarity | cosine similarity search |
| k=3        | retrieve top 3 chunks    |

---

# 7. Build RAG Chain (LCEL)

```python
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

llm = ChatOpenAI(temperature=0.7)
```

---

## Helper Function

```python
import textwrap

def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text
```

---

## Prompt Template

```python
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
```

---

## Chain Construction

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Step‑by‑Step Flow

1. Question enters chain
2. Retriever fetches documents
3. Context injected into prompt
4. LLM generates answer
5. Output parsed as string

---

## Execute Queries

```python
text_reply = chain.invoke(question1)
print(wrap_text(text_reply))

text_reply = chain.invoke("Tell me about the movie which have rating more than 7.")

text_reply = chain.invoke(question3)

print(wrap_text(text_reply))
```

---

# 8. Self Query Retrieval

## Concept

A **Self‑Query Retriever** allows an LLM to:

1. Understand natural language query
2. Convert it into structured query
3. Apply metadata filters automatically
4. Retrieve precise documents

---

## Why Needed?

Basic RAG only uses semantic similarity.

Self‑Query adds:

* Logical filtering
* Metadata reasoning
* Structured querying

Example:

"animated movie after 1990 with rating > 8"

LLM converts into:

```
Genre = animated
Year > 1990
Rating > 8
```

---

# 9. Install Dependencies

```python
!pip install langchain
%pip install --upgrade --quiet langchain-chroma
!pip install langchain_openai
!pip install langchain_chroma
```

---

# 10. Create Documents Again

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
```

(Same docs definition as above)

---

## Build Vector Store

```python
vectorstore = Chroma.from_documents(docs, embedding())
```

---

# 11. Define Metadata Schema

```python
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction','comedy','drama','thriller','romance','action','animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]
```

---

## Document Description

```python
document_content_description = "Brief summary of a movie"
```

---

# 12. Query Constructor

```python
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
```

---

## Install Parser Dependency

```python
!pip install lark
```

---

## Output Parser

```python
output_parser = StructuredQueryOutputParser.from_components()
```

---

## Build Query Constructor Chain

```python
query_constructor = prompt | llm | output_parser
```

---

## Example Structured Query

```python
query_constructor.invoke(
    {
        "query": "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
    }
)
```

Output (conceptually):

```
StructuredQuery(
 query='taxi driver',
 filter=genre='science fiction' AND year between 1990–2000 AND director='Luc Besson'
)
```

---

# 13. SelfQueryRetriever Creation

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)
```

---

## Execute Retrieval

```python
retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)
```

---

# 14. RAG Chain using Self‑Query

```python
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Run Queries

```python
text_reply = chain.invoke("Tell me about the movie which have rating more than 7.")
print(wrap_text(text_reply))
```

---

# 15. Basic RAG vs Self‑Query Retriever

| Feature             | Basic RAG           | Self‑Query Retriever          |
| ------------------- | ------------------- | ----------------------------- |
| Retrieval Method    | Semantic similarity | Semantic + metadata filtering |
| Query Understanding | No                  | Yes (LLM powered)             |
| Structured Filters  | ❌                   | ✅                             |
| Accuracy            | Medium              | High                          |
| Information Loss    | Possible            | Reduced                       |
| Complex Queries     | Weak                | Strong                        |

---

# 16. Internal Architecture Comparison

## Basic RAG

Query → Embedding → Vector Search → Top‑k → LLM

## Self‑Query

Query → LLM → Structured Query → Metadata Filter + Vector Search → LLM

---

# 17. Key Takeaways

* Basic RAG works well for simple documents.
* Similarity search alone causes information loss.
* Self‑Query Retriever enables intelligent filtering.
* Metadata design becomes extremely important.
* Ideal for enterprise document search systems.

---

# Final Summary

Basic RAG = **Semantic Matching**

Self‑Query Retrieval = **Semantic + Logical Retrieval**

This evolution moves RAG from keyword‑like retrieval toward intelligent database querying powered by LLM reasoning.

---

**End of Document**
