# Hybrid Retrieval-Augmented Generation (RAG) — End‑to‑End Implementation

---

## Overview

This repository demonstrates a **complete Retrieval‑Augmented Generation (RAG) pipeline**, progressing step‑by‑step from:

* Sparse Retrieval (TF‑IDF)
* Dense Embeddings
* Cosine Similarity Ranking
* PDF Loading & Chunking
* Vector Database (Chroma)
* Keyword Retrieval (BM25)
* Hybrid Search (Dense + Sparse)
* Quantized Open‑Source LLM
* RetrievalQA using LangChain

The notebook builds an **industry‑style Hybrid RAG system** using open‑source tools.

---

## Open In Colab

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

---

## Sparse Retrieval Example (TF‑IDF)

```python
# Sample documents
documents = [
    "This is a list which containig sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings."
]
```

```python
query="keyword-based search"
```

```python
import re
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text
```

```python
preprocess_documents=[preprocess_text(doc) for doc in documents]
```

```python
preprocess_documents
```

```python
print("Preprocessed Documents:")
for doc in preprocess_documents:
    print(doc)
```

```python
print("Preprocessed Query:")
print(query)
```

```python
preprocessed_query = preprocess_text(query)
```

```python
preprocessed_query
```

```python
vector=TfidfVectorizer()
```

```python
X=vector.fit_transform(preprocess_documents)
```

```python
X.toarray()
```

```python
X.toarray()[0]
```

```python
query_embedding=vector.transform([preprocessed_query])
```

```python
query_embedding.toarray()
```

```python
similarities = cosine_similarity(X, query_embedding)
```

```python
similarities
```

```python
np.argsort(similarities,axis=0)
```

```python
ranked_documents = [documents[i] for i in ranked_indices]
```

```python
#Ranking
ranked_indices=np.argsort(similarities,axis=0)[::-1].flatten()
```

```python
ranked_indices
```

```python
# Output the ranked documents
for i, doc in enumerate(ranked_documents):
    print(f"Rank {i+1}: {doc}")
```

```python
query
```

---

## Dense Embedding Example

```python
documents = [
    "This is a list which containig sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings."
]
```

```python
#https://huggingface.co/sentence-transformers
```

```python
document_embeddings = np.array([
    [0.634, 0.234, 0.867, 0.042, 0.249],
    [0.123, 0.456, 0.789, 0.321, 0.654],
    [0.987, 0.654, 0.321, 0.123, 0.456]
])
```

```python
# Sample search query (represented as a dense vector)
query_embedding = np.array([[0.789, 0.321, 0.654, 0.987, 0.123]])
```

```python
# Calculate cosine similarity between query and documents
similarities = cosine_similarity(document_embeddings, query_embedding)
```

```python
similarities
```

```python
ranked_indices = np.argsort(similarities, axis=0)[::-1].flatten()
```

```python
ranked_indices
```

```python
# Output the ranked documents
for i, idx in enumerate(ranked_indices):
    print(f"Rank {i+1}: Document {idx+1}")
```

---

## Load PDF for RAG

```python
doc_path="/content/Retrieval-Augmented-Generation-for-NLP.pdf"
```

```python
!pip install pypdf
```

```python
!pip install langchain_community
```

```python
from langchain_community.document_loaders import PyPDFLoader
```

```python
loader=PyPDFLoader(doc_path)
```

```python
docs=loader.load()
```

---

## Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=30)
```

```python
chunks = splitter.split_documents(docs)
```

```python
chunks
```

---

## Embeddings

```python
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
```

```python
HF_TOKEN=""  # Replace with your Hugging Face API token
```

```python
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
```

---

## Vector Store (Chroma)

```python
!pip install chromadb
```

```python
from langchain.vectorstores import Chroma
```

```python
vectorstore=Chroma.from_documents(chunks,embeddings)
```

```python
vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
```

```python
vectorstore_retreiver
```

---

## Keyword Retrieval (BM25)

```python
!pip install rank_bm25
```

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
```

```python
keyword_retriever = BM25Retriever.from_documents(chunks)
```

```python
keyword_retriever.k =  3
```

```python
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])
```

```python
Mixing vector search and keyword search for Hybrid search
hybrid_score = (1 — alpha) * sparse_score + alpha * dense_score
```

---

## Load Quantized LLM

```python
model_name = "HuggingFaceH4/zephyr-7b-beta"
```

```python
!pip install bitsandbytes
```

```python
!pip install accelerate
```

```python
import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )
from langchain import HuggingFacePipeline
```

```python
# function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    """
    model_name: Name or path of the model to be loaded.
    return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model
```

```python
# initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    model_name: Name or path of the model for tokenizer initialization.
    return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer
```

```python
tokenizer = initialize_tokenizer(model_name)
```

```python
model = load_quantized_model(model_name)
```

```python
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
```

```python
llm = HuggingFacePipeline(pipeline=pipeline)
```

---

## RetrievalQA Chains

```python
from langchain.chains import RetrievalQA
```

```python
normal_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore_retreiver
)
```

```python
hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ensemble_retriever
)
```

---

## Query Execution

```python
response1 = normal_chain.invoke("What is Abstractive Question Answering?")
```

```python
response1
```

```python
print(response1.get("result"))
```

```python
response2 = hybrid_chain.invoke("What is Abstractive Question Answering?")
```

```python
response2
```

```python
print(response2.get("result"))
```

