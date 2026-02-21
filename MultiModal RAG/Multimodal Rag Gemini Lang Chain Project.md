# Multimodal RAG using Gemini + LangChain

---

## 🚀 Project Overview

This project demonstrates how to build a **Multimodal Retrieval-Augmented Generation (Multimodal RAG)** system using:

* **Google Gemini (Text + Vision Models)**
* **LangChain Framework**
* **FAISS Vector Database**
* **Image + Text Understanding**

The system accepts an **image input**, extracts semantic meaning using a **Vision LLM**, retrieves relevant information from a **knowledge base**, and generates a **grounded final response**.

---

## 🧠 What is Multimodal RAG?

Multimodal RAG extends traditional RAG by allowing multiple data modalities such as:

| Modality             | Example                        |
| -------------------- | ------------------------------ |
| Text                 | Documents, PDFs                |
| Images               | Product photos                 |
| Vision Understanding | Object detection & recognition |

### Pipeline Concept

```
Image → Vision Model → Text Query → Retriever → Context → LLM → Final Answer
```

---

## 🏗️ System Architecture

```
                IMAGE INPUT
                      ↓
              Gemini Vision Model
                      ↓
               Text Description
                      ↓
                 Retriever
                (FAISS DB)
                      ↓
              Relevant Context
                      ↓
               Gemini Text Model
                      ↓
                 Final Answer
```

---

## 📦 Installation

Install all dependencies:

```bash
pip install --upgrade \
  langchain \
  langchain-google-genai \
  "langchain[docarray]" \
  faiss-cpu \
  pypdf
```

---

## 🔑 Environment Setup

Store your Google API key securely.

```python
from google.colab import userdata
import os

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

---

## 🤖 Model Loader

Loads either text or vision Gemini models.

```python
def load_model(model_name):
    if model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    return llm
```

| Model             | Purpose             |
| ----------------- | ------------------- |
| gemini-pro        | Text reasoning      |
| gemini-pro-vision | Image understanding |

---

## 🖼️ Image Processing

Download and display images dynamically.

```python
def get_image(url, filename, extension):
    content = requests.get(url).content
    with open(f'/content/{filename}.{extension}', 'wb') as f:
        f.write(content)
    image = Image.open(f"/content/{filename}.{extension}")
    image.show()
    return image
```

---

## 👁️ Vision Model Usage

Send text + image together to Gemini Vision.

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": image}
    ]
)
```

The model produces a semantic description of the image.

---

## 📄 Loading Knowledge Base

Text knowledge is loaded using LangChain loaders.

```python
loader = TextLoader("/content/nike_shoes.txt")
text = loader.load()[0].page_content
```

---

## ✂️ Text Chunking

Splits large text into smaller semantic chunks.

```python
def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=10
    )
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs
```

---

## 🔢 Embeddings Generation

Convert text into vector representations.

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
```

---

## 🧮 Vector Database (FAISS)

Store embeddings for similarity search.

```python
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

Retriever performs semantic search:

```
Query → Similar Documents
```

---

## 🧩 Prompt Template

Defines how retrieved context is injected.

```python
template = """
```

{context}

```

{query}

Provide brief information and store location.
"""
```

---

## 🔗 Text RAG Chain

```python
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm_text
    | StrOutputParser()
)
```

### Flow

```
User Query
   ↓
Retriever
   ↓
Context Injection
   ↓
Text LLM
   ↓
Final Answer
```

---

## 🌐 Multimodal RAG Chain

Vision output becomes input to RAG.

```python
full_chain = (
    RunnablePassthrough()
    | llm_vision
    | StrOutputParser()
    | rag_chain
)
```

### Multimodal Flow

```
Image
 ↓
Vision LLM
 ↓
Generated Text Query
 ↓
Retriever
 ↓
Knowledge Context
 ↓
Text LLM
 ↓
Grounded Response
```

---

## ▶️ Running the System

```python
result = full_chain.invoke([message])
```

Output contains:

* Product identification
* Context-aware explanation
* Store/location information

---

## 📊 Key Components Summary

| Component       | Role                  |
| --------------- | --------------------- |
| Gemini Vision   | Understand image      |
| Gemini Text     | Generate answers      |
| FAISS           | Vector search         |
| Embeddings      | Semantic encoding     |
| Retriever       | Fetch relevant chunks |
| Prompt Template | Context formatting    |
| RAG Chain       | Knowledge grounding   |

---

## ✅ Features

* Multimodal input (Image + Text)
* Retrieval-Augmented Generation
* Vision-to-Text conversion
* Semantic search using FAISS
* Context-grounded responses
* Modular LangChain pipeline

---

## 🧪 Example Use Case

Input:

```
Nike sandal image
```

System performs:

1. Detect brand & model from image
2. Retrieve product information
3. Generate grounded explanation

Output:

```
Brand: Nike
Model: Calm Slides
Description + Store Location
```

---

## 📁 Project Structure

```
project/
│
├── notebook.ipynb
├── nike_shoes.txt
├── README.md
└── assets/
```

---

## 🔮 Future Improvements

* Multimodal embeddings (CLIP)
* Hybrid search (BM25 + Vector)
* Reranking models
* Streaming responses
* Production API deployment
* LangGraph agent workflow

---

## 🧑‍💻 Technologies Used

* LangChain
* Google Gemini API
* FAISS
* Python
* PIL
* Matplotlib


---

## ⭐ Acknowledgements

* Google Gemini
* LangChain
* FAISS
