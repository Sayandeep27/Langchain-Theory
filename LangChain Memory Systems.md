# 🧠 LangChain Memory Systems — Complete Guide

## Open In Colab

ConversationEntityMemory is a memory class provided by LangChain, designed to track and store information about entities that arise in a conversation. It allows the AI to "remember" key facts about people, places, or concepts mentioned during a conversation, so it can refer back to them later on, improving the conversational experience.

---

## Key Features

### Entity Tracking

It identifies entities (e.g., names, places, concepts) and stores relevant information about them. For instance, if you mention "Tanmay" in one part of a conversation, it can remember details about "Tanmay" for later reference.

### Context-Aware

It helps the AI maintain context by remembering details about the entities mentioned during the chat, ensuring more natural, fluid conversations over time.

### Customization

You can customize what to store and how to retrieve it during future interactions.

---

## Installation

```python
!pip install langchain
!pip install -U langchain-community
!pip install langchain_google_genai
```

---

## Setup Environment

```python
import warnings
warnings.filterwarnings('ignore')

import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

---

## Load Gemini Model

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",convert_system_message_to_human=True)

print(model.invoke("hi").content)
```

---

# ConversationEntityMemory

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=model)
```

### Adding Inputs

```python
_input= {"input": "i am very hungry."}
memory.load_memory_variables(_input)

_input= {"input": "sunny & mayank are working on a hackathon project"}
memory.load_memory_variables(_input)

_input= {"input": "My name is John, and I'm planning a trip to Paris."}
memory.load_memory_variables(_input)

_input= {"input": "Sunny is a great person who values gratitude."}
memory.load_memory_variables(_input)
```

---

### Saving Context

```python
memory.save_context(
    {"Human": "Sunny and Mayank are working on a hackathon project"},
    {"AI": "That's awesome! What's the hackathon project about?"}
)
```

---

### Query Memory

```python
memory.load_memory_variables({"input": "who is Sunny?"})
```

---

### Add More Context

```python
memory.save_context(
    {"Human": "It's a machine learning project focused on healthcare."},
     {"AI": "Sounds exciting! Are they building a prediction model or something else?"}
)

memory.save_context(
    {"Human": "Yes, they are building prediction model."},
    {"AI": "Wishing Sunny and Mayank all the best for their project!"}
)

memory.load_memory_variables({"input": "who is Sunny?"})
```

---

## Internal Prompt Used for Entity Update

```text
You are an AI assistant helping a human keep track of facts about relevant people, places, and concepts in their life. Update the summary of the provided entity in the "Entity" section based on the last line of your conversation with the human. If you are writing the summary for the first time, return a single sentence.

The update should only include facts that are relayed in the last line of conversation about the provided entity, and should only contain facts about the provided entity.

If there is no new information about the provided entity or the information is not worth noting (not an important or relevant fact to remember long-term), return the existing summary unchanged.

Full conversation history (for context):
{history}

Entity to summarize:
{entity}

Existing summary of {entity}:
{summary}

Last line of conversation:
Human: {input}
Updated summary:
```

---

## Entity Extraction Prompt

```text
You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation.

Return the output as a single comma-separated list, or NONE if there is nothing of note.
```

---

## ConversationChain with Entity Memory

```python
from langchain.chains import ConversationChain
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

conversation = ConversationChain(
    llm=model,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=model)
)
```

### Example Conversation

```python
conversation.predict(input="Deven & Sam are working on a hackathon project")
conversation.memory.entity_store.store

conversation.predict(input="They are trying to add more complex memory structures to Langchain")
conversation.predict(input="They are adding in a key-value store for entities mentioned so far in the conversation.")
conversation.predict(input="What do you know about Deven & Sam?")
```

```python
from pprint import pprint
pprint(conversation.memory.entity_store.store)
```

```python
conversation.predict(input="Sam is the founder of a company called Daimon.")
pprint(conversation.memory.entity_store.store)
conversation.predict(input="What do you know about Sam?")
```

---

# ConversationSummaryMemory

## Installation

```python
!pip install langchain
!pip install langchain_community
!pip install langchain-groq
```

---

## Setup

```python
import os
from google.colab import userdata

GROQ_API_KEY=userdata.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

LANGCHAIN_KEY_REMOVED=userdata.get('LANGCHAIN_KEY_REMOVED')
os.environ["LANGCHAIN_KEY_REMOVED"] = LANGCHAIN_KEY_REMOVED

os.environ["LANGCHAIN_PROJECT"]="memorylogs"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
```

---

## Load Groq Model

```python
from langchain_groq import ChatGroq
model=ChatGroq(model_name="Gemma2-9b-It")

model.invoke("Hi, what's up?")
```

---

## Using ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ChatMessageHistory

memory = ConversationSummaryMemory(llm=model, return_messages=True)
```

### Save Context

```python
memory.save_context(
    {"input": "Sunny and Mayank are working on a hackathon project."},
    {"output": "That's awesome! What's the hackathon project about?"}
)
```

```python
memory.load_memory_variables({})
summary=memory.load_memory_variables({})
print(summary["history"][0].content)
```

### Add More Context

```python
memory.save_context(
    {"input": "It's a machine learning project focused on healthcare."},
    {"output": "Sounds exciting! Are they building a prediction model or something else"}
)
memory.save_context(
    {"input": "Yes, they’re working on a model to predict patient outcomes."},
    {"output": "Impressive! How far along are they with the project?"}
)
```

```python
memory.load_memory_variables({})
summary=memory.load_memory_variables({})
print(summary["history"][0].content)
```

---

## Progressive Summarization Prompt

```text
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
```

---

## Access Chat Memory

```python
memory.chat_memory
memory.chat_memory.messages
```

```python
messages = memory.chat_memory.messages
previous_summary=""
memory.predict_new_summary(messages, previous_summary)
```

---

## Create Memory from Messages

```python
history = ChatMessageHistory()

history.add_user_message("Hi")
history.add_ai_message("Hello, how can I assist you today?")

ConversationSummaryMemory.from_messages(
    llm=model,
    chat_memory=history,
    memory_key="summary",
    human_prefix="User",
    ai_prefix="AI"
)
```

```python
memory = ConversationSummaryMemory.from_messages(
    llm=model,
    chat_memory=history,
    memory_key="summary",
    human_prefix="User",
    ai_prefix="AI"
)

memory.buffer
```

---

## ConversationChain with Summary Memory

```python
from langchain.chains import ConversationChain

conversation_with_summary = ConversationChain(
    llm=model,
    memory=ConversationSummaryMemory(llm=model),
    verbose=True
)
```

### Example Usage

```python
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Sunny and Mayank are working on a mlops production ready project.")
conversation_with_summary.predict(input="It's project focused on healthcare.")
conversation_with_summary.predict(input="so can you describe mlops pipeline to me with in six point.")
conversation_with_summary.predict(input="How many total points are there?")
conversation_with_summary.predict(input="can you give me 5th point with explaination")
```

---

## Conversation Summary Buffer Memory

While summary is good, recent conversation has high correlation to upcoming query.

A summary of old conversation with a buffer memory of last few conversation is a good combination.

You can set the token limit which defines how much historical conversation to be kept along with the summary.

---

```python
from langchain.memory import ConversationSummaryBufferMemory

memory2 = ConversationSummaryBufferMemory(llm=model,return_messages=True)
```

### Save Context

```python
memory2.save_context(
    {"input": "It's a machine learning project focused on healthcare."},
    {"output": "Sounds exciting! Are they building a prediction model or something else"}
)
memory2.save_context(
    {"input": "Yes, they’re working on a model to predict patient outcomes."},
    {"output": "Impressive! How far along are they with the project?"}
)
```

```python
memory2.load_memory_variables({})
```

---

### Token Limited Buffer

```python
memory3 = ConversationSummaryBufferMemory(llm=model,return_messages=True,max_token_limit=50)
```

```python
memory3.save_context(
    {"input": "Sunny and Mayank are working on a hackathon project."},
    {"output": "That's awesome! What's the hackathon project about?"}
)
memory3.save_context(
    {"input": "It's a machine learning project focused on healthcare."},
    {"output": "Sounds exciting! Are they building a prediction model or something else?"}
)
memory3.save_context(
    {"input": "Yes, they’re working on a model to predict patient outcomes."},
    {"output": "Impressive! Wishing Sunny and Mayank all the best for their project."}
)
```

```python
memory3.load_memory_variables({})["history"]
memory3.load_memory_variables({})["history"][0].content
```

---

### Smaller Token Limit

```python
memory4 = ConversationSummaryBufferMemory(llm=model,return_messages=True,max_token_limit=20)
```

```python
memory4.save_context(
    {"input": "Sunny and Mayank are working on a hackathon project."},
    {"output": "That's awesome! What's the hackathon project about?"}
)
memory4.save_context(
    {"input": "It's a machine learning project focused on healthcare."},
    {"output": "Sounds exciting! Are they building a prediction model or something else?"}
)
memory4.save_context(
    {"input": "Yes, they’re working on a model to predict patient outcomes."},
    {"output": "Impressive! Wishing Sunny and Mayank all the best for their project."}
)
```

```python
memory4.load_memory_variables({})
```

---

## ConversationChain using Summary Buffer Memory

```python
from langchain.chains import ConversationChain

conversation_with_summary = ConversationChain(
    llm=model,
    memory=ConversationSummaryBufferMemory(llm=model, max_token_limit=40),
    verbose=True,
)
```

```python
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation on machine learning!")
conversation_with_summary.predict(input="give me some points for writing about the document")
conversation_with_summary.predict(input="can you list out the resources from the previous message")
```

---

# Additional Memory Types

## Conversation Knowledge Graph Memory

Uses a knowledge graph to store information and relationships between entities.

## VectorStore-Backed Memory

Uses vector embeddings to store and retrieve information based on semantic similarity.

## ConversationTokenBufferMemory

Instead of remembering "k" conversations like ConversationBufferWindowMemory, this remembers discussions based on a maximum token limit.

---

# Summary Table

| Memory Type                     | Purpose                 | Best Use Case       |
| ------------------------------- | ----------------------- | ------------------- |
| ConversationEntityMemory        | Tracks entities         | Personal assistants |
| ConversationSummaryMemory       | Summarizes history      | Long conversations  |
| ConversationSummaryBufferMemory | Summary + recent buffer | Balanced memory     |
| KnowledgeGraphMemory            | Entity relationships    | Complex reasoning   |
| VectorStore Memory              | Semantic recall         | RAG systems         |
| TokenBufferMemory               | Token-based recall      | LLM token control   |

---

# End of Guide
