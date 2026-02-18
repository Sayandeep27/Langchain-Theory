# LangChain Runnables — Complete Professional Guide (LCEL)

---

> Covers **Runnables**, **RunnableSequence**, **RunnableParallel**, **RunnablePassthrough**, **Streaming**, and **RunnableMap vs RunnableParallel** 

---

# 1. What is a Runnable?

In **LangChain**, **Runnables** are the **core execution abstraction** used in **LCEL (LangChain Expression Language)**.
They define **how data flows between components** like prompts, LLMs, retrievers, parsers, and custom functions.

Think of a Runnable as:

> **Anything that takes input → processes it → returns output**

This unified interface allows chaining, parallel execution, branching, streaming, and composition easily.

---

## Every Major Component is a Runnable

* PromptTemplate
* LLM / ChatModel
* OutputParser
* Retriever
* Custom Python function
* Chains

So instead of different APIs, everything behaves the same.

---

## Basic Runnable Example

```python
from langchain_core.runnables import RunnableLambda

def add_exclamation(text):
    return text + "!"

runnable = RunnableLambda(add_exclamation)

runnable.invoke("Hello")
```

Output:

```
Hello!
```

---

## Runnable Methods

| Method      | Purpose             |
| ----------- | ------------------- |
| `invoke()`  | Run once            |
| `batch()`   | Run multiple inputs |
| `stream()`  | Stream output       |
| `ainvoke()` | Async execution     |

Example:

```python
runnable.batch(["Hi", "Hello"])
```

---

# 2. Runnable Sequence (Pipeline Execution)

## Concept

A **RunnableSequence** executes components **one after another**.

```
Input
  ↓
Prompt
  ↓
LLM
  ↓
Parser
  ↓
Output
```

Equivalent to a **pipeline**.

---

## How to Create a Runnable Sequence

Using LCEL operator `|`

```python
chain = runnable1 | runnable2 | runnable3
```

This is the MOST IMPORTANT concept in modern LangChain.

---

## Example: Prompt → LLM → Parser

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

llm = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "LangChain"})
print(result)
```

### Flow

```
{"topic": "LangChain"}
        ↓
Prompt formats text
        ↓
LLM generates response
        ↓
Parser converts to string
```

---

## Custom RunnableSequence Example

```python
from langchain_core.runnables import RunnableLambda

chain = (
    RunnableLambda(lambda x: x + 1)
    | RunnableLambda(lambda x: x * 2)
)

chain.invoke(3)
```

Output:

```
8
```

Because:

```
3 + 1 = 4
4 * 2 = 8
```

---

# 3. Runnable Parallel (Run Tasks Simultaneously)

## Concept

Runs multiple runnables **at the same time** using the SAME input.

```
          → Runnable A →
Input →
          → Runnable B →
```

Useful when:

* Multiple LLM calls
* Multiple retrievers
* Feature generation
* Multi-agent outputs

---

## Syntax

```python
from langchain_core.runnables import RunnableParallel
```

---

## Example

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

runnable = RunnableParallel(
    double=RunnableLambda(lambda x: x * 2),
    square=RunnableLambda(lambda x: x ** 2),
)

runnable.invoke(3)
```

Output:

```python
{
  "double": 6,
  "square": 9
}
```

---

### Execution Diagram

```
        → x * 2  → 6
3 →
        → x^2    → 9
```

---

## Real LLM Example (Very Important)

Generate:

* summary
* keywords
* sentiment

simultaneously.

```python
parallel_chain = RunnableParallel(
    summary=prompt_summary | llm | parser,
    sentiment=prompt_sentiment | llm | parser,
)
```

One input → multiple outputs.

---

# 4. Runnable Passthrough (Keep Original Input)

## Concept

Sometimes you want to:

* keep original input
* add new computed fields

This is what **RunnablePassthrough** does.

It **passes input forward unchanged**.

---

## Why Needed?

Suppose:

```
question → retriever → context
```

But you ALSO need the original question later.

Passthrough preserves it.

---

## Example

```python
from langchain_core.runnables import RunnablePassthrough
```

---

### Basic Example

```python
chain = RunnablePassthrough()

chain.invoke("Hello")
```

Output:

```
Hello
```

(It does nothing — just forwards.)

---

## Real Use Case (RAG Pattern)

### Step 1 — Retrieve context but keep question

```python
rag_chain = {
    "context": retriever,
    "question": RunnablePassthrough()
}
```

Input:

```
"What is LangChain?"
```

Output becomes:

```python
{
  "context": "...retrieved docs...",
  "question": "What is LangChain?"
}
```

---

### Step 2 — Send both into prompt

```python
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)
```

---

### Flow Diagram

```
Question
   ↓
 ┌───────────────┐
 │ Parallel Step │
 │               │
 │ retriever     │ → context
 │ passthrough   │ → question
 └───────────────┘
         ↓
      Prompt
         ↓
        LLM
```

This pattern is used in **almost every modern RAG pipeline**.

---

# 5. Combining Sequence + Parallel + Passthrough

This is where LangChain becomes powerful.

---

## Advanced Example (Production Style RAG)

```python
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Multi-Step Hybrid Example

```python
chain = (
    RunnableParallel(
        original=RunnablePassthrough(),
        processed=RunnableLambda(lambda x: x.upper())
    )
    | RunnableLambda(lambda x: f"{x['original']} -> {x['processed']}")
)

chain.invoke("hello")
```

Output:

```
hello -> HELLO
```

---

# 6. Mental Model (VERY IMPORTANT)

## RunnableSequence

```
A → B → C
(step-by-step pipeline)
```

---

## RunnableParallel

```
        → A →
Input →
        → B →
(run simultaneously)
```

---

## RunnablePassthrough

```
Input → (unchanged forward)
```

Used for **data routing**.

---

# 7. When to Use What

| Runnable        | Use Case                         |                  |
| --------------- | -------------------------------- | ---------------- |
| **Sequence (`   | `)**                             | Linear pipelines |
| **Parallel**    | Multiple outputs from same input |                  |
| **Passthrough** | Preserve original input          |                  |
| **Lambda**      | Custom logic                     |                  |

---

# 8. Why Runnables Are Important (Big Picture)

Old LangChain:

```
LLMChain
SequentialChain
Custom chains
```

New LangChain (LCEL):

```
Everything = Runnable
```

Benefits:

* composable
* async-ready
* streaming support
* easy debugging
* parallel execution
* production scalable

---

# 9. One-Line Summary

* **RunnableSequence** → do tasks **one after another**
* **RunnableParallel** → do tasks **at the same time**
* **RunnablePassthrough** → **keep input while adding new data**

---

# PART 1 — Streaming with Runnables

---

## 1. What is Streaming?

Normally:

```
User → LLM → FULL response returned
```

You wait until generation finishes.

**Streaming** means:

```
User → LLM → token → token → token → token
```

Output arrives **gradually**.

Example:

Instead of:

```
"LangChain is a framework for..."
```

You receive:

```
Lang
LangChain
LangChain is
LangChain is a
...
```

---

## 2. Why Streaming is Important

Streaming is critical for:

* Chat applications (ChatGPT-like UX)
* Low perceived latency
* Real-time UI updates
* Voice assistants
* Agents showing thinking steps

---

## 3. Streaming in Runnables

Every Runnable supports:

```
.stream()
```

instead of:

```
.invoke()
```

---

## Basic Example

### Normal Execution

```python
response = chain.invoke({"topic": "AI"})
print(response)
```

Waits → prints full output.

---

### Streaming Execution

```python
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

Now tokens arrive live.

---

## Example Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply"
)

llm = ChatOpenAI(streaming=True)

chain = prompt | llm | StrOutputParser()
```

Streaming:

```python
for token in chain.stream({"topic": "LangChain"}):
    print(token, end="")
```

---

## Execution Flow During Streaming

```
Input
  ↓
Prompt (instant)
  ↓
LLM (streams tokens)
  ↓
Parser processes chunks
  ↓
Output streamed
```

Only components that support streaming actually stream (LLM usually).

---

## Important Concept

### Streaming propagates automatically

If one component streams → whole chain streams.

You don’t need special logic.

---

## Streaming in RAG (Real Use Case)

```python
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Streaming:

```python
for chunk in rag_chain.stream("What is RAG?"):
    print(chunk, end="")
```

Retriever runs first → LLM streams answer.

---

## Advanced: Stream Events (Agent Debugging)

You can stream internal events:

```python
for event in chain.stream_events(input):
    print(event)
```

You’ll see:

* start
* end
* tokens
* intermediate steps

Used heavily in **LangSmith debugging**.

---

## Mental Model

```
invoke()  = final answer
stream()  = live generation
batch()   = many inputs
```

---

# PART 2 — RunnableMap vs RunnableParallel (BIG CONFUSION)

This confuses almost everyone learning LCEL.

Let’s fix it permanently.

---

## First: RunnableParallel

You already saw:

```python
RunnableParallel(
    a=runnable1,
    b=runnable2
)
```

### Behavior

* SAME input
* MULTIPLE runnables
* RUNS simultaneously

```
input → A
      → B
```

Output:

```python
{
  "a": resultA,
  "b": resultB
}
```

---

## RunnableMap — What is it?

RunnableMap applies **different runnables to different input fields**.

It works on **dictionary inputs**.

---

### Key Difference (IMPORTANT)

| Feature                | RunnableParallel    | RunnableMap          |
| ---------------------- | ------------------- | -------------------- |
| Input                  | Single value        | Dictionary           |
| Input shared?          | YES                 | NO                   |
| Each runnable receives | Same input          | Different field      |
| Purpose                | Fan-out computation | Field transformation |

---

## RunnableParallel Example

```python
parallel = RunnableParallel(
    double=lambda x: x*2,
    square=lambda x: x**2
)

parallel.invoke(3)
```

Output:

```
{
 "double": 6,
 "square": 9
}
```

Both receive **3**.

---

## RunnableMap Example

Input is structured:

```python
{
   "name": "John",
   "age": 25
}
```

We want different processing.

---

### Example Code

```python
from langchain_core.runnables import RunnableMap, RunnableLambda

mapper = RunnableMap({
    "name": RunnableLambda(lambda x: x.upper()),
    "age": RunnableLambda(lambda x: x + 10)
})

mapper.invoke({
    "name": "john",
    "age": 25
})
```

Output:

```python
{
  "name": "JOHN",
  "age": 35
}
```

Each runnable receives its **own field**.

---

## Visual Difference

---

### RunnableParallel

```
        → summarize →
Input →
        → sentiment →
```

Same input everywhere.

---

### RunnableMap

```
{
 name → process_name
 age  → process_age
}
```

Field-wise transformation.

---

## Real RAG Example (Where People Get Confused)

### Using Parallel

```python
{
   "context": retriever,
   "question": RunnablePassthrough()
}
```

Input:

```
"What is AI?"
```

Both get SAME input.

---

### Using Map (Field Processing)

Suppose prompt output:

```python
{
   "question": "...",
   "context": "..."
}
```

Now we want:

* embed context
* clean question

```python
RunnableMap({
   "question": clean_question,
   "context": embed_docs
})
```

Each field handled separately.

---

## Shortcut Syntax (Important)

In LCEL:

```python
{
   "a": runnable1,
   "b": runnable2
}
```

is internally converted to **RunnableParallel**.

This is why beginners think Map and Parallel are identical.

They are NOT.

---

## When to Use Which

### Use RunnableParallel when:

* One input → many outputs
* Multi-LLM calls
* Multi-retriever pipelines
* Feature generation

Example:

```
question → summary + keywords + sentiment
```

---

### Use RunnableMap when:

* Processing structured data
* Transforming dictionary fields
* Post-processing outputs

Example:

```
{name, age, city} → clean each field differently
```

---

# SUPER IMPORTANT MEMORY TRICK

## RunnableParallel

> **Same input → many runnables**

---

## RunnableMap

> **Many inputs (fields) → matching runnables**

---

# Combined Example (Expert-Level)

```python
chain = (
    RunnableParallel(
        raw=RunnablePassthrough(),
        analysis=llm_chain
    )
    | RunnableMap({
        "raw": RunnableLambda(str.upper),
        "analysis": RunnableLambda(lambda x: x[:50])
    })
)

chain.invoke("Explain AI")
```

Flow:

```
Input
 ↓
Parallel (duplicate processing)
 ↓
Map (field-wise transformation)
 ↓
Output
```

---

# Final Intuition (Think Like Data Flow)

```
Sequence  → assembly line
Parallel  → cloning machine
Map       → column-wise processor
Passthrough → wire connection
Streaming → live output pipe
```

---

**END OF DOCUMENT**
