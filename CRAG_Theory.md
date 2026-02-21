# Corrective Retrieval‑Augmented Generation (CRAG)

---

## 📌 Overview

**Corrective Retrieval‑Augmented Generation (CRAG)** is an advanced Retrieval‑Augmented Generation (RAG) architecture designed to improve the reliability, accuracy, and factual grounding of Large Language Model (LLM) responses.

Traditional RAG retrieves documents based only on similarity scores and directly sends them to the LLM for generation. However, similarity does **not guarantee correctness**.

CRAG introduces a **knowledge correction layer** that evaluates, filters, refines, and supplements retrieved knowledge before answer generation.

---

## 🎯 Why CRAG is Needed

Below are the key reasons why CRAG improves traditional RAG systems:

| Problem              | Description                                              | How CRAG Solves It                    |
| -------------------- | -------------------------------------------------------- | ------------------------------------- |
| Irrelevant Retrieval | Similar documents may not answer the query               | Evaluates and filters documents       |
| Noise & Errors       | Outdated or low‑quality information appears in retrieval | Removes noisy content                 |
| Hallucinations       | LLM generates incorrect answers from poor context        | Validates knowledge before generation |
| Reliability          | Critical domains require verified data                   | Adds evaluation and correction steps  |
| Ranking Issues       | Similarity ranking is imperfect                          | Re‑ranks using quality + relevance    |
| Dynamic Knowledge    | Static KB becomes outdated                               | Triggers web search when needed       |
| Bias Reduction       | Retrieval may favor frequent patterns                    | Adds validation beyond similarity     |

---

## 🧠 Example Problem

### Query

```
What do koalas eat?
```

### Vanilla RAG Retrieval

| Retrieved Document           | Status       |
| ---------------------------- | ------------ |
| Koalas eat eucalyptus leaves | ✅ Relevant   |
| Pandas eat bamboo            | ❌ Irrelevant |
| Kangaroos graze grass        | ❌ Irrelevant |

Mixed context may confuse the LLM.

### CRAG Solution

CRAG filters irrelevant documents and keeps only validated information.

Final Answer:

```
Koalas primarily eat eucalyptus leaves.
```

---

## 🏗️ CRAG Architecture

CRAG divides the pipeline into three main stages:

```
1. Retrieval
2. Knowledge Correction
3. Generation
```

The **Knowledge Correction Layer** is the key innovation.

---

## ⚙️ Step‑by‑Step Working of CRAG

---

### 1️⃣ Input Query

The system begins with a user query:

```
X = "What do koalas eat?"
```

---

### 2️⃣ Retrieval (Vanilla RAG Step)

The retriever searches the vector database.

```
Top‑K documents → d1, d2, d3...
```

Selection is based only on embedding similarity.

---

### 3️⃣ Retrieval Evaluator (Core CRAG Component)

The evaluator checks whether retrieved documents truly answer the query.

#### Evaluation Criteria

* Semantic relevance
* Factual correctness
* Completeness
* Consistency
* Freshness

Each document receives a **relevance score**.

---

### 4️⃣ Decision Phase

CRAG classifies retrieval quality into three categories.

#### ✅ Correct

```
At least one document has high relevance.
```

Action:

```
Proceed to Knowledge Refinement
```

---

#### ⚠️ Ambiguous

```
Medium confidence in retrieved documents.
```

Action:

```
Combine internal + external knowledge
```

---

#### ❌ Incorrect

```
All documents have low relevance.
```

Action:

```
Trigger external web search
```

---

### 5️⃣ Corrective Step — Knowledge Refinement (If Correct)

Instead of directly sending documents to the LLM, CRAG refines them.

#### (a) Decompose

```
Document → smaller strips/chunks
```

Purpose:

* Fine‑grained filtering
* Remove irrelevant sentences

---

#### (b) Filter

Removes:

* noisy content
* outdated information
* unrelated text

---

#### (c) Re‑rank

Documents are ranked using:

| Factor     | Meaning           |
| ---------- | ----------------- |
| Similarity | Semantic match    |
| Quality    | Factual accuracy  |
| Freshness  | Updated knowledge |
| Coverage   | Completeness      |

---

#### (d) Deduplication

Prevents repeated or duplicated information.

---

#### (e) Recompose

Filtered knowledge becomes:

```
k_in (internal refined knowledge)
```

---

### 6️⃣ Web Search (If Incorrect)

CRAG rewrites the query:

```
Original: What do koalas eat?
Rewritten: koala diet eucalyptus leaves wikipedia
```

Then performs web search:

```
k1, k2, k3 → Selected → k_ex
```

Where:

```
k_ex = external knowledge
```

This enables dynamic knowledge retrieval.

---

### 7️⃣ Knowledge Combining (If Ambiguous)

CRAG merges both knowledge sources:

```
k_in + k_ex
```

Reason:

* Internal data may be partially correct
* External search fills missing gaps

---

### 8️⃣ Answer Generation

The LLM receives only corrected knowledge.

| Decision  | Generator Input |
| --------- | --------------- |
| Correct   | X + k_in        |
| Ambiguous | X + k_in + k_ex |
| Incorrect | X + k_ex        |

Final answer is generated using validated context.

---

## 🔄 Full CRAG Pipeline

```
Query
  ↓
Retrieve Documents
  ↓
Evaluate Retrieval Quality
  ↓
Decision
  ├── Correct → Refine Knowledge
  ├── Ambiguous → Combine Knowledge
  └── Incorrect → Web Search
  ↓
Generate Answer
```

---

## ✅ Advantages of CRAG

| Advantage              | Description                         |
| ---------------------- | ----------------------------------- |
| Improved Accuracy      | Filters misleading information      |
| High Reliability       | Suitable for critical domains       |
| Reduced Hallucinations | Validates context before generation |
| Domain Adaptability    | Custom evaluators possible          |
| Better Reasoning       | Cleaner context improves logic      |
| Explainability         | Decisions are traceable             |
| Optimized QA Mapping   | Matches intent, not just similarity |

---

## ⚠️ Challenges of CRAG

| Challenge            | Description                            |
| -------------------- | -------------------------------------- |
| Complex Architecture | More components than RAG               |
| Scalability Issues   | Higher compute requirements            |
| Domain Dependence    | Correction models may need tuning      |
| High Latency         | Extra evaluation steps                 |
| Over‑Filtering       | Important data may be removed          |
| External Bias Risk   | Web sources may contain misinformation |

---

## 📊 CRAG vs Traditional RAG

| Feature               | RAG    | CRAG   |
| --------------------- | ------ | ------ |
| Retrieval Validation  | ❌      | ✅      |
| Noise Filtering       | ❌      | ✅      |
| Dynamic Knowledge     | ❌      | ✅      |
| Hallucination Control | Medium | Strong |
| Reliability           | Medium | High   |
| System Complexity     | Low    | High   |

---

## 🧩 One‑Line Intuition

**RAG:**

```
Retrieve → Generate
```

**CRAG:**

```
Retrieve → Verify → Correct → Generate
```

---

## 📚 Key Takeaway

CRAG enhances traditional RAG by introducing a **self‑correcting retrieval mechanism** that ensures only reliable, relevant, and validated knowledge reaches the LLM, significantly improving factual accuracy and reducing hallucinations.

---

## ⭐ Summary

CRAG transforms retrieval from a passive similarity search into an **active quality‑controlled knowledge pipeline**, making modern AI systems more trustworthy and production‑ready.

---
