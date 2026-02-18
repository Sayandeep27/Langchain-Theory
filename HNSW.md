# Hierarchical Navigable Small World (HNSW)

---

## Overview

HNSW graphs enable efficient approximate nearest neighbor (ANN) search through a **multi-layered graph structure**. This approach leverages graph theory to navigate large datasets quickly and accurately by creating a hierarchical system where each level acts as a simplified overview of the one below it.

Building an HNSW involves three major steps:

* Constructing the HNSW index
* Querying vectors
* Performing vector searches

---

## What is HNSW?

HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm that enables fast similarity search in high-dimensional datasets.

### Core Concept

* Uses **hierarchical layers** of proximity graphs
* Higher layers provide coarse navigation
* Lower layers provide fine-grained precision
* Exploits the **small-world phenomenon** to minimize search hops

This structure enables logarithmic scaling complexity.

---

## Building an HNSW

### Major Steps

1. Construct hierarchy
2. Insert elements
3. Optimize connections
4. Perform search

---

## HNSW Index Construction

### 1. Forming the Hierarchy

#### Level Assignment

* Elements are randomly assigned levels.
* Fewer elements exist at higher levels.
* Distribution follows an **exponentially decaying probability**.
* Ensures balanced handling of dynamic datasets.

#### Layer Construction

Probabilistic tiering is governed by parameter **mL**.

* Determines probability of appearing in successive layers
* Probability decreases exponentially as layers ascend
* Each layer forms a proximity graph

Higher layers → broader overview
Lower layers → finer detail

---

### 2. Adding Elements

#### Greedy Search

New elements:

* Start from the top layer
* Connect to nearest neighbors while descending

Key parameters:

| Parameter      | Description                     |
| -------------- | ------------------------------- |
| M              | Maximum connections per element |
| efConstruction | Dynamic candidate list size     |

These parameters influence search quality and efficiency.

---

### 3. Optimization

#### Heuristic Selection

Ensures diverse and effective connections to maintain graph balance.

#### Scale Separation

* Upper layers contain longer links
* Lower layers contain shorter links
* Improves navigation efficiency

---

## Searching in HNSW

### Start at the Top

Begin search from a random high-level entry point.

### Greedy Descent

At each layer:

* Move toward closest neighbor to query vector
* Descend to next level

### Refinement

Lower layers refine results and improve accuracy.

Each trajectory from entry point to closest vector represents a candidate exploration.

### Search Parameter

| Parameter                | Meaning                         |
| ------------------------ | ------------------------------- |
| efSearch (numCandidates) | Number of explored trajectories |

Higher values increase probability of finding true nearest neighbors.

---

## Insertions and Deletions

### Insertions

When a new node is added:

1. Determine hierarchical position via probabilistic tiering (mL).
2. Identify predefined entry point.
3. Establish connections with nearest neighbors.

If neighbors exceed allowed connections:

* Excess links are pruned.

The process repeats layer by layer until reaching the bottom layer.

---

### Deletions

Removing nodes requires:

* Updating neighbor connections
* Maintaining linked lists across layers
* Preserving graph navigability and complexity guarantees

Ensures structural integrity of the graph.

---

## Python Implementation Example

```python
class HNSW(_BaseIndex):

   def __init__(self, L=5, mL=0.62, efc=10):
       self._L = L
       self._mL = mL
       self._efc = efc
       self._index = [[] for _ in range(L)]

   def create(self, dataset):
       for v in dataset:
           self.insert(v)

   def search(self, query, ef=1):
       if not self._index[0]:
           return []
       best_v = 0
       for graph in self._index:
           best_d, best_v = HNSW._search_layer(graph, best_v, query, ef=1)[0]
           if graph[best_v][2]:
               best_v = graph[best_v][2]
           else:
               return HNSW._search_layer(graph, best_v, query, ef=ef)

   def _get_insert_layer(self):
       l = -int(np.log(np.random.random()) * self._mL)
       return min(l, self._L-1)

   def insert(self, vec, efc=10):
       if not self._index[0]:
           i = None
           for graph in self._index[::-1]:
               graph.append((vec, [], i))
               i = 0
           return
       l = self._get_insert_layer()
       start_v = 0
       for n, graph in enumerate(self._index):
           if n < l:
               _, start_v = self._search_layer(graph, start_v, vec, ef=1)[0]
           else:
               node = (vec, [], len(self._index[n+1]) if n < self._L-1 else None)
               nns = self._search_layer(graph, start_v, vec, ef=efc)
               for nn in nns:
                  node[1].append(nn[1])
                  graph[nn[1]][1].append(len(graph))
               graph.append(node)
           start_v = graph[start_v][2]
```

---

## Search Process

HNSW organizes data points hierarchically.

* Higher levels act as shortcuts
* Lower levels refine similarity

Benefits:

* Reduced distance calculations
* Faster navigation
* High scalability

Ideal for real-time applications such as recommendation systems and similarity search.

---

## Tuning Your HNSW Search

### Candidate Expansion

Increasing candidate count improves recall at the cost of latency.

### Prefiltering

* Initial filtering step
* Reduces search space
* Improves efficiency

Compared to brute-force search, HNSW achieves significantly higher speed and precision.

---

## Graph Structure

HNSW builds upon the **small-world property** where most nodes are reachable in few hops.

Each point connects:

* Within its layer
* Across hierarchical levels

Additional structures used:

* Priority queues
* Hash tables

Cosine similarity is commonly used as the distance metric.

---

## Insertions (Detailed)

Upon insertion:

1. Identify entry layer via probabilistic tiering (mL).
2. Connect to M closest neighbors.
3. Prune excess connections if exceeding Mmax.
4. Continue downward until bottom layer.

Connectivity constraints:

* Mmax → upper layers
* Mmax0 → bottom layer

---

## Deletions (Detailed)

Deletion requires careful updates:

* Neighbor links adjusted across layers
* Prevent orphaned nodes
* Preserve navigability

Different implementations apply varied connectivity preservation strategies.

---

## Use Cases of HNSW

### Image Retrieval

* Efficient indexing of image feature vectors
* Enables real-time reverse image search

### Natural Language Processing (NLP)

* Semantic search
* Document similarity
* Question-answering systems

### Music Recommendation

* Fast similarity retrieval for songs
* Personalized playlists

### Anomaly Detection

* Fraud detection
* Cybersecurity monitoring
* Network anomaly identification

### Recommendation Systems

* Collaborative filtering
* User/item similarity search

---

## How to Build an HNSW (Practical Steps)

1. Assess content and identify multilayer structure.
2. Create main categories and subcategories.
3. Design intuitive navigation between layers.
4. Implement search functionality.
5. Regularly review and optimize hierarchy.

Challenges:

* Maintaining simplicity while handling complexity
* Avoiding information overload
* Ensuring logical organization

User testing and continuous optimization are essential.

---

## Tools & Technologies

Helpful tools include:

* Content management systems (e.g., WordPress plugins)
* UX tools like Adobe XD and Sketch

These simplify hierarchical navigation design and maintenance.

---

## SEO Optimization with Navigable Small World Graphs

Best practices:

* Use descriptive keyword-rich headings
* Maintain clear hierarchical URL structure
* Add internal linking between related categories
* Optimize images and minimize code for speed
* Regularly update content for relevance

---

## Conclusion

HNSW significantly enhances:

* User experience
* Content discoverability
* Semantic search efficiency

Vector databases frequently rely on HNSW for ANN queries. For example, MongoDB Atlas Vector Search uses HNSW to enable semantic search across products, images, and documents.

Understanding and implementing HNSW enables scalable, high-performance similarity search systems capable of handling large high-dimensional datasets efficiently.

---

**End of README**
