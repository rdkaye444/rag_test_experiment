# Contextual Drift Detection Cheat Sheet (RAG Systems)

## 1. Purpose
Detect when a RAG retriever starts returning **semantically different** or **less relevant** documents for the same queries over time, even if they still appear plausible.

---

## 2. Core Concept
- **Anchor Queries**: Fixed set of high-value queries representing critical user needs.
- **Baseline Retrieval Set**: Top_K results (text, IDs, embeddings) for each anchor query from a *known-good* system state.
- **Gold Set**: Curated list of relevant documents for each anchor query.
- **Must-Hit Docs**: Subset of gold docs that are critical and must always be retrieved.

---

## 3. Key Drift Detection Tests

### A. Ground Truth Recall
**Goal**: Ensure retriever still surfaces *any* gold-labeled doc for each anchor query.
- **Pass Criteria**: At least one doc in `current_topK` is in the gold set.
- **Signal Type**: Coverage metric.
- **Drift Indication**: Drop in recall@K over baseline.

### B. Direct Recall
**Goal**: Ensure must-hit docs are always retrieved for certain queries.
- **Pass Criteria**: All must-hit docs appear in `current_topK`.
- **Signal Type**: Fidelity metric (P1 alert if fail).
- **Drift Indication**: Must-hit doc missing = immediate high-priority drift.

### C. Similarity Drift
**Goal**: Detect semantic neighborhood changes in retrieved docs.
- **Method**: 
  1. For each baseline doc, find most similar doc in current_topK.
  2. Compute average max cosine similarity.
- **Signal Type**: Baseline comparison metric.
- **Drift Indication**: Significant drop in average similarity.

### D. Domain Term Frequency Shift
**Goal**: Detect erosion of critical domain-specific vocabulary.
- **Method**: Compare term frequencies of key domain terms in current_topK vs. baseline.
- **Drift Indication**: Median frequency drop > threshold (e.g., -30%).

### E. Topic Mix Shift (Optional)
**Goal**: Detect change in topic/category distribution of retrieved docs.
- **Method**: Classify docs into topics, compare distribution to baseline using KL-divergence.
- **Drift Indication**: Divergence > threshold.

---

## 4. Workflow

1. **Baseline Creation (Once per system version)**
   - Select anchor queries.
   - Retrieve top_K results from known-good system.
   - Store:
     - Document text, IDs, embeddings
     - Domain term stats
     - Topic/classific
