# RAG Retrieval Evaluation: Two-Pipeline Model

## Pipeline 1 – Evaluation / Regression Suite
**Purpose**: Prove retrieval quality and guard against regressions **before** deploying.

**When run**:
- During development/model selection
- Before releasing a new retriever index, embedding model, or query rewrite logic
- On every CI/CD run affecting retrieval

**Typical query scope**:
- Large, representative query set (hundreds–thousands)
- Covers multiple user intents and domains, not just anchors

**Tests included**:
1. **Ground Truth Recall**
   - Measure recall@K against a **large gold set**
   - Optimizes embedding models, chunk sizes, ranking strategies
2. **Direct Recall**
   - “Must-hit” regression checks for business-critical docs
   - Failures here block deploy
3. **Other offline retrieval metrics**
   - MRR, NDCG, precision@K
4. **Optional similarity drift check**
   - Run vs. a known-good frozen index as an early smoke test
   - Catches major distribution changes

**Key characteristics**:
- Larger query set = more coverage
- More metrics (recall curves, per-domain breakdowns)
- Not tied to live system changes — this is offline evaluation

---

## Pipeline 2 – Contextual Drift Detection Suite
**Purpose**: Monitor the live retriever for **gradual semantic drift** in production.

**When run**:
- On a schedule (daily/hourly) in staging and/or production
- After deploys as a canary check

**Typical query scope**:
- **Anchor queries only** — fixed, small set (e.g., 20–50)
- Chosen for high business value and domain coverage

**Tests included**:
1. **Ground Truth Recall (anchor version)**
   - Check if recall@K drops for anchors compared to baseline
2. **Direct Recall (anchor version)**
   - Must-hit docs for anchors — failure = P1 incident
3. **Similarity Drift**
   - Compare embeddings of current top_K vs. baseline top_K from a known-good run
4. **Domain Term Frequency Shift**
   - Track frequency of critical domain terms
5. **Topic Mix Shift** (optional)
   - Compare topic distribution to baseline

**Key characteristics**:
- Fast to run — designed for ongoing monitoring
- Uses smaller query set but more **baseline-dependent** metrics
- Alerts rather than full evaluation reports
- Focused on detecting **change over time** rather than absolute quality

---

## How They Work Together
- **Evaluation suite** = broad coverage, run before changes to validate overall retrieval performance
- **Drift suite** = narrow but high-value coverage, run continuously to detect gradual semantic or content shifts in production

**Shared components**:
- Both can run **Ground Truth Recall** and **Direct Recall** — difference is the **query set** and **purpose**
- Drift suite adds baseline-comparison metrics (similarity drift, term frequency shift) that wouldn’t make sense in a purely pre-deploy setting
