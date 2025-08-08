# Building a Large Gold Query Set for RAG Evaluation/Regression Pipeline

## 1. Where the Queries Come From
- **Real usage logs** (best source)
  - Search queries, chat prompts, FAQ clicks, internal helpdesk tickets, support emails.
  - Mine high-frequency queries, high-value journeys, and failure cases.
- **Docs themselves**
  - Turn **headings, TOCs, and section titles** into question templates.
  - Extract Q/A from existing **FAQs, runbooks, SOPs, policy docs**.
- **Subject-matter experts (SMEs)**
  - SME workshops: ask for top 50 “must-answer” questions per domain.
  - Red-team prompts: SMEs craft tricky, easily-confused questions.
- **Structured data**
  - Schema, catalogs, API specs → generate fact-seeking queries that should hit one canonical source.
- **Synthetic generation (controlled!)**
  - Use LLMs to **paraphrase** real queries, generate **entity variations**, and create **hard negatives**.
  - Keep prompts templated to avoid drifting away from real user intent.

---

## 2. How to Turn Queries Into “Gold”
You need **query → relevant doc(s)** labels. Do it in layers:

### A. Automatic/Heuristic Pre-Labeling (cheap, high recall)
- **Exact/near-exact matches**: if a doc section literally answers the query, label it.
- **Anchors & metadata**: map queries that include **official titles/IDs** to the canonical doc.
- **Citations mining**: if your docs cite a canonical source, propagate relevance.
- **Regex & dictionaries**: for compliance terms, product names, SKUs, version numbers.

> Output: a big pile of *candidate* labels with confidence scores.

### B. Human Review (raise precision)
- Build a small adjudication UI (or use a labeling tool). Show:
  - Query, top-20 retrieved candidates, doc snippets.
  - Annotator picks **all** relevant docs (not just one).
- **Guidelines** doc: what counts as “relevant”? (direct answer? evidence? definitional?)
- Use **double annotation** on a 10–20% sample; compute **inter-annotator agreement**; reconcile disagreements → update guidelines.

### C. Active Learning (efficient scaling)
- Train a light relevance model on current labels.
- Sample **uncertain** or **disagreeing** query–doc pairs for human review.
- Iterate until marginal gain drops.

---

## 3. Ensuring Coverage & Quality
- **Coverage matrix**: rows = domains/intents/entities; cols = difficulty (simple, multi-hop, disambiguation). Fill every cell.
- **Balance head/tail**: include frequent queries **and** rare, high-impact ones (audits, outages).
- **Hard negatives**: include near-miss docs that look right but are wrong (e.g., GDPR vs HIPAA) to test rankers.
- **De-dup & paraphrases**: keep 2–5 paraphrases per important query to avoid overfitting to wording.

---

## 4. Packaging the Gold Set (what to store)
Minimal schema (CSV/Parquet/DB):
- `query_id`, `query_text`, `domain`, `intent`, `difficulty`
- `gold_doc_ids` (list), `must_hit_doc_ids` (subset, optional)
- `rationale` (why labeled relevant), `source` (logs/SME/synthetic)
- `version_tags` (embedding_version, index_snapshot_id, guidelines_version)

---

## 5. Maintenance Plan
- **Version everything** (queries, labels, guidelines). Never edit in place—append a new version.
- **Refresh cadence**: monthly for dynamic corpora; quarterly for stable policy sets.
- **Drift intake loop**: any production drift incident → add a new query or label to the gold set.
- **Spot checks**: sample 2–5% each cycle; re-adjudicate to catch label rot.

---

## 6. Privacy & Compliance
- Strip PII from log-derived queries (hash user IDs, redact emails/order IDs).
- Keep a “do not include” list for sensitive or rare personal queries.
- If needed, generate **synthetic look-alikes** from sensitive logs and discard originals.

---

## 7. Quick Bootstrap Recipe (60–90 days)
1. **Week 1–2**: Pull 5–10k raw queries from logs; cluster; pick ~800 diverse reps. Add 200 SME “must-answer.”
2. **Week 3–4**: Heuristic pre-labeling against your corpus; get 3–5 candidates per query.
3. **Week 5–6**: Human adjudication pass on all; double-label 15% for quality.
4. **Week 7–8**: Active learning round; add hard negatives; finalize must-hit list (50–150).
5. **Week 9**: Freeze **Eval v1.0** (queries + labels + guidelines). Wire into CI.

---

## 8. Pro Tips
- Keep **gold set independent** of any single retriever’s baseline results (don’t bake in its biases).
- Track **per-query metrics** (recall@K, MRR) and **per-domain aggregates**; promote failing queries to must-hit if they’re business-critical.
- Store **doc snapshots** (IDs + stable anchors) so labels survive re-chunking; use canonical IDs with stable offsets.
