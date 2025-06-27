# 🧪 RAG Testing Framework – Retrieval-Augmented Generation Evaluation with `pytest`

## 📌 Overview
This project is a test framework built in Python using `pytest` to evaluate **Retrieval-Augmented Generation (RAG)** pipelines. It is designed to test both the **retrieval accuracy** and the **generation quality** of LLM-based systems.

The goal is to simulate real-world QA/test scenarios for AI-powered applications using RAG workflows — with an emphasis on:
- Retrieval correctness (e.g., top-k document relevance)
- Generation fidelity (e.g., semantic alignment, hallucination resistance)
- Edge cases and adversarial queries

> ⚠️ This is a self-guided learning project. While not deployed in production, it is structured to mirror real evaluation pipelines and could be adapted for staging/test environments.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation** is a technique where external documents are retrieved via vector search and passed to an LLM to improve factual accuracy and grounding. It typically involves:

1. **Embedding a user query**
2. **Retrieving top-k matching documents from a vector store**
3. **Feeding those documents + query to an LLM for final response generation**

This framework helps validate both steps:
- **Retrieval**: Are the right documents returned?
- **Generation**: Does the LLM produce a response consistent with the retrieved evidence?

---

## 🧪 What This Framework Tests

| Test Category         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Retrieval tests**   | Verifies that top-k documents include expected matches (based on embeddings) |
| **Generation tests**  | Verifies that the LLM output semantically aligns with expectations       |
| **Adversarial cases** | Tests model behavior with misleading, ambiguous, or distractor queries |
| **Semantic diff**     | Optional embedding similarity check between expected and actual outputs |

---

## 🗂 Project Structure

