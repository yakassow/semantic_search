# Semantic Search for Administrative Texts 🏛️🔍

> **Note:** This repository serves as a proof-of-concept (PoC) to demonstrate the practical application of Natural Language Processing (NLP) in the context of German administrative and legal texts.

## 📌 Project Overview
Finding specific regulations in bureaucratic documents is often frustrating because traditional keyword searches fail when synonyms are used (e.g., searching for "Gartenzaun" when the law says "Einfriedung"). 

Drawing from my background in both Computational Linguistics and Law, this project implements a **Semantic Search Engine**. Instead of exact string matching, it uses transformer-based vector embeddings to understand the *meaning* of a user's query and retrieve the most contextually relevant legal paragraph.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **NLP & Embeddings:** `sentence-transformers` (Hugging Face)
* **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Optimized for German and multilingual semantic similarity)
* **Data Processing:** `numpy` for cosine similarity calculations

## 🚀 How It Works (The Pipeline)
1. **Document Ingestion:** Loads sample paragraphs from administrative regulations (e.g., Bavarian Building Code / BayBO).
2. **Vectorization:** Converts the text chunks into dense vector representations (embeddings).
3. **Query Processing:** Embeds the user's natural language question (e.g., "Wie hoch darf mein Zaun sein?").
4. **Retrieval:** Calculates the cosine similarity via dot product to find and return the best-matching legal document.

## 🚧 Current Status & Next Steps
* **Phase 1 (Done):** Core retrieval logic and embedding pipeline implemented.
* **Phase 2 (Planned):** Replacing the hardcoded array with an external document loader (e.g., reading `.txt` or `.pdf` files).
* **Phase 3 (Planned):** Implementing a simple UI using `Streamlit` for easier interaction.
