# RAGA: RAG AI Assistant

RAGA is a local RAG AI Assistant that helps users ask questions about company documents. It supports document upload, hybrid retrieval, source-backed answers, document summaries, and simple explanations for non-technical users.

The project is designed around a practical business problem: employees often need answers from HR policies, onboarding guides, IT support documents, and internal company files, but searching through multiple documents manually is slow and frustrating.

RAGA provides a simple chat interface where users can ask questions and receive answers grounded in uploaded documents.

---

## Live Demo

Try RAGA here:

https://raga-rag-assistant.streamlit.app/

Note: The first load may take a little time because the app loads the local embedding model.

---

## Features

* Upload and read company documents
* Ask questions through a chat interface
* Retrieve relevant document sections using hybrid retrieval
* Show retrieved evidence for transparency
* Summarize uploaded documents
* Explain answers in simpler terms when users do not understand
* Support multiple document formats
* Run locally without requiring a paid API key

---

## Supported File Types

RAGA currently supports:

* `.txt`
* `.pdf`
* `.docx`
* `.pptx`

PowerPoint files are processed slide by slide, allowing RAGA to show evidence from specific slides.

Image files such as `.png` and `.jpeg` are not currently supported because they require OCR.

---

## Tech Stack

- Python
- Streamlit
- rank-bm25
- sentence-transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- pypdf
- python-docx
- python-pptx
- Pillow

---

## Retrieval Approach

RAGA uses a hybrid retrieval approach that combines keyword retrieval and semantic vector retrieval.

### Keyword Retrieval

Keyword retrieval is handled using BM25 through the `rank-bm25` library. This helps RAGA find exact terms such as policy names, file-specific wording, technical terms, and keywords like "VPN", "annual leave", or "password reset".

### Vector Retrieval

Vector retrieval uses `sentence-transformers/all-MiniLM-L6-v2` to create semantic embeddings for document chunks and user questions.

This allows RAGA to find relevant information even when the user phrases a question differently from the wording in the document.

For example:

- User asks: "Can I work from home?"
- Document says: "Employees may work remotely."

Vector retrieval helps connect those two meanings.

### Hybrid Retrieval

RAGA combines BM25 keyword results and vector search results using weighted Reciprocal Rank Fusion. This helps balance exact keyword matching with semantic meaning-based search while allowing the retrieval weights to be adjusted and compared.

## How It Works

RAGA uses a Retrieval-Augmented Generation style pipeline.

```text
Uploaded documents
        ↓
Document loader
        ↓
Text chunker
        ↓
Keyword retrieval
        ↓
Vector retrieval
        ↓
Hybrid retrieval
        ↓
Local answer generator
        ↓
Streamlit chat interface
```

The assistant does not answer from general knowledge. It searches the uploaded documents first, retrieves the most relevant chunks, and then generates a local answer from the retrieved evidence.

---

## Local Answer Generation

RAGA currently uses a free local answer generator rather than a paid LLM API.

The system selects the most relevant retrieved evidence and extracts a useful answer from it. This keeps the project free to run while still demonstrating the core RAG pipeline.

Optional LLM-based answer generation could be added later.

---

## Summary Mode

Users can ask RAGA to summarize documents.

Example prompts:

```text
Summarize all documents
Give me a simple summary
Summarize the onboarding document
```

The summary mode uses extractive summarization, meaning it selects important sentences from the uploaded documents.

---

## Simple Explanation Mode

RAGA can simplify answers for non-technical users.

Example prompts:

```text
I do not understand
Explain this simply
Explain in plain English
```

This feature helps make company documentation easier to understand.

---

## Example Questions

Users can ask:

```text
How many annual leave days do employees get?
Can employees work remotely?
How do I reset my password?
Do I need VPN access?
Summarize all documents.
Explain that in simple terms.
```

---

RAG_AI_Assistant/
├── app.py
├── README.md
├── requirements.txt
├── evaluate_retrieval.py
├── compare_hybrid_weights.py
├── inspect_chunks.py
├── assets/
│   └── wheel.png
├── data/
│   ├── uploads/
│   └── evaluation/
│       └── questions.json
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── retriever.py
│   ├── vector_retriever.py
│   ├── hybrid_retriever.py
│   ├── local_answerer.py
│   ├── summarizer.py
│   └── simplifier.py
└── tests/

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MahadevPradeepMenon/RAGA-RAG_AI_Assistant.git
cd RAGA-RAG_AI_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate.bat
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Retrieval Evaluation

RAGA includes a small retrieval evaluation script to measure whether the retriever is returning the expected source documents for test questions.

The evaluation set is stored in:

```text 
data/evaluation/questions.json
```
---

### Hybrid Weight Comparison

RAGA compares different BM25/vector weighting settings for hybrid retrieval.

On the current demo evaluation set, the vector-heavy setting performed best:

| Setting | BM25 Weight | Vector Weight | Hit@1 | Hit@3 | Recall@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Keyword-heavy | 0.70 | 0.30 | 0.95 | 1.00 | 1.00 | 0.97 |
| Balanced | 0.50 | 0.50 | 0.95 | 1.00 | 1.00 | 0.97 |
| Vector-heavy | 0.30 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 |

Based on these results, RAGA uses a vector-heavy hybrid retrieval configuration by default.

---

### Embedding Model Comparison

RAGA compares different embedding models to check whether vector retrieval performance improves with alternative models.

On the current demo evaluation set, the default MiniLM model and the QA-focused MiniLM model both achieved perfect retrieval scores:

| Model | Hit@1 | Hit@3 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| MiniLM baseline | 1.00 | 1.00 | 1.00 | 1.00 |
| QA-focused MiniLM | 1.00 | 1.00 | 1.00 | 1.00 |
| Paraphrase MiniLM | 0.95 | 1.00 | 1.00 | 0.97 |

Based on these results, RAGA keeps `sentence-transformers/all-MiniLM-L6-v2` as the default embedding model because it is lightweight and performs strongly on the current evaluation set.

---

### Evaluation Dashboard

RAGA also includes a Streamlit evaluation dashboard for inspecting retrieval performance visually.

To run it:

```bash
streamlit run evaluation_dashboard.py
```

---

### Current Evaluation Results

Using the demo evaluation set of 20 questions across the included sample documents, RAGA achieved:

| Metric | Score |
|---|---:|
| Hit@1 | 0.95 |
| Hit@3 | 1.00 |
| Recall@3 | 1.00 |
| MRR | 0.97 |

These results are based on source-level retrieval evaluation using the included demo documents. 

The evaluation set is intentionally small and based on demo documents, so these results should be treated as a basic validation of the retrieval pipeline rather than a full benchmark.

---

## Current Limitations

* No OCR support for scanned PDFs or image files
* Scanned PDFs may not extract text correctly
* Uses local extractive answer generation rather than a full LLM
* Local answers may be less fluent than LLM-generated responses
* Vector retrieval uses `sentence-transformers/all-MiniLM-L6-v2`, which is lightweight but may not perform as strongly as larger embedding models
* The evaluation set is currently small and based on demo documents
* No authentication or user accounts
* Not deployed yet

---

## Why This Project Matters

RAGA demonstrates practical AI system design rather than just a chatbot interface.

It shows:

* document ingestion
* chunking
* keyword retrieval
* vector retrieval
* hybrid retrieval
* source-backed answers
* summarization
* simple explanation mode
* user-focused interface design

The project is built around a real business use case: helping non-technical users find answers from internal company documents.
