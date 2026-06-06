# RAGA: RAG AI Assistant

RAGA is a local RAG AI Assistant that helps users ask questions about company documents. It supports document upload, hybrid retrieval, source-backed answers, document summaries, and simple explanations for non-technical users.

The project is designed around a practical business problem: employees often need answers from HR policies, onboarding guides, IT support documents, and internal company files, but searching through multiple documents manually is slow and frustrating.

RAGA provides a simple chat interface where users can ask questions and receive answers grounded in uploaded documents.

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

RAGA combines BM25 keyword results and vector search results using basic rank-fusion logic. This helps balance exact keyword matching with semantic meaning-based search.

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

## Retrieval Approach

RAGA uses hybrid retrieval.

### Keyword Retrieval

Keyword retrieval is useful when exact terms matter, such as:

* VPN
* HR
* password reset
* annual leave
* policy names
* IT helpdesk

### Vector Retrieval

Vector retrieval is useful when the user phrases something differently from the document.

For example:

```text
User question:
Can I work from home?

Document wording:
Employees may work remotely up to three days per week.
```

Vector retrieval helps connect similar meanings even when the words are different.

### Hybrid Retrieval

Hybrid retrieval combines keyword retrieval and vector retrieval to improve accuracy across different question types.

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

## Project Structure

```text
RAG_AI_Assistant/
├── app.py
├── README.md
├── requirements.txt
├── assets/
│   └── wheel.png
├── data/
│   └── uploads/
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── retriever.py
│   ├── vector_retriever.py
│   ├── hybrid_retriever.py
│   ├── local_answerer.py
│   ├── summarizer.py
│   └── simplifier.py
└── tests/
```

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

## Screenshots

Add screenshots here before submitting the project.

Suggested screenshots:

1. RAGA home screen
2. Document upload sidebar
3. Question-answer example
4. Retrieved evidence expanded
5. Summary mode example
6. Simple explanation mode example

---

## Demo Video

Add a short demo video link here.

Suggested demo structure:

1. Explain the problem RAGA solves
2. Show uploaded documents
3. Ask a document-based question
4. Show retrieved evidence
5. Ask for a document summary
6. Ask for a simpler explanation

---

## Current Limitations

* Does not currently support PNG/JPEG OCR
* Scanned PDFs may not extract text correctly
* Does not use a paid LLM API
* Local answer generation is less fluent than a full LLM response
* No authentication or user accounts
* Not deployed yet
* The system currently uses local extractive answer generation rather than a full LLM.
* Vector retrieval uses `sentence-transformers/all-MiniLM-L6-v2`, which is lightweight and suitable for local semantic search, but may not perform as strongly as larger embedding models.
* Chunking is document-type-aware but not yet evaluated across a large benchmark dataset.
* No OCR support for scanned PDFs or image files.

---

## Future Improvements

Possible future improvements include:

* OCR support for scanned PDFs and images
* Optional OpenAI or local LLM answer generation
* FastAPI backend
* Multilingual answers
* Better evaluation metrics for retrieval quality
* Document-level analytics
* Deployment to Streamlit Community Cloud or another hosting platform

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
