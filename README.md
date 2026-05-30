# Data Science Knowledge Assistant — Multi-Document RAG Pipeline

A retrieval-augmented generation (RAG) system that answers natural language 
questions across a corpus of data science and machine learning documents.
Built with LangChain, FAISS, HuggingFace Embeddings, and Google Gemini.

## What it does

Ask any data science or statistics question and get a grounded, cited answer 
pulled directly from source documents — not from the LLM's training data.
The system also includes a Statistical Method Selector and HyDE-enhanced 
retrieval for improved answer quality on complex queries.

## Demo queries

**General knowledge:**
- "What is the difference between prediction and inference?"
- "What is the bias variance tradeoff?"
- "Explain the difference between lasso and ridge regression."

**Statistical Method Selector:**
- "I have 100 observations and 200 features. Which method avoids overfitting?"
- "I need interpretability over accuracy. What method should I use?"

**Cross-document synthesis:**
- "What is the core idea behind RAG and how does it relate to statistical learning?"
- "How does SHAP help interpret predictions from complex models like XGBoost?"

## Documents indexed

| Document | Type | Pages |
|---|---|---|
| An Introduction to Statistical Learning (ISLR, 2nd ed.) | Textbook | 615 |
| Attention is All You Need — Vaswani et al. 2017 | Research Paper | 15 |
| Retrieval-Augmented Generation — Lewis et al. 2020 | Research Paper | 13 |
| XGBoost — Chen & Guestrin 2016 | Research Paper | 8 |
| SHAP — Lundberg & Lee 2017 | Research Paper | 9 |

**Total: 2,461 chunks indexed across 5 documents**

## Tech stack

| Tool | Purpose |
|---|---|
| LangChain + LCEL | Pipeline orchestration |
| FAISS | Vector similarity search |
| HuggingFace all-MiniLM-L6-v2 | Local sentence embeddings |
| Google Gemini 1.5 Flash | Answer generation |
| PyPDF | PDF text extraction |
| Kaggle Notebooks | Development environment |

## Architecture
5 PDFs (textbook + 4 research papers)
↓
PyPDFLoader → raw text per page
↓
RecursiveCharacterTextSplitter
chunk_size=800, chunk_overlap=100
↓
HuggingFace all-MiniLM-L6-v2
384-dimensional vectors, runs locally
↓
FAISS Index → saved to disk (2,461 vectors)
↓
Query → embed → retrieve top 8 chunks → Gemini → grounded answer + citations

## Features

### 1. Multi-document retrieval
Retrieves across all 5 documents simultaneously. A single query can 
synthesize information from the textbook and a research paper in one answer.

### 2. Statistical Method Selector
Structured input interface — describe your data, goal, and constraints, 
and the system recommends the appropriate statistical method with 
justification and citations from ISLR.

### 3. HyDE — Hypothetical Document Embeddings
Instead of embedding the raw question, HyDE first generates a hypothetical 
answer and uses that as the retrieval query. A hypothetical answer resembles 
a real document chunk more than a question does, improving retrieval quality 
for broad or interpretability-focused queries.

**Concrete finding:** The question "Which method should I use when 
interpretability matters more than accuracy?" returned no useful answer 
with standard retrieval. With HyDE, it correctly retrieved chunks about 
decision trees, the flexibility-interpretability tradeoff (Figure 2.7, ISLR), 
and ensemble method limitations — producing a well-cited, accurate answer.

## Key engineering decisions

**Why HuggingFace embeddings over Gemini embeddings?**
Gemini's free tier allows 100 embedding requests per minute. With 2,461 
chunks, using a local embedding model eliminates API quota constraints 
entirely while preserving Gemini quota for generation where quality matters most.

**Why chunk_size=800 with chunk_overlap=100?**
800 characters captures roughly one dense paragraph — large enough to be 
self-contained, small enough to be precise. 100-character overlap prevents 
context loss at chunk boundaries.

**Why k=8 over k=4?**
Tested both retrieval depths on complex multi-concept questions. k=8 produced 
noticeably richer answers — for example, the lasso vs ridge comparison gained 
the constraint geometry explanation (sphere vs polyhedron) only visible at k=8. 
k=4 was more focused but missed relevant content for broad questions.

**Why LCEL over RetrievalQA?**
LangChain deprecated RetrievalQA in favor of the more composable and 
transparent LangChain Expression Language (LCEL) pipeline.

## Findings and limitations

**What works well:**
- Specific technical queries with clear constraints retrieve accurately
- Cross-document synthesis works for well-defined topics
- High-dimensional and regularization questions answered with precise citations
- HyDE significantly improves retrieval for broad interpretability questions

**Known limitation — context scattering:**
When an answer requires synthesizing information spread across multiple 
chapters, standard retrieval sometimes misses relevant chunks. HyDE partially 
addresses this. A more complete solution would involve query decomposition 
or re-ranking — flagged as future work.

**Known limitation — PDF extraction noise:**
R code blocks in ISLR extract with character-level spacing artifacts 
(e.g. `>A u t o [ 1 : 4 ,]`). This affects retrieval quality for 
code-specific queries but not conceptual ones.

## Setup

```bash
pip install langchain langchain-core langchain-google-genai \
            langchain-community faiss-cpu pypdf \
            langchain-text-splitters sentence-transformers
```

Add your Gemini API key as a Kaggle Secret named `GEMINI_API_KEY`.

## Project structure
ds-rag-assistant/
├── notebook.ipynb        # full pipeline
├── islr_faiss_index/     # saved FAISS index (not committed)
│   ├── index.faiss
│   └── index.pkl
└── README.md

## Future work
- RAGAS evaluation — formal scoring on faithfulness and answer relevancy
- Query decomposition for multi-hop questions
- Streamlit deployment on Hugging Face Spaces
- Chunking strategy comparison (fixed vs semantic chunking)
