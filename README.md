📘 RAG Pipeline Optimizer (Groq + Streamlit + FAISS + LangChain)

A multi-pipeline Retrieval-Augmented Generation system with automated LLM-based evaluation.

🚀 Overview

The RAG Pipeline Optimizer is an intelligent system that evaluates multiple RAG pipelines in parallel to determine which retrieval strategy produces the best answer for any question.

It uses:

Groq LLaMA models for ultra-fast inference

FAISS and Chroma for vector retrieval

Sentence-Transformers for embeddings

Streamlit for a beautiful UI

LLM-as-a-Judge to automatically select the best pipeline

This tool is perfect for:

Researching RAG configurations

Comparing chunk sizes, models, and retrieval strategies

Understanding which pipeline gives the most accurate answer

Building your own advanced retrieval benchmarks

🎯 Features
✔ Multi-Pipeline Retrieval

Runs 4 different RAG pipelines (A, B, C, D) with:

Different chunk sizes

Different retrieval depths

Optional reranking

Different LLM models

✔ Automated Pipeline Evaluation

A separate Groq model analyzes all four answers and chooses:

🏆 Best Pipeline: A/B/C/D

This turns your project into a real RAG research tool.

✔ Document Ingestion Pipeline

Upload a PDF → Split → Embed → Build FAISS vectorstore.

Each pipeline uses its own vectorstore for experimentation.

✔ Instant UI (Streamlit)

Includes:

Input question box

Beautiful 4-column comparison grid

Expandable retrieved documents viewer

Highlighted "Best Pipeline" badge

🧠 Architecture
PDF → Ingestion → FAISS Vectorstores → 4 Pipelines → Groq LLM → Judge Model → Streamlit UI

📁 Project Structure
rag_pipeline_optimizer/
│
├── backend/
│   ├── app/
│   │   ├── ingest.py
│   │   ├── pipelines.py
│   │   ├── retrievers.py
│   │   ├── config.py
│   │   ├── server.py
│   │   └── data/vectorstores/
│   ├── venv/
│   └── requirements.txt
│
├── frontend/
│   ├── app.py   ← Streamlit UI
│
└── README.md

🛠 Installation Guide
📌 Requirements
Component	Version
Python	3.10.x (recommended)
pip	latest
Groq SDK	0.12+
FAISS CPU	1.8.0
Streamlit	1.40+
📥 Step 1 — Clone the repo
git clone https://github.com/bindushreebade/rag-pipeline-optimizer.git
cd rag-pipeline-optimizer

📥 Step 2 — Create Backend Virtual Environment
cd backend
python -m venv venv
venv\Scripts\activate

📦 Step 3 — Install Backend Dependencies
pip install -r requirements.txt

requirements.txt (recommended content)
fastapi==0.115.0
uvicorn==0.30.0
groq==0.12.0
sentence-transformers==2.6.0
langchain==0.2.x
langchain-community==0.2.x
pydantic==2.12.0
pydantic-settings==2.2.1
faiss-cpu==1.8.0
python-dotenv==1.0.1
PyPDF2==3.0.1
requests
streamlit

🔑 Step 4 — Add Your .env File (IMPORTANT)

Location:

backend/.env


Content:

GROQ_API_KEY=your_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

📚 Step 5 — Ingest Your PDF
python app/ingest.py


This builds FAISS vectorstores for pipelines A/B/C/D.

▶ Step 6 — Run Backend API Server
uvicorn app.server:app --reload


API will be available at:

http://127.0.0.1:8000/docs

🖥 Step 7 — Run Streamlit Frontend

Open a second terminal:

cd frontend
streamlit run app.py


UI opens automatically at:

http://localhost:8501/

🚀 Usage Flow

Upload or ingest your PDF

Ask a question

All 4 pipelines run in parallel

LLaMA evaluates them

UI shows:

🏆 Best Pipeline: C


Expand retrieved docs to debug retrieval quality

This is a full RAG research system.

📸 Screenshots (Add later)
/assets/ui.png
/assets/pipelines.png
/assets/judge.png

🧩 Tech Stack

Groq LLaMA models

FAISS CPU / ChromaDB

LangChain

Sentence Transformers

FastAPI

Streamlit

❤️ Contributing

PRs welcome — especially:

Additional pipelines

Reranking techniques

Scoring visualizations

Long-context support

⭐ Final Notes

This project is extremely useful for:

Evaluating RAG strategies

Comparing chunk sizes

Testing LLM models

Debugging retrieval quality

Choosing best RAG settings for production