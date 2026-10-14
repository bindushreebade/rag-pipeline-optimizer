# RAG Pipeline Optimizer

A research-paper question-answering system with four retrieval pipelines
and a reproducible evaluation harness.

> Upload research PDFs, ask questions, compare retrieval strategies side
> by side, and evaluate retrieval quality against a hand-labelled
> benchmark.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

## Screenshots

Replace these placeholders with your three screenshots.

  -----------------------------------------------------------------------------------------------------------------------------------------
  Screenshot 1                                  Screenshot 2                                  Screenshot 3
  --------------------------------------------- --------------------------------------------- ---------------------------------------------
  `![Screenshot 1](path/to/screenshot-1.png)`   `![Screenshot 2](D:\Programs\GenAI_Chat-Bot\image.png)`   `![Screenshot 3](path/to/screenshot-3.png)`

  -----------------------------------------------------------------------------------------------------------------------------------------

## Overview

The system answers questions about uploaded PDFs, compares four
retrieval strategies, and evaluates them using **MRR, Recall, Hit rate,
nDCG, and Precision**.

-   **Benchmark:** 57 questions from one research paper.
-   **Ground truth:** Page labels checked against the PDF.
-   **Evaluation:** Top 5 retrieved chunks.
-   **Answering pipeline:** Pipeline D.

## Results

### Original baseline vs. final system

  ----------------------------------------------------------------------------------
  Setup                   MRR@5     Recall@5        Hit@5       nDCG@5   Precision@5
  ---------------- ------------ ------------ ------------ ------------ -------------
  Baseline: MiniLM        0.667        0.720        0.860        0.423         0.386
  embeddings,                                                          
  500-character                                                        
  chunks, dense                                                        
  search                                                               

  Final: BGE-base,    **0.738**    **0.863**    **0.965**    **0.515**     **0.446**
  1000-character                                                       
  chunks, hybrid +                                                     
  rerank                                                               
  ----------------------------------------------------------------------------------

Recall@5 rose by 20% and Hit@5 by 12% (6 more questions out of 57 now
have a correct page in the top 5). MRR improved by 11%, a smaller change
that is within the noise of a 57-question set.

### Four pipelines (final configuration)

  ---------------------------------------------------------------------------------------------
  Pipeline   Retrieval       MRR@5   Recall@5     Hit@5    nDCG@5   Precision@5   Seconds/query
  ---------- ----------- --------- ---------- --------- --------- ------------- ---------------
  A          Dense           0.666      0.798     0.930     0.402         0.337            0.05

  B          Dense +         0.686      0.790     0.930     0.454         0.396            0.03
             BM25 (RRF)                                                         

  C          Dense +         0.743      0.863     0.965     0.517         0.446            0.86
             rerank                                                             

  D          Dense +         0.738      0.863     0.965     0.515         0.446            1.02
             BM25 +                                                             
             rerank                                                             
  ---------------------------------------------------------------------------------------------

Latency was measured in Google Colab and depends on hardware.

### What drove the improvement?

  -----------------------------------------------------------------------
  Change                              Observed effect
  ----------------------------------- -----------------------------------
  BGE embeddings + 1000-character     Recall@5 0.72 → 0.80; Hit@5 0.86 →
  chunks (vs. MiniLM, 500)            0.93; MRR unchanged

  Cross-encoder reranking             MRR 0.67 → 0.74; Recall 0.80 →
                                      0.86; largest gains in Precision
                                      and nDCG

  Adding BM25 (hybrid)                Small gain on its own (B vs. A); no
                                      measurable gain once reranking is
                                      used (C vs. D)
  -----------------------------------------------------------------------

### Findings

-   **Reranking is the largest single improvement.** It lifts
    Precision@5 and nDCG@5 the most, meaning more of the top 5 chunks
    are relevant.
-   **Hybrid retrieval did not clearly beat dense retrieval on this
    paper.** C and D score similarly, so hybrid retrieval is described
    as input to the reranker, not as the source of the gain.
-   **A larger reranker (`bge-reranker-large`) performed worse.** It
    raised MRR slightly but lost about 0.09 Recall@5 and was about 3×
    slower; the default is `bge-reranker-base`.
-   **Chunk size and candidate-pool size mattered little.** Chunks of
    750 and 1000 characters were close, 1500 was not better, and
    candidate K of 20 vs. 30 made almost no difference.
-   **Label auditing mattered more than tuning.** The first benchmark
    labels were one page off for many questions. After checking labels
    against the PDF, scores changed far more than any model change did.

## Limitations

-   One paper and 57 questions: treat the numbers as indicative, not as
    a general claim.
-   The same questions were used to develop and evaluate the system, so
    there is no separate hold-out set.
-   Labels are page-level. A chunk counts as relevant if it overlaps a
    labelled page, which is lenient.
-   Retrieval quality is measured; answer correctness and faithfulness
    are not part of the benchmark.

## Retrieval pipelines

  -----------------------------------------------------------------------
  ID                      Retrieval               Reranking
  ----------------------- ----------------------- -----------------------
  A                       Dense (bi-encoder       No
                          cosine similarity)      

  B                       Dense + BM25, merged    No
                          with Reciprocal Rank    
                          Fusion (RRF)            

  C                       Dense                   Cross-encoder

  D                       Dense + BM25, merged    Cross-encoder
                          with Reciprocal Rank    
                          Fusion (RRF)            
  -----------------------------------------------------------------------

All four pipelines read the same index, so metric differences come from
the retrieval strategy alone. The app answers questions with **Pipeline
D**.

## Default configuration

Defaults are defined in `backend/app/config.py`.

  Setting                      Default
  ---------------------------- ---------------------------------------------
  Embedding model              `BAAI/bge-base-en-v1.5`
  Reranker                     `BAAI/bge-reranker-base`
  Answer model                 `llama-3.3-70b-versatile` (Groq)
  Judge model                  `llama-3.1-8b-instant` (Groq, configurable)
  Chunk size / overlap         1000 / 200 characters
  Candidates per retriever     30
  Chunks used for the answer   5

## How evaluation works

### Ground truth

`backend/app/data/ground_truth.json` lists each question and the page or
pages containing its answer. Page-level labels remain valid when chunk
size changes. Each question can also carry an `evidence` list of phrases
to make relevance stricter.

Example:

``` json
{
  "id": "Q12",
  "question": "What construction does Theorem 8 use ...?",
  "source_pages": [7]
}
```

### Retrieval and scoring

Each pipeline returns its top 5 chunks per question. A chunk is relevant
if its page range overlaps a labelled page.

  -----------------------------------------------------------------------
  Metric                              Meaning
  ----------------------------------- -----------------------------------
  MRR@k                               Average of `1 / rank` of the first
                                      relevant chunk

  Recall@k                            Share of labelled pages covered by
                                      the top k

  Hit@k                               Share of questions with at least
                                      one relevant chunk in the top k

  Precision@k                         Share of the top k chunks that are
                                      relevant

  nDCG@k                              Rank-aware gain with binary
                                      relevance
  -----------------------------------------------------------------------

Questions are split by ID into **dev (odd)** and **test (even)** for
tuning with `--split`.

## Architecture

``` text
Streamlit UI (frontend/app.py)
        │  HTTP
        ▼
FastAPI (backend/app/server.py)
   ├─ /upload  ─────────► ingest.py: PDF → clean → chunk → embed
   ├─ /ask, /compare ───► pipelines.py: retrieve → (rerank) → Groq answer
   ├─ /metrics/retrieval ► metrics.py: evaluate pipelines against ground truth
   └─ /history/* ───────► database.py: MySQL chat history
                                │
              retrievers.py: Chroma (dense) + BM25 (sparse)
```

### Pipeline D retrieval flow

``` text
question ─► dense top-30 ─┐
                          ├─► Reciprocal Rank Fusion ─► cross-encoder rerank ─► top-5 ─► LLM answer
question ─► BM25  top-30 ─┘
```

Chunking runs over the whole document as one text, so a definition or
theorem is not cut at a page break. Every chunk records its start and
end page.

## Project structure

``` text
backend/
├── app/
│   ├── config.py                         # settings (reads .env)
│   ├── ingest.py                         # PDF loading, cleaning, chunking, indexing
│   ├── retrievers.py                     # embeddings, Chroma store, BM25 index
│   ├── pipelines.py                      # pipelines A–D, answer generation, judge
│   ├── metrics.py                        # evaluation metrics and splits
│   ├── sweep.py                          # chunk-size / candidate-K sweep
│   ├── rebuild_comparable_collections.py # builds the benchmark index
│   ├── server.py                         # FastAPI endpoints
│   ├── database.py                       # MySQL chat history
│   ├── data/ground_truth.json            # labelled benchmark
│   └── uploads/                          # PDFs
├── requirements.txt
└── .env
frontend/
├── app.py                                # Streamlit UI
└── pages/
```

## Setup

### Requirements

-   Python 3.10+
-   MySQL (for chat history)
-   A Groq API key

### 1. Clone and enter the backend directory

``` bash
git clone <your-repo-url>
cd rag-pipeline-optimizer/backend
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate
# Windows:
.venv\\Scripts\\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `backend/.env`:

``` dotenv
GROQ_API_KEY=your_key
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rag_optimizer
```

### 5. Run the backend and UI

From `backend`:

``` bash
uvicorn app.server:app --reload
```

Backend URL: `http://localhost:8000`

In another terminal:

``` bash
cd ../frontend
streamlit run app.py
```

## Reproduce the benchmark

Run from `backend`:

``` bash
python -m app.rebuild_comparable_collections
python -m app.metrics --split all
python -m app.sweep --chunk-sizes 750 1000 1500 --candidate-ks 20 30
```

Results are saved under `backend/app/data/`.

### Reproduce the baseline row

``` bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  python -m app.rebuild_comparable_collections --chunk-size 500 --overlap 100

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  python -m app.metrics --split all
```

On PowerShell, set the variable first:

``` powershell
$env:EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

Rebuild with the default settings afterwards.

> The sweep re-embeds the paper for every chunk size and reranks many
> candidates, so it is slow on a CPU. A free GPU runtime on Google Colab
> runs it in minutes.

If you change `EMBEDDING_MODEL`, delete `backend/app/data/vectorstores`
and rebuild the index. Vectors from different models are not compatible.

## API

  ----------------------------------------------------------------------------
  Method                  Endpoint                     Purpose
  ----------------------- ---------------------------- -----------------------
  POST                    `/upload`                    Upload and index PDFs

  GET                     `/documents`                 List uploaded PDFs

  POST                    `/ask`                       Answer a question with
                                                       Pipeline D

  POST                    `/compare`                   Run all four pipelines
                                                       and pick the best
                                                       answer with an LLM
                                                       judge

  GET                     `/metrics/retrieval`         Evaluate pipelines
                                                       (`retrieve_k`,
                                                       `final_k`, `split`)

  POST                    `/history/save`              Save a chat entry

  POST                    `/history/save-comparison`   Save a four-pipeline
                                                       comparison

  GET                     `/history`,                  Read history and
                          `/history/comparisons`,      statistics
                          `/history/stats`             

  DELETE                  `/history/{id}`              Delete a chat entry

  GET                     `/health`                    Service status
  ----------------------------------------------------------------------------

## Tech stack

FastAPI, Streamlit, ChromaDB, `rank-bm25`, `sentence-transformers` (BGE
embedder and reranker), LangChain text splitters, PyPDF, Groq (Llama 3.3
70B), and MySQL.

## Roadmap

-   Evaluate answer faithfulness and correctness, not only retrieval.
-   Add a second paper to the benchmark and a true hold-out set.
-   Add section-aware chunking with headings added to each chunk.
-   Run the four pipelines in parallel in the comparison endpoint.
