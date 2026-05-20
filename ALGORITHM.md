## Algorithm: Multi-Pipeline RAG Evaluation System with LLM-as-a-Judge

### Pseudocode

```
Algorithm: RAG_MultiPipeline_Evaluation_System

Input: PDF Document D, User Query Q
Output: Best Answer A_best, Pipeline Winner, All Pipeline Results

BEGIN

================================================================================
PHASE 1: DOCUMENT INGESTION AND VECTORSTORE INITIALIZATION
================================================================================

1. Document Upload & Text Extraction
   D ← Load PDF Document
   pages ← Extract pages from D
   raw_text ← Concatenate all pages
   cleaned_text ← Normalize(raw_text)    // Remove extra whitespace, special chars
   
2. Metadata Storage
   document_id ← Generate_UUID()
   document_name ← Get_Filename(D)
   Store_Metadata(document_id, document_name, upload_timestamp)

3. Parallel Vectorstore Creation (For 4 Pipelines)
   
   FOR pipeline IN {A, B, C, D} DO
   
      // Configure Pipeline Parameters
      config[pipeline].chunk_size ← Get_Chunk_Size(pipeline)
      config[pipeline].chunk_overlap ← Get_Overlap(pipeline)
      config[pipeline].retrieval_k ← Get_Retrieval_K(pipeline)
      
      // Chunk the document
      chunks[pipeline] ← Split_Document(
         cleaned_text,
         chunk_size = config[pipeline].chunk_size,
         overlap = config[pipeline].chunk_overlap
      )
      
      // Generate Embeddings (shared model across all pipelines)
      embeddings_model ← SentenceTransformer("all-MiniLM-L6-v2")
      
      FOR each chunk c_i in chunks[pipeline] DO
         embedding_vector ← embeddings_model.Encode(c_i)
         metadata[i] ← {
            document_id: document_id,
            chunk_index: i,
            page_number: Get_Page_Number(c_i),
            chunk_text: c_i
         }
      END FOR
      
      // Create Vectorstore (FAISS or Chroma)
      IF pipeline IN {A, B} THEN
         vectorstore[pipeline] ← FAISS.Index(embeddings)
      ELSE IF pipeline IN {C, D} THEN
         vectorstore[pipeline] ← Chroma.Index(embeddings, metadata)
      END IF
      
      Store_Vectorstore(vectorstore[pipeline], path="/vectorstores/{pipeline}/")
   
   END FOR
   
   RETURN "✓ Document ingested and 4 vectorstores created"

================================================================================
PHASE 2: QUERY PROCESSING AND PARALLEL RETRIEVAL
================================================================================

4. Query Embedding
   Q ← Receive User Query
   query_embedding ← embeddings_model.Encode(Q)
   
5. Parallel Semantic Retrieval (PARALLEL EXECUTION)
   
   FOR EACH pipeline IN {A, B, C, D} IN PARALLEL DO
   
      k ← config[pipeline].retrieval_k        // 3 or 5 documents
      
      // Similarity Search
      similarity_scores ← vectorstore[pipeline].Similarity_Search(
         query_embedding,
         k = k,
         threshold = 0.3
      )
      
      retrieved_docs[pipeline] ← Extract_Top_K(similarity_scores, k)
      
      // Store retrieval metadata
      retrieval_metrics[pipeline] ← {
         documents_retrieved: LEN(retrieved_docs[pipeline]),
         avg_similarity_score: MEAN(similarity_scores),
         retrieval_time_ms: Get_Elapsed_Time(),
         top_score: MAX(similarity_scores)
      }
      
      IF LEN(retrieved_docs[pipeline]) == 0 THEN
         retrieved_docs[pipeline] ← []   // Empty list for this pipeline
      END IF
   
   END FOR PARALLEL

================================================================================
PHASE 3: PARALLEL ANSWER GENERATION (4 PIPELINES)
================================================================================

6. Construct Retrieval Context and Generate Answers
   
   groq_client ← Initialize_Groq_Client(api_key)
   
   FOR EACH pipeline IN {A, B, C, D} IN PARALLEL DO
   
      // Build context from retrieved documents
      context ← Concatenate_Documents(retrieved_docs[pipeline])
      
      IF context IS EMPTY THEN
         context ← "[No relevant documents found]"
      END IF
      
      // Construct prompt for this pipeline
      prompt ← Format_Prompt(
         question = Q,
         context = context,
         instruction = "Answer ONLY using facts explicitly present in the context. 
                        If the answer is not found in the context, reply exactly: 
                        'Answer not found in the document.' Do NOT infer, assume, or generalize."
      )
      
      // Call Groq LLaMA 3.3-70B for answer generation
      response ← groq_client.Chat.Complete(
         model = "llama-3.3-70b-versatile",
         messages = [{"role": "user", "content": prompt}],
         temperature = 0.2,
         max_tokens = 500
      )
      
      answers[pipeline] ← response.choices[0].message.content
      
      // Store generation metadata
      generation_metrics[pipeline] ← {
         model: "llama-3.3-70b-versatile",
         prompt_tokens: LEN(Tokenize(prompt)),
         response_tokens: LEN(Tokenize(answers[pipeline])),
         generation_time_ms: Get_Elapsed_Time()
      }
      
      // Store retrieved documents for this pipeline
      pipeline_docs[pipeline] ← [doc.text FOR doc IN retrieved_docs[pipeline]]
   
   END FOR PARALLEL
   
   RETURN {
      answers[A], answers[B], answers[C], answers[D],
      pipeline_docs[A], pipeline_docs[B], pipeline_docs[C], pipeline_docs[D]
   }

================================================================================
PHASE 4: AUTOMATED EVALUATION (LLM-AS-A-JUDGE)
================================================================================

7. Judge Model Evaluation
   
   // Compile all pipeline results
   evaluation_input ← {
      original_question: Q,
      pipeline_a: {answer: answers[A], docs: pipeline_docs[A]},
      pipeline_b: {answer: answers[B], docs: pipeline_docs[B]},
      pipeline_c: {answer: answers[C], docs: pipeline_docs[C]},
      pipeline_d: {answer: answers[D], docs: pipeline_docs[D]}
   }
   
   // Create judge prompt
   judge_prompt ← Format_Judge_Prompt(
      question = Q,
      answer_a = answers[A],
      answer_b = answers[B],
      answer_c = answers[C],
      answer_d = answers[D],
      documents_a = pipeline_docs[A],
      documents_b = pipeline_docs[B],
      documents_c = pipeline_docs[C],
      documents_d = pipeline_docs[D],
      
      scoring_criteria = "
         1. ACCURACY: Does the answer match retrieved facts? (0-25 points)
         2. COMPLETENESS: Are all relevant points covered? (0-25 points)
         3. CLARITY: Is the answer easy to understand? (0-25 points)
         4. CONCISENESS: No unnecessary verbosity? (0-25 points)
         
         Output format: 'BEST: X' where X ∈ {A, B, C, D}
      "
   )
   
   // Call judge model
   judge_response ← groq_client.Chat.Complete(
      model = "llama-3.3-70b-versatile",
      messages = [{"role": "user", "content": judge_prompt}],
      temperature = 0.0,  // Deterministic evaluation
      max_tokens = 100
   )
   
   // Parse judge decision
   judge_output ← judge_response.choices[0].message.content
   best_pipeline ← Extract_Best_Pipeline(judge_output)  // Extract A/B/C/D
   judge_reasoning ← Extract_Reasoning(judge_output)
   
   validation_result ← {
      best_pipeline: best_pipeline,
      reasoning: judge_reasoning,
      evaluation_time_ms: Get_Elapsed_Time()
   }

================================================================================
PHASE 5: ANSWER VALIDATION AND SCORING
================================================================================

8. Validate Quality and Assign Confidence Score
   
   A_best ← answers[best_pipeline]
   retrieved_chunks_best ← pipeline_docs[best_pipeline]
   
   // Quality checks
   IF A_best CONTAINS "Answer not found in the document" THEN
      confidence_score ← 0.2
      quality_status ← "LOW"
   ELSE IF LEN(retrieved_chunks_best) == 0 THEN
      confidence_score ← 0.3
      quality_status ← "LOW_RETRIEVAL"
   ELSE IF validation_result.best_pipeline IS NULL THEN
      confidence_score ← 0.5
      quality_status ← "UNCERTAIN"
   ELSE
      confidence_score ← 0.85
      quality_status ← "HIGH"
   END IF
   
   // Extract source information
   source_pages ← [chunk.page_number FOR chunk IN retrieved_chunks_best]
   source_references ← UNIQUE(source_pages)

================================================================================
PHASE 6: DATABASE PERSISTENCE (MySQL)
================================================================================

9. Store Results in MySQL Database
   
   // Save individual chat interaction
   chat_record ← {
      question: Q,
      answer: A_best,
      pipeline: best_pipeline,
      document_name: document_name,
      retrieved_documents: JSON_Serialize(retrieved_chunks_best),
      timestamp: NOW(),
      confidence_score: confidence_score
   }
   
   chat_id ← INSERT INTO chat_history (chat_record)
   
   // Save multi-pipeline comparison
   comparison_record ← {
      question: Q,
      pipeline_a_answer: answers[A],
      pipeline_b_answer: answers[B],
      pipeline_c_answer: answers[C],
      pipeline_d_answer: answers[D],
      best_pipeline: best_pipeline,
      document_name: document_name,
      timestamp: NOW(),
      retrieval_metrics: JSON_Serialize(retrieval_metrics),
      generation_metrics: JSON_Serialize(generation_metrics),
      judge_reasoning: judge_reasoning
   }
   
   comparison_id ← INSERT INTO pipeline_comparison (comparison_record)
   
   // Update statistics
   UPDATE pipeline_wins SET count = count + 1 WHERE pipeline = best_pipeline

================================================================================
PHASE 7: RESPONSE CONSTRUCTION AND RETURN
================================================================================

10. Format Final Response
    
    final_response ← {
       // Best Answer
       answer: A_best,
       confidence_score: confidence_score,
       
       // Pipeline Comparison
       all_pipelines: {
          A: {answer: answers[A], docs_count: LEN(pipeline_docs[A])},
          B: {answer: answers[B], docs_count: LEN(pipeline_docs[B])},
          C: {answer: answers[C], docs_count: LEN(pipeline_docs[C])},
          D: {answer: answers[D], docs_count: LEN(pipeline_docs[D])}
       },
       
       // Winner Information
       best_pipeline: best_pipeline,
       judge_reasoning: judge_reasoning,
       
       // Source Information
       source_pages: source_references,
       retrieved_documents: retrieved_chunks_best,
       
       // Metadata
       document_name: document_name,
       retrieval_metrics: retrieval_metrics,
       generation_metrics: generation_metrics,
       total_time_ms: Get_Total_Elapsed_Time(),
       timestamp: NOW()
    }
    
    RETURN final_response

================================================================================
PHASE 8: ERROR HANDLING AND FALLBACK
================================================================================

11. Error Handling
    
    IF ERROR during retrieval THEN
       RETRY vectorstore search (max 2 attempts)
       IF still fails THEN
          RETURN error_response with "Retrieval service unavailable"
       END IF
    END IF
    
    IF ERROR during answer generation THEN
       FALLBACK to simpler prompt
       IF still fails THEN
          RETURN pre-cached similar question answer
       END IF
    END IF
    
    IF ERROR during judge evaluation THEN
       SELECT best_pipeline by similarity score (deterministic fallback)
       log_warning("Judge evaluation failed, using similarity-based selection")
    END IF

END ALGORITHM
```

---

### Complexity Analysis

**Time Complexity:**
- Document Ingestion: **O(n·m)** where n = number of pages, m = avg words per page
- Retrieval (Parallel): **O(log d)** per pipeline where d = number of chunks (FAISS complexity)
- Answer Generation (Parallel): **O(c·t)** where c = context length, t = token generation time
- Judge Evaluation: **O(1)** — single LLM call
- **Total: O(n·m + 4·log d + 4·c·t + 1)** ≈ **~12-15 seconds** end-to-end

**Space Complexity:**
- Vectorstores: **O(4·n·e)** where e = embedding dimension (384 for all-MiniLM-L6-v2)
- Chat History: **O(q·a)** where q = queries, a = avg answer length
- Total: **~500MB - 2GB** depending on document count

---

### Pipeline-Specific Configurations

```
Pipeline A (Balanced):
  - Chunk Size: 900 tokens
  - Retrieval K: 3 documents
  - Vectorstore: FAISS
  - Strategy: Best for general Q&A
  
Pipeline B (Deep Retrieval):
  - Chunk Size: 900 tokens
  - Retrieval K: 5 documents
  - Vectorstore: FAISS
  - Strategy: Best for complex topics
  
Pipeline C (Fine-grained):
  - Chunk Size: 512 tokens
  - Retrieval K: 5 documents
  - Vectorstore: Chroma
  - Strategy: Best for detail-oriented queries
  
Pipeline D (Long Context):
  - Chunk Size: 1500 tokens
  - Retrieval K: 3 documents
  - Vectorstore: Chroma
  - Strategy: Best for complex reasoning
```

---

### Key Differences from Original Algorithm

✅ **4 Pipelines instead of 1** — Simultaneous comparison for robustness
✅ **LLM-as-a-Judge** — Automated, objective evaluation
✅ **Groq instead of Ollama** — 100x faster inference
✅ **MySQL instead of SQLite** — Enterprise-grade persistence
✅ **Parallel Execution** — All 4 pipelines run simultaneously
✅ **Comprehensive Metrics** — Retrieval, generation, and evaluation metrics logged
✅ **Multi-Pipeline Comparison Storage** — Full audit trail of all 4 answers
✅ **Confidence Scoring** — Dynamically calculated based on retrieval quality
✅ **Error Resilience** — Fallback strategies for each phase

