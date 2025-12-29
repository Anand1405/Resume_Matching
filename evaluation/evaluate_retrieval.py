import sys
import os
import json
import asyncio
import pandas as pd
from typing import List, Dict, Set

# Fix path to allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import ResumeOrchestrator

# Helper to load ground truth
def load_labels(labels_path: str) -> Dict[str, Dict]:
    with open(labels_path, 'r') as f:
        return json.load(f)

def load_jd(jd_path: str) -> str:
    with open(jd_path, 'r', encoding='utf-8') as f:
        return f.read()

def normalize_string(s: str) -> str:
    """Removes spaces, underscores, and lowers case for fuzzy matching."""
    if not s: return ""
    return str(s).lower().replace(" ", "").replace("_", "").replace(".txt", "")

def get_retrieval_metrics(retrieved_docs: List[Dict], relevant_files: Set[str], k: int):
    """
    Calculates Precision@K and Recall@K using Name-to-Filename matching.
    """
    # Slice to top K
    top_k = retrieved_docs[:k]
    
    retrieved_relevant = 0
    
    for doc in top_k:
        # 1. Get the Extracted Name from Metadata
        candidate_name = doc.get('metadata', {}).get('name', 'Unknown')
        
        # 2. Normalize it (e.g., "Alice Chen" -> "alicechen")
        clean_candidate = normalize_string(candidate_name)
        
        # 3. Check if this name exists inside any of the Relevant Filenames
        # (e.g., is "alicechen" inside "resume_1_alice_chen.txt"?)
        is_match = False
        if len(clean_candidate) > 3: # Avoid matching empty or short strings
            for r_file in relevant_files:
                clean_r_file = normalize_string(r_file)
                if clean_candidate in clean_r_file:
                    is_match = True
                    break
        
        if is_match:
            retrieved_relevant += 1
            
    # Precision: Portion of top K that are relevant
    precision = retrieved_relevant / k if k > 0 else 0
    
    # Recall: Portion of ALL relevant docs that appear in top K
    total_relevant = len(relevant_files)
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
    
    return {
        "k": k,
        "retrieved_relevant": retrieved_relevant,
        "precision": round(precision, 3),
        "recall": round(recall, 3)
    }

async def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resumes_dir = os.path.join(base_dir, "dataset", "resumes")
    jd_path = os.path.join(base_dir, "dataset", "job_description.txt")
    labels_path = os.path.join(base_dir, "dataset", "labels.json")

    # 1. Setup
    if not os.path.exists(resumes_dir):
        print(f"Error: Dataset not found at {resumes_dir}")
        return

    jd_text = load_jd(jd_path)
    labels = load_labels(labels_path)
    
    # Define "Relevant" documents.
    # We consider anything NOT "Poor Match" as relevant for retrieval purposes.
    relevant_files = set()
    for fname, data in labels.items():
        if data['category'] in ["Excellent Match", "Good Match", "Partial Match"]:
            relevant_files.add(fname)
            
    print(f"Total Documents: {len(labels)}")
    print(f"Total Relevant Documents (Excl/Good/Partial): {len(relevant_files)}")

    # 2. Ingest Resumes (Async Fix)
    orchestrator = ResumeOrchestrator()
    resume_files = [
        os.path.join(resumes_dir, f) 
        for f in os.listdir(resumes_dir) 
        if f.endswith(".txt") or f.endswith(".pdf")
    ]
    
    # FIX: Added 'await' here
    await orchestrator.ingest_resumes(resume_files)
    
    # 3. Run Hybrid Search
    print("Running Hybrid Search...")
    max_k = 15
    # Assuming search_hybrid is synchronous (faiss/bm25 usually are). 
    # If orchestrator.indexer.search_hybrid is async, add await. 
    # Based on standard implementations it's usually sync.
    results = orchestrator.indexer.search_hybrid(jd_text, k=max_k)
    
    # 4. Calculate Metrics for different K
    metrics_data = []
    k_values = [3, 5, 10, 15]
    
    for k in k_values:
        if k > len(results):
            continue
        m = get_retrieval_metrics(results, relevant_files, k)
        metrics_data.append(m)
        
    # 5. Display Results
    print("\n--- Retrieval Performance ---")
    df = pd.DataFrame(metrics_data)
    print(df.to_string(index=False))
    
    print("\n--- Detailed Rank Order ---")
    for i, res in enumerate(results):
        candidate_name = res.get('metadata', {}).get('name', 'Unknown')
        score = res.get('rrf_score', res.get('score', 0))

        print(f"{i+1}. {candidate_name} (Score: {score:.3f})")

if __name__ == "__main__":
    asyncio.run(main())