import sys
import os
import json
import asyncio
import pandas as pd
from typing import List, Dict, Any, Set

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import ResumeOrchestrator

# --- Helpers ---

def load_labels(labels_path: str) -> Dict[str, Any]:
    with open(labels_path, 'r') as f:
        return json.load(f)

def load_jd(jd_path: str) -> str:
    with open(jd_path, 'r', encoding='utf-8') as f:
        return f.read()

def normalize_string(s: str) -> str:
    """Normalizes strings for fuzzy matching (e.g., 'Alice Chen' -> 'alicechen')."""
    if not s: return ""
    return str(s).lower().replace(" ", "").replace("_", "").replace(".txt", "")

def parse_score_range(range_str: str) -> tuple[float, float]:
    """Parses '0.85-1.0' into (0.85, 1.0)."""
    try:
        parts = range_str.split('-')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

def match_candidate_to_label(candidate_name: str, labels: Dict[str, Any]) -> tuple[str, Dict]:
    """Finds the label entry that matches the candidate name."""
    clean_candidate = normalize_string(candidate_name)
    if len(clean_candidate) < 3: 
        return None, None

    for label_key, label_data in labels.items():
        if clean_candidate in normalize_string(label_key):
            return label_key, label_data
    return None, None

# --- Retrieval Metrics ---

def calculate_retrieval_metrics(results: List[Dict], labels: Dict[str, Any], k_values: List[int]) -> pd.DataFrame:
    # Define "Relevant" as anyone NOT labeled "Poor Match"
    relevant_files = {
        fname for fname, data in labels.items() 
        if data['category'] != 'Poor Match'
    }
    
    metrics = []
    
    for k in k_values:
        top_k = results[:k]
        retrieved_relevant = 0
        
        for res in top_k:
            name = res.get('metadata', {}).get('name', 'Unknown')
            # Check if this name matches a relevant filename
            _, label_data = match_candidate_to_label(name, labels)
            if label_data and label_data['category'] != 'Poor Match':
                retrieved_relevant += 1
        
        precision = retrieved_relevant / k if k > 0 else 0
        recall = retrieved_relevant / len(relevant_files) if len(relevant_files) > 0 else 0
        
        metrics.append({
            "k": k,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3)
        })
        
    return pd.DataFrame(metrics)

# --- Matching Metrics ---

def calculate_matching_metrics(results: List[Dict], labels: Dict[str, Any]) -> pd.DataFrame:
    data = []
    correct_count = 0
    total_count = 0

    for res in results:
        # Get Candidate Name
        name = res.get('candidate_name', res.get('name', 'Unknown'))
        
        # Get Score (Handle 0-100 vs 0-1 scaling)
        raw_score = float(res.get('final_score', res.get('score', 0)))
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score
        
        # Find Ground Truth
        label_key, label_data = match_candidate_to_label(name, labels)
        
        if label_data:
            target_range = label_data.get('score_range', '0-0')
            min_s, max_s = parse_score_range(target_range)
            
            # Check Accuracy (with 0.05 tolerance)
            is_accurate = (min_s <= score <= max_s) or \
                          (abs(score - min_s) < 0.05) or \
                          (abs(score - max_s) < 0.05)
            
            if is_accurate:
                correct_count += 1
            total_count += 1
            
            data.append({
                "Candidate": name,
                "Score": round(score, 3),
                "Target Range": target_range,
                "Match": "✅" if is_accurate else "❌"
            })
        else:
            print(f"Warning: Could not find label for {name}")

    if total_count > 0:
        print(f"\nMatching Accuracy: {round((correct_count/total_count)*100, 2)}%")
        
    return pd.DataFrame(data)

# --- Main Pipeline ---

async def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resumes_dir = os.path.join(base_dir, "dataset", "resumes")
    jd_path = os.path.join(base_dir, "dataset", "job_description.txt")
    labels_path = os.path.join(base_dir, "dataset", "labels.json")

    # Load Data
    if not os.path.exists(resumes_dir):
        print("Dataset not found.")
        return

    jd_text = load_jd(jd_path)
    labels = load_labels(labels_path)
    resume_files = [os.path.join(resumes_dir, f) for f in os.listdir(resumes_dir) if f.endswith(".txt")]

    orchestrator = ResumeOrchestrator()

    # 1. Ingest (Async)
    print("\n--- Phase 1: Ingestion ---")
    await orchestrator.ingest_resumes(resume_files)

    # 2. Evaluate Retrieval
    print("\n--- Phase 2: Retrieval Evaluation ---")
    # Get top 15 candidates purely from search
    retrieval_results = orchestrator.indexer.search_hybrid(jd_text, k=15)
    
    retrieval_df = calculate_retrieval_metrics(retrieval_results, labels, k_values=[3, 5, 10, 15])
    print(retrieval_df.to_string(index=False))

    # 3. Evaluate Matching (Full Pipeline)
    print("\n--- Phase 3: Matching Evaluation ---")
    # This runs retrieval again internally + the Matching Agent
    pipeline_results = await orchestrator.process_resumes(jd_text)
    
    matching_df = calculate_matching_metrics(pipeline_results, labels)
    if not matching_df.empty:
        print(matching_df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())