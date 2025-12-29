import os
import json
import faiss
import numpy as np
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

# Assuming these exist in your project structure
from retrieval.embeddings import GeminiEmbeddingGenerator
from config.settings import settings

class ResumeIndexer:
    def __init__(self, embedder: Optional[GeminiEmbeddingGenerator] = None):
        self.embedder = embedder or GeminiEmbeddingGenerator()
        self.index_path = os.path.join(settings.INDEX_PATH, "resumes.index")
        self.metadata_path = os.path.join(settings.INDEX_PATH, "metadata.jsonl")
        
        # Ensure directories exist
        os.makedirs(settings.INDEX_PATH, exist_ok=True)

        self.dimension = 768 
        self.index = None
        self.metadata = [] 
        
        # --- Lexical Search Attributes ---
        self.bm25 = None
        self.tokenized_corpus = []

        self.load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25. Can be replaced with NLTK/Spacy."""
        # Lowercase and split by non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def _build_bm25(self):
        """Rebuilds the BM25 index from current metadata."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = [json.loads(line) for line in f]
                
                # --- Rebuild Lexical Index on Load ---
                print("Building lexical index...")
                self.tokenized_corpus = [self._tokenize(doc['normalized_text']) for doc in self.metadata]
                self._build_bm25()
                
                print(f"Loaded index with {self.index.ntotal} documents.")
            except Exception as e:
                print(f"Error loading index: {e}. Starting fresh.")
                self._init_fresh_index()
        else:
            self._init_fresh_index()

    def _init_fresh_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.tokenized_corpus = []
        self.bm25 = None

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + '\n')
        print("Index and metadata saved.")

    def index_resume(self, resume_id: str, normalized_text: str, metadata: Dict[str, Any]):
        # 1. Semantic Indexing (FAISS)
        embedding = self.embedder.generate(normalized_text)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.index.add(np.array([embedding], dtype=np.float32))

        # 2. Storage
        meta_record = {
            "id": resume_id,
            "normalized_text": normalized_text,
            **metadata
        }
        self.metadata.append(meta_record)

        # 3. Lexical Indexing (BM25)
        # Note: BM25Okapi is immutable, so we must rebuild or append to corpus.
        # For huge datasets, you wouldn't rebuild every time, but for resumes it's fine.
        tokens = self._tokenize(normalized_text)
        self.tokenized_corpus.append(tokens)
        self._build_bm25() 

        self.save_index()

    def search_semantic(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Standard FAISS search."""
        if not self.index or self.index.ntotal == 0:
            return []

        q_emb = self.embedder.generate(query_text)
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm

        distances, indices = self.index.search(np.array([q_emb], dtype=np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append({
                "id": self.metadata[idx]['id'],
                "doc_index": idx,
                "score": float(distances[0][i]),
                "metadata": self.metadata[idx]
            })
        return results

    def search_lexical(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Keyword-based BM25 search."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query_text)
        # Get top N scores
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top k scores
        top_n_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_n_indices:
            if scores[idx] == 0: continue # Filter irrelevant results
            results.append({
                "id": self.metadata[idx]['id'],
                "doc_index": idx,
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def search_hybrid(self, query_text: str, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF).
        Retrieves top k*2 from both methods to ensure overlap, then fuses.
        """
        # Fetch more candidates than final k to allow fusion to work
        candidate_k = k * 2
        
        semantic_results = self.search_semantic(query_text, k=candidate_k)
        lexical_results = self.search_lexical(query_text, k=candidate_k)

        # Dictionary to hold fused scores: {doc_index: rrf_score}
        doc_scores = {}

        # Helper to update RRF scores
        # Formula: Score = 1 / (rank + k_constant)
        def update_rrf(results):
            for rank, result in enumerate(results):
                doc_idx = result['doc_index']
                if doc_idx not in doc_scores:
                    doc_scores[doc_idx] = 0.0
                doc_scores[doc_idx] += 1.0 / (rank + rrf_k)

        update_rrf(semantic_results)
        update_rrf(lexical_results)

        # Sort by RRF score descending
        sorted_doc_indices = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        # Format final output
        final_results = []
        for doc_idx in sorted_doc_indices[:k]:
            final_results.append({
                "rrf_score": doc_scores[doc_idx],
                "metadata": self.metadata[doc_idx]
            })

        return final_results