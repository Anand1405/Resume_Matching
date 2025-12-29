import asyncio
import hashlib
import json
from typing import List, Dict, Any
from retrieval.indexer import ResumeIndexer
from agents.resume_extractor import ResumeExtractionAgent
from agents.matcher import MatchingAgent
from config.settings import settings
import traceback

class ResumeOrchestrator:
    def __init__(self):
        self.extractor = ResumeExtractionAgent()
        self.matcher = MatchingAgent()
        # Singleton pattern for Indexer to keep index in memory if possible
        self.indexer = ResumeIndexer()

    def _get_file_hash(self, file_path: str) -> str:
        """Simple hash to avoid re-processing identical files."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    async def ingest_resumes(self, resume_paths: List[str]):
        """
        Extracts and Indexes resumes.
        """
        print(f"Ingesting {len(resume_paths)} resumes...")
        for path in resume_paths:
            try:
                file_id = self._get_file_hash(path)

                # Check if already indexed (naive check via ID in metadata)
                existing = [m for m in self.indexer.metadata if m.get('id') == file_id]
                if existing:
                    print(f"Skipping {path} (already indexed)")
                    continue

                print(f"Extracting {path}...")
                # Run Extraction Agent
                data = await self.extractor.extract(path)
                # Convert Pydantic model to dict for storage
                metadata = data.to_dict()['message']['content'][0]['toolUse']['input']
                
                normalized_text = metadata.pop('normalized_text', '')
                print("Normalized Text Sample:", normalized_text[:200])
                
                # Index
                # The indexer now handles both Vector and BM25 indexing internally
                self.indexer.index_resume(file_id, normalized_text, metadata)
                print(f"Indexed {path}")

            except Exception as e:
                print(f"Error processing {path}: {e}")
                traceback.print_exc()

    async def process_resumes(self, job_description: str, resume_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Full Pipeline: Ingest (if needed) -> Retrieve (Hybrid) -> Re-rank.
        """
        # 1. Ingest new files if provided
        if resume_paths:
            await self.ingest_resumes(resume_paths)

        # 2. Retrieve Candidates (Hybrid Search)
        print("Retrieving candidates using Hybrid Search (RRF)...")
        
        # Use search_hybrid instead of search
        # Note: k is doubled here to allow re-ranker to filter down later
        candidates = self.indexer.search_hybrid(job_description, k=settings.TOP_N_CANDIDATES * 2)
        
        if not candidates:
            return []

        # 3. Re-rank / Match
        print(f"Re-ranking {len(candidates)} candidates...")
        results = []
        for cand in candidates:
            try:
                meta = cand['metadata']
                # Create a concise profile for the matcher
                profile = meta.copy()
                del profile['id'] 

                analysis = await self.matcher.evaluate(job_description, profile)

                # Flatten result
                res = analysis.to_dict()['message']['content'][0]['toolUse']['input']
                
                res['file_id'] = meta.get('id')
                
                # Handle RRF Score from hybrid search (fallback to 'score' if standard search used)
                res['retrieval_score'] = cand.get('rrf_score', cand.get('score', 0))

                # Add legacy fields for UI compatibility
                res['score'] = res['final_score']
                res['name'] = res['candidate_name']

                results.append(res)

            except Exception as e:
                print(f"Error matching candidate: {e}")
                traceback.print_exc()

        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results

    def get_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "total_candidates": 0,
                "average_final_score": 0.0,
                "average_experience_score": 0.0,
                "average_skills_score": 0.0,
                "average_education_score": 0.0,
                "average_projects_score": 0.0,
                "top_candidate": None
            }
        
        total = len(results)
        avg_final = sum(r.get("final_score", 0) for r in results) / total
        avg_exp = sum(r.get("experience_score", 0) for r in results) / total
        avg_skills = sum(r.get("skills_score", 0) for r in results) / total
        avg_edu = sum(r.get("education_score", 0) for r in results) / total
        avg_proj = sum(r.get("projects_score", 0) for r in results) / total
        
        return {
            "total_candidates": total,
            "average_final_score": round(avg_final, 3),
            "average_experience_score": round(avg_exp, 3),
            "average_skills_score": round(avg_skills, 3),
            "average_education_score": round(avg_edu, 3),
            "average_projects_score": round(avg_proj, 3),
            "top_candidate": results[0]['candidate_name'],
            "top_candidate_score": results[0]['final_score']
        }