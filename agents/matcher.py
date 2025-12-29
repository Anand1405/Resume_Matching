from typing import List, Optional
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.gemini import GeminiModel
from config.settings import settings
from tools.scoring_tool import calculate_score

# --- Data Models ---
class MatchAnalysis(BaseModel):
    candidate_name: str
    final_score: float
    breakdown: str
    experience_score: float
    skills_score: float
    education_score: float
    projects_score: float
    strengths: List[str]
    gaps: List[str]
    reasoning: str

# --- Agent ---
class MatchingAgent:
    def __init__(self):
        self.model = GeminiModel(
            client_args={"api_key": settings.GOOGLE_API_KEY},
            model_id=settings.RERANK_MODEL,
            params={"temperature": 0.1, "max_output_tokens": 4096},
        )

        FEW_SHOT_EXAMPLES = """
FEW-SHOT EXAMPLE 1 (Strong match)

JOB DESCRIPTION:
Role: Backend Python Engineer
Requirements (Required):
- 4+ years Python
- FastAPI
- PostgreSQL
- Redis
- AWS (EC2, S3)
- Docker
Nice-to-have:
- Kafka

CANDIDATE PROFILE:
{
  "name": "Aarav Mehta",
  "summary": "5 years building APIs in Python/FastAPI; AWS deployments; Dockerized microservices.",
  "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "AWS", "Docker"],
  "education": "B.Tech CS",
  "projects": "Built multi-tenant SaaS with FastAPI, Redis caching, Postgres; deployed on AWS."
}

Sub-scores (0.00–1.00) computed by the assistant (before tool call):
- experience_score = 0.90
- skills_score     = 0.92
- education_score  = 0.85
- projects_score   = 0.88

MANDATORY TOOL CALL (exactly once; NOTE the required 'scores' object):
calculate_score(
  scores={
    "experience_score": 0.90,
    "skills_score": 0.92,
    "education_score": 0.85,
    "projects_score": 0.88
  }
)

Tool output (example):
{
  "final_score": 0.900,
  "breakdown": "Final (0.9) = Exp (0.9 * 0.3) + Skills (0.92 * 0.4) + Edu (0.85 * 0.1) + Proj (0.88 * 0.2)",
  "experience_score": 0.9,
  "skills_score": 0.92,
  "education_score": 0.85,
  "projects_score": 0.88
}

Final JSON output (must copy tool output fields):
{
  "candidate_name": "Aarav Mehta",
  "final_score": 0.9,
  "breakdown": "Final (0.9) = Exp (0.9 * 0.3) + Skills (0.92 * 0.4) + Edu (0.85 * 0.1) + Proj (0.88 * 0.2)",
  "experience_score": 0.9,
  "skills_score": 0.92,
  "education_score": 0.85,
  "projects_score": 0.88,
  "strengths": ["Meets/exceeds relevant experience", "Covers all required backend skills", "Deployment experience on AWS with Docker"],
  "gaps": ["Kafka is only a nice-to-have; not demonstrated"],
  "reasoning": "Candidate exceeds tenure and matches nearly all required skills with strong project evidence; education aligns well."
}

FEW-SHOT EXAMPLE 2 (Weak match)

JOB DESCRIPTION:
Role: Data Scientist (NLP)
Requirements (Required):
- 3+ years NLP
- Transformers
- PyTorch
- Experiment tracking (MLflow/W&B)
- Strong statistics

CANDIDATE PROFILE:
{
  "name": "Neha Singh",
  "summary": "2 years as BI analyst; some Python; dashboards; basic ML course projects.",
  "skills": ["SQL", "Tableau", "Python (basic)", "Pandas"],
  "education": "MBA",
  "projects": "Sentiment analysis notebook using scikit-learn."
}

Sub-scores (0.00–1.00) computed by the assistant (before tool call):
- experience_score = 0.25
- skills_score     = 0.20
- education_score  = 0.40
- projects_score   = 0.30

MANDATORY TOOL CALL (exactly once; with 'scores' object):
calculate_score(
  scores={
    "experience_score": 0.25,
    "skills_score": 0.20,
    "education_score": 0.40,
    "projects_score": 0.30
  }
)

Final JSON output (copy tool output fields):
{
  "candidate_name": "Neha Singh",
  "final_score": 0.26,
  "breakdown": "Final (0.26) = Exp (0.25 * 0.3) + Skills (0.2 * 0.4) + Edu (0.4 * 0.1) + Proj (0.3 * 0.2)",
  "experience_score": 0.25,
  "skills_score": 0.2,
  "education_score": 0.4,
  "projects_score": 0.3,
  "strengths": ["Some Python exposure", "Has completed a basic sentiment project"],
  "gaps": ["Missing transformers and PyTorch", "Insufficient NLP experience", "No experiment tracking evidence", "Stats strength not evidenced"],
  "reasoning": "Candidate background is BI-oriented and does not meet core NLP/transformer requirements or tenure."
}
"""

        self.agent = Agent(
            model=self.model,
            tools=[calculate_score],
            system_prompt=f"""
You are an expert HR Recruiter. Your task is to evaluate a candidate against a Job Description (JD).

GOAL
- Produce a MatchAnalysis that is faithful to the JD and candidate evidence.
- You must compute the four sub-scores yourself first, then call the scoring tool once to compute the final score.

SCORING RUBRIC (each sub-score is a float from 0.00 to 1.00)
1) experience_score (Relevance & Tenure)
   - Consider required years and how directly past roles match the JD scope.

2) skills_score (Match of REQUIRED skills)
   - Identify REQUIRED vs NICE-TO-HAVE from the JD.
   - Score based on required skill coverage + evidence of use (work/projects), not just keyword listing.

3) education_score (Degree / credential alignment)
   - Evaluate how well the candidate's formal education matches JD requirements.

4) projects_score (Complexity & Relevance)
   - Evaluate relevance, scope, and whether projects demonstrate real-world/production-like work.

PROCESS (FOLLOW EXACTLY)
A) Extract JD requirements and separate them into:
   - Required
   - Nice-to-have
B) Using the candidate profile, compute these four sub-scores yourself:
   - experience_score
   - skills_score
   - education_score
   - projects_score
C) MANDATORY: Call the `calculate_score` tool EXACTLY ONCE using the tool signature:
   calculate_score(scores={{
     "experience_score": <float 0..1>,
     "skills_score": <float 0..1>,
     "education_score": <float 0..1>,
     "projects_score": <float 0..1>
   }})

HARD RULES
- Do NOT compute final_score yourself.
- Do NOT call `calculate_score` more than once.
- After the single tool call, do not call any additional tools.
- Keep `reasoning` short and high-signal.
- `strengths` and `gaps` must be concise lists of strings grounded in evidence.

{FEW_SHOT_EXAMPLES}
""",
        )

    async def evaluate(self, jd: str, candidate_data: dict) -> MatchAnalysis:
        prompt = f"""
JOB DESCRIPTION:
{jd}

CANDIDATE PROFILE:
{candidate_data}
"""
        result = self.agent(structured_output_model=MatchAnalysis, prompt=prompt)
        print("MatchingAgent.evaluate result:", result.structured_output)
        
        return result
