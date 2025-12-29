from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.gemini import GeminiModel
from config.settings import settings
from tools.file_ingestion import read_resume_file

# --- Data Models ---
class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: str
    years_duration: int = Field(..., description="Duration of this role in years")
    description: str = Field(..., description="Bullet points or summary of work")
    skills_used: List[str] = Field(default_factory=list)

class Education(BaseModel):
    institution: str
    degree: str
    year: str

class SkillDetail(BaseModel):
    skill_name: str
    years_experience: Optional[int] = None
    last_used: Optional[str] = None
    context: Optional[str] = Field(None, description="Where/how was this skill used?")

class ResumeData(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: str = Field(..., description="Short professional summary")

    # Mandatory Fields per Spec
    total_exp_years: int = Field(..., description="Total work experience in years (estimated if needed)")
    relevant_exp: Dict[str, int] = Field(..., description="Breakdown of experience years by domain (e.g., {'Backend': 48, 'ML': 24})")

    work_history: List[WorkExperience]
    education: List[Education]

    certs: List[str] = Field(default_factory=list, description="List of certifications")

    skills_primary: List[str]
    skills_recent: List[str] = Field(..., description="Skills used in the most recent roles")
    skills_detailed: List[SkillDetail] = Field(default_factory=list)

    normalized_text: str = Field(..., description="A dense, keyword-rich summary string suitable for embedding and indexing.")

# --- Agent ---
class ResumeExtractionAgent:
    def __init__(self):
        # Configure Gemini 3 Flash
        self.model = GeminiModel(
            client_args={
                "api_key": settings.GOOGLE_API_KEY,
            },
            model_id=settings.EXTRACTION_MODEL,
            params={
                "temperature": 0.1,
                "max_output_tokens": 4096
            }
        )

        self.agent = Agent(
            model=self.model,
            tools=[read_resume_file],
            system_prompt="""You are an expert Resume Parser. Your job is to extract structured data from resume files into a strict JSON schema.

**MANDATORY EXTRACTION RULES:**
1.  **Total Experience**: Calculate strictly in YEARS.
2.  **Relevant Experience Breakdown**: Categorize experience by domain (e.g., 'Python', 'Management', 'Cloud') and estimate years for each.
3.  **Skills**:
    *   `skills_primary`: The core competency set.
    *   `skills_recent`: Only skills explicitly used in the last 2-3 roles.
    *   `skills_detailed`: For top 5 skills, provide context and duration.
4.  **Certs**: Extract all certification names.
5.  **Normalized Text**: Create a dense paragraph concatenating Name, Titles, Summary, Experience and Primary Skills.

Use the `read_resume_file` tool to get content. Do NOT hallucinate data. Use 'Unknown' for missing text fields, 0 for missing numbers.
"""
        )

    async def extract(self, file_path: str) -> ResumeData:
        # Prompting the agent to use the tool on the file
        prompt = f"Extract structured data from this resume file: {file_path}"

        # Using structured_output to enforce Pydantic schema
        result = self.agent(
            structured_output_model=ResumeData,
            prompt=prompt
        )
        
        return result
