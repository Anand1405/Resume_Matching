from strands import tool
from pydantic import BaseModel, Field

class ScoreInput(BaseModel):
    experience_score: float = Field(..., description="Score for experience (0-1)")
    skills_score: float = Field(..., description="Score for skills (0-1)")
    education_score: float = Field(..., description="Score for education (0-1)")
    projects_score: float = Field(..., description="Score for projects (0-1)")

class ScoreOutput(BaseModel):
    final_score: float
    breakdown: str
    experience_score: float
    skills_score: float
    education_score: float
    projects_score: float

@tool
def calculate_score(scores: ScoreInput) -> ScoreOutput:
    """
    Calculates the weighted final score based on experience, skills, education, and projects.
    Weights:
    - Experience: 30%
    - Skills: 40%
    - Education: 10%
    - Projects: 20%
    """
    print("Calculating score with inputs:", scores)
    final = (0.30 * scores.get("experience_score")) + \
            (0.40 * scores.get("skills_score")) + \
            (0.10 * scores.get("education_score")) + \
            (0.20 * scores.get("projects_score"))

    final_rounded = round(final, 3)

    breakdown = (
        f"Final ({final_rounded}) = "
        f"Exp ({scores.get('experience_score')} * 0.3) + "
        f"Skills ({scores.get('skills_score')} * 0.4) + "
        f"Edu ({scores.get('education_score')} * 0.1) + "
        f"Proj ({scores.get('projects_score')} * 0.2)"
    )

    return ScoreOutput(
        final_score=final_rounded,
        breakdown=breakdown,
        experience_score=scores.get("experience_score"),
        skills_score=scores.get("skills_score"),
        education_score=scores.get("education_score"),
        projects_score=scores.get("projects_score")
    )
