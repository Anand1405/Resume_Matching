
import pytest
from tools.scoring_tool import calculate_score, ScoreInput

def test_calculate_score_deterministic():
    input_data = ScoreInput(
        experience_score=0.9,
        skills_score=0.8,
        education_score=1.0,
        projects_score=0.7
    )
    # Expected: (0.9*0.3) + (0.8*0.4) + (1.0*0.1) + (0.7*0.2)
    # = 0.27 + 0.32 + 0.10 + 0.14 = 0.83

    result = calculate_score(input_data)

    assert result.final_score == 0.83
    assert "Final (0.83)" in result.breakdown

def test_calculate_score_zeros():
    input_data = ScoreInput(
        experience_score=0.0,
        skills_score=0.0,
        education_score=0.0,
        projects_score=0.0
    )
    result = calculate_score(input_data)
    assert result.final_score == 0.0

def test_calculate_score_ones():
    input_data = ScoreInput(
        experience_score=1.0,
        skills_score=1.0,
        education_score=1.0,
        projects_score=1.0
    )
    result = calculate_score(input_data)
    assert result.final_score == 1.0
