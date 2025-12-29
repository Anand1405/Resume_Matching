import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.resume_extractor import ResumeExtractionAgent
from agents.matcher import MatchingAgent

@pytest.mark.asyncio
async def test_resume_extractor():
    # Mock the Agent class used inside resume_extractor
    with patch("agents.resume_extractor.Agent") as MockAgent:
        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance
        
        # Mock successful extraction response
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "message": {
                "content": [{
                    "toolUse": {
                        "input": {
                            "name": "John Doe",
                            "email": "john@example.com",
                            "skills": ["Python", "AI"],
                            "experience": ["Dev at TechCorp"],
                            "education": ["BS CS"]
                        }
                    }
                }]
            }
        }
        mock_instance.side_effect = AsyncMock(return_value=mock_response)

        extractor = ResumeExtractionAgent()
        result = await extractor.extract("dummy.pdf")
        
        data = result.to_dict()['message']['content'][0]['toolUse']['input']
        assert data['name'] == "John Doe"
        assert "Python" in data['skills']

@pytest.mark.asyncio
async def test_matching_agent():
    with patch("agents.matcher.Agent") as MockAgent:
        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance
        
        # Mock scoring response
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "message": {
                "content": [{
                    "toolUse": {
                        "input": {
                            "score": 88,
                            "reasoning": "Excellent fit.",
                            "missing_skills": []
                        }
                    }
                }]
            }
        }
        mock_instance.side_effect = AsyncMock(return_value=mock_response)

        matcher = MatchingAgent()
        candidate = {"name": "Alice", "skills": ["Go"]}
        result = await matcher.match("Job Description", candidate)
        
        score_data = result.to_dict()['message']['content'][0]['toolUse']['input']
        assert score_data['score'] == 88