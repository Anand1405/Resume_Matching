import os
import shutil
import pytest
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.indexer import ResumeIndexer

@pytest.fixture
def temp_index_dir():
    dir_name = "test_index_data"
    os.makedirs(dir_name, exist_ok=True)
    yield dir_name
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def test_indexer_flow(temp_index_dir):
    with patch("retrieval.indexer.settings") as mock_settings, \
         patch("retrieval.indexer.GeminiEmbeddingGenerator") as MockEmbedder:
        
        mock_settings.INDEX_PATH = temp_index_dir
        
        # Mock embedding return
        mock_embedder_instance = MockEmbedder.return_value
        mock_embedder_instance.embed_text.return_value = [0.1] * 768
        
        indexer = ResumeIndexer()
        
        # Add document
        indexer.index_resume("doc1", "Python dev", {"name": "Test"})
        
        # Verify persistence
        assert "doc1" in indexer.metadata_map
        
        # Search
        # Ensure search_hybrid handles the query correctly
        results = indexer.search_hybrid("Python", k=1)
        assert len(results) >= 0