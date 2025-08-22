"""
Brain 2 Test Suite
Comprehensive tests for Qwen3-Reranker-4B implementation
"""

# Test configuration
TEST_MODEL_PATH = "/workspace/models/qwen3/reranker-4b"
TEST_CACHE_DIR = "/workspace/models/cache/test"
TEST_REDIS_URL = "redis://redis:6379/1"  # Use different DB for tests

# Test data
SAMPLE_QUERY = "artificial intelligence machine learning"
SAMPLE_DOCUMENTS = [
    {
        "text": "Artificial intelligence is transforming modern technology through machine learning algorithms.",
        "doc_id": "doc1",
        "metadata": {"source": "tech_article", "category": "AI"}
    },
    {
        "text": "The weather today is sunny with a chance of rain in the afternoon.",
        "doc_id": "doc2", 
        "metadata": {"source": "weather_report", "category": "weather"}
    },
    {
        "text": "Machine learning models require large datasets for effective training and validation.",
        "doc_id": "doc3",
        "metadata": {"source": "research_paper", "category": "ML"}
    }
]

# Expected test results
EXPECTED_TOP_DOCUMENT = "doc1"  # Most relevant to AI/ML query
EXPECTED_MIN_RELEVANCE_SCORE = 0.7  # Minimum score for relevant documents
