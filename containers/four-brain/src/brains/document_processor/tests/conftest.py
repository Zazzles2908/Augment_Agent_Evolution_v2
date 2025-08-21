"""
Pytest configuration and fixtures for Brain 4 tests
Based on implementation_part_6.md specifications
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_settings(temp_directory):
    """Create mock settings for testing"""
    
    class MockSettings:
        def __init__(self):
            self.model_cache_dir = temp_directory / "models"
            self.temp_dir = temp_directory / "temp"
            self.data_dir = temp_directory / "data"
            self.database_url = "postgresql://test:test@localhost:5432/test"
            self.redis_url = "redis://localhost:6379/1"
            self.max_concurrent_tasks = 2
            self.batch_size_documents = 2
            self.max_file_size_mb = 10
            self.supported_formats = ["pdf", "docx", "txt", "md"]
            self.chunk_size = 1000
            self.max_vram_usage = 0.75
            self.target_vram_usage = 0.65
            
            # Create directories
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    return MockSettings()

@pytest.fixture
def mock_docling_converter():
    """Create mock Docling converter"""
    converter = Mock()
    
    # Mock conversion result
    mock_result = Mock()
    mock_document = Mock()
    mock_document.export_to_markdown.return_value = "# Test Document\nContent here"
    mock_document.export_to_json.return_value = '{"title": "Test Document"}'
    mock_document.pages = [Mock()]
    mock_document.texts = []
    mock_document.tables = []
    mock_document.pictures = []
    mock_result.document = mock_document
    
    converter.convert.return_value = mock_result
    
    return converter

@pytest.fixture
def sample_documents(temp_directory):
    """Create sample documents for testing"""
    docs = []
    
    # PDF document
    pdf_doc = temp_directory / "sample.pdf"
    pdf_doc.write_text("Sample PDF content for testing document processing")
    docs.append(str(pdf_doc))
    
    # Text document
    txt_doc = temp_directory / "sample.txt"
    txt_doc.write_text("Sample text content for testing text processing")
    docs.append(str(txt_doc))
    
    # Markdown document
    md_doc = temp_directory / "sample.md"
    md_doc.write_text("# Sample Markdown\n\nContent for testing markdown processing")
    docs.append(str(md_doc))
    
    # Large document
    large_doc = temp_directory / "large.txt"
    large_doc.write_text("Large document content " * 1000)
    docs.append(str(large_doc))
    
    return docs

@pytest.fixture
async def mock_database():
    """Create mock database connection"""
    
    class MockConnection:
        async def execute(self, query, *args):
            return None
        
        async def fetch(self, query, *args):
            return []
        
        async def fetchrow(self, query, *args):
            return {
                "task_id": "test_123",
                "status": "completed",
                "content_text": "Test content"
            }
        
        async def executemany(self, query, args_list):
            return None
    
    class MockPool:
        def acquire(self):
            return MockConnection()
        
        async def close(self):
            pass
    
    return MockPool()

@pytest.fixture
def mock_redis():
    """Create mock Redis connection"""
    
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def ping(self):
            return True
        
        async def publish(self, channel, message):
            return 1
        
        async def subscribe(self, *channels):
            return True
        
        async def close(self):
            pass
    
    return MockRedis()

# Test configuration functions

def pytest_addoption(parser):
    """Add custom pytest options"""
    
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    
    parser.addoption(
        "--performance",
        action="store_true", 
        default=False,
        help="Run performance tests"
    )

def pytest_configure(config):
    """Configure pytest markers"""
    
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options"""
    
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    if not config.getoption("--performance"):
        skip_performance = pytest.mark.skip(reason="need --performance option to run")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)
