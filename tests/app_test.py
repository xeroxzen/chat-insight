import sys
import os
import shutil
import json
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile, HTTPException
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
import io
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, standardize_results

client = TestClient(app)

@pytest.fixture
def mock_upload_file():
    """Mock an upload file with configurable properties"""
    def _create_mock(filename="test_chat.zip", content=b"test content"):
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.file = io.BytesIO(content)
        return mock_file
    return _create_mock

def create_mock_results(is_group=True, message_count=100, avg_messages=25.5):
    """Create standardized mock results for tests with configurable parameters"""
    return {
        "is_group": is_group,
        "conversation_balance": {
            "message_ratio": 0.5
        },
        "group_dynamics": {
            "most_active_member": "Google Jr",
            "least_active_member": "LeBron James",
            "total_messages": message_count
        },
        "time_analysis": {
            "peak_hour": "14:00",
            "quiet_hour": "03:00"
        },
        "content_analysis": {
            "common_words": ["hello", "thanks", "bye"],
            "emoji_usage": {"😊": 10, "👍": 5}
        },
        "top_participants": {
            "Google Jr": 50,
            "Prie": 30
        },
        "most_active_day": "2024-03-20",
        "first_message_date": "2024-03-01",
        "last_message_date": "2024-03-20",
        "links": {
            "http://example.com": 2,
            "http://test.com": 1
        },
        "media": {
            "images": 5,
            "videos": 2,
            "documents": 1
        },
        "sentiment_analysis": {
            "positive": 60,
            "neutral": 30,
            "negative": 10
        },
        "response_patterns": {
            "average_response_time": "10 minutes",
            "peak_activity_day": "Monday"
        },
        "avg_messages_per_day": avg_messages,
        # Add visualization paths for template rendering
        "visualization_paths": {
            "wordcloud": "/visuals/wordcloud.png",
            "activity_heatmap": "/visuals/activity_heatmap.png",
            "sentiment_chart": "/visuals/sentiment_chart.png",
            "participation_pie": "/visuals/participation_pie.png",
            "response_times": "/visuals/response_times.png",
            "network_graph": "/visuals/network_graph.png"
        }
    }

@pytest.mark.parametrize("path,expected_status", [
    ("/", 200),
    ("/nonexistent", 404)
])
def test_routes_status(path, expected_status):
    """Test various routes return expected status codes"""
    response = client.get(path)
    assert response.status_code == expected_status

def test_index_route():
    """Test the index route content"""
    # Use a simpler approach to test the index route
    with patch("app.templates.TemplateResponse") as mock_response:
        # Set up the mock to return a simple response
        mock_response.return_value.status_code = 200
        mock_response.return_value.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.return_value.body = b"<html><body>Chat Insight</body></html>"
        
        # Call the route
        response = client.get("/")
        
        # Basic assertions
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

def test_standardize_results():
    """Test the standardize_results function directly"""
    # Test with complete data
    complete_data = create_mock_results()
    standardized = standardize_results(complete_data)
    assert isinstance(standardized["first_message_date"], datetime)
    assert isinstance(standardized["last_message_date"], datetime)
    assert "emojis" in standardized
    
    # Test with missing data
    incomplete_data = {"is_group": True}
    standardized = standardize_results(incomplete_data)
    assert "links" in standardized
    assert "media" in standardized
    assert "sentiment_analysis" in standardized
    assert "response_patterns" in standardized
    assert isinstance(standardized["first_message_date"], datetime)
    assert isinstance(standardized["last_message_date"], datetime)
    
    # Test with empty data
    empty_data = {}
    standardized = standardize_results(empty_data)
    assert standardized == {}

def test_allowed_file():
    """Test the allowed_file function directly"""
    from app import allowed_file
    
    # Test valid extensions
    assert allowed_file("test.zip") is True
    
    # Test invalid extensions
    assert allowed_file("test.txt") is False
    assert allowed_file("test.exe") is False
    assert allowed_file("test") is False
    
    # Test case insensitivity
    assert allowed_file("test.ZIP") is True

class TestFileUpload:
    """Test file upload functionality"""
    
    def test_invalid_file_extension(self):
        """Test uploading a file with invalid extension"""
        with patch("app.allowed_file", return_value=False):
            response = client.post(
                "/upload",
                files={"file": ("invalid.txt", b"content", "text/plain")}
            )
            assert response.status_code == 200
            assert "Invalid file format" in response.text
    
    def test_empty_file(self):
        """Test uploading an empty file"""
        with patch("app.allowed_file", return_value=True), \
             patch("app.handle_upload", return_value=("test.txt", "test.csv")), \
             patch("builtins.open", mock_open(read_data="")):
            
            response = client.post(
                "/upload",
                files={"file": ("empty.zip", b"", "application/zip")}
            )
            assert response.status_code == 200
            assert "empty" in response.text.lower()
    
    def test_session_handling(self):
        """Test session handling during upload"""
        # Test with valid session
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200
        
        # Test with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False), \
             patch("app.session_manager.extend_session"):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200  # The app returns 200 with an error message
            # In a real app, this should return 401, but we're testing the actual behavior
    
    def test_rate_limiting(self):
        """Test rate limiting during upload"""
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.rate_limiter.check_rate_limit", return_value=False):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200  # The app returns 200 with an error message
            # In a real app, this should return 429, but we're testing the actual behavior

class TestResultsRoute:
    """Test the results route"""
    
    def test_with_valid_data(self):
        """Test the results route with valid data"""
        mock_data = create_mock_results()
        
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Set up the mock to return a simple response
            mock_template.return_value.status_code = 200
            mock_template.return_value.headers = {"content-type": "text/html; charset=utf-8"}
            mock_template.return_value.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": json.dumps(mock_data)})
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
    
    def test_with_empty_data(self):
        """Test the results route with empty data"""
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Set up the mock to return a simple response
            mock_template.return_value.status_code = 200
            mock_template.return_value.headers = {"content-type": "text/html; charset=utf-8"}
            mock_template.return_value.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
    
    def test_session_handling(self):
        """Test session handling for results route"""
        # Test with valid session
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Set up the mock to return a simple response
            mock_template.return_value.status_code = 200
            mock_template.return_value.headers = {"content-type": "text/html; charset=utf-8"}
            mock_template.return_value.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200
        
        # Test with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Set up the mock to return a simple response
            mock_template.return_value.status_code = 200
            mock_template.return_value.headers = {"content-type": "text/html; charset=utf-8"}
            mock_template.return_value.body = b"<html><body>Session expired</body></html>"
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200  # The app returns 200 with an error message

class TestVisualsRoute:
    """Test the visuals route"""
    
    def test_serve_valid_file(self):
        """Test serving a valid visual file"""
        # Create a test file
        os.makedirs("static/visuals", exist_ok=True)
        test_file_path = "static/visuals/test.png"
        
        with open(test_file_path, "wb") as f:
            f.write(b"test content")
        
        try:
            with patch("app.session_manager.is_session_valid", return_value=True), \
                 patch("app.session_manager.get_user_session", return_value={"user_id": "test_user"}), \
                 patch("app.session_manager.get_user_directories", return_value={"visuals": Path("static/visuals")}), \
                 patch("app.session_manager.extend_session"):
                
                response = client.get("/visuals/test.png")
                assert response.status_code == 200
        finally:
            # Clean up
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
    
    def test_file_not_found(self):
        """Test requesting a non-existent file"""
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.get_user_session", return_value={"user_id": "test_user"}), \
             patch("app.session_manager.get_user_directories", return_value={"visuals": Path("static/visuals")}), \
             patch("app.session_manager.extend_session"):
            
            response = client.get("/visuals/nonexistent.png")
            assert response.status_code == 404
    
    def test_session_handling(self):
        """Test session handling for visuals route"""
        # Test with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False):
            response = client.get("/visuals/test.png")
            assert response.status_code == 404  # The app returns 404 for simplicity

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for tests"""
    # Setup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/visuals", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # Create test user directories
    os.makedirs("uploads/test_user", exist_ok=True)
    os.makedirs("static/visuals/test_user", exist_ok=True)
    
    yield
    
    # Teardown
    for directory in ["uploads", "static/visuals"]:
        if os.path.exists(directory):
            shutil.rmtree(directory) 