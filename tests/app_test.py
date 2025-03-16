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
import zipfile
from starlette.middleware.sessions import SessionMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, standardize_results

# Add session middleware to the app for testing
app.add_middleware(SessionMiddleware, secret_key="test_secret_key")

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
        # Adding visualization paths for template rendering
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
    # Create a test client with a mocked app
    with patch("app.session_manager.get_user_session"), \
         patch("app.rate_limiter.check_rate_limit", return_value=True):
        
        # Call the route
        response = client.get("/")
        
        # Basic assertions
        assert response.status_code == 200

def test_standardize_results():
    """Test the standardize_results function directly"""
    # Testing with complete data
    complete_data = create_mock_results()
    standardized = standardize_results(complete_data)
    assert isinstance(standardized["first_message_date"], datetime)
    assert isinstance(standardized["last_message_date"], datetime)
    assert "emojis" in standardized
    
    # Testing with missing data
    incomplete_data = {"is_group": True}
    standardized = standardize_results(incomplete_data)
    assert "links" in standardized
    assert "media" in standardized
    assert "sentiment_analysis" in standardized
    assert "response_patterns" in standardized
    assert isinstance(standardized["first_message_date"], datetime)
    assert isinstance(standardized["last_message_date"], datetime)
    
    # Testing with empty data
    empty_data = {}
    standardized = standardize_results(empty_data)
    expected_empty_result = {"visualization_paths": {}}
    assert standardized == expected_empty_result

def test_allowed_file():
    """Test the allowed_file function directly"""
    from app import allowed_file
    
    # Testing valid extensions
    assert allowed_file("test.zip") is True
    
    # Testing invalid extensions
    assert allowed_file("test.txt") is False
    assert allowed_file("test.exe") is False
    assert allowed_file("test") is False
    
    # Testing case insensitivity
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
        # Testing with valid session
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200
        
        # Testing with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False), \
             patch("app.session_manager.extend_session"):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200
    
    def test_rate_limiting(self):
        """Test rate limiting during upload"""
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.rate_limiter.check_rate_limit", return_value=False):
            
            response = client.post(
                "/upload",
                files={"file": ("test.zip", b"content", "application/zip")}
            )
            assert response.status_code == 200

class TestResultsRoute:
    """Test the results route"""
    
    def test_with_valid_data(self):
        """Test the results route with valid data"""
        mock_data = create_mock_results()
        
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Setting up the mock to return a simple response
            mock_instance = mock_template.return_value
            mock_instance.status_code = 200
            mock_instance.headers = {"content-type": "text/html; charset=utf-8"}
            mock_instance.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": json.dumps(mock_data)})
            assert response.status_code == 200
            
            # Verifying that TemplateResponse was called with the correct template
            mock_template.assert_called_once()
            assert "results.html" in mock_template.call_args[0] or "results.html" in str(mock_template.call_args)
    
    def test_with_empty_data(self):
        """Test the results route with empty data"""
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Setting up the mock to return a simple response
            mock_instance = mock_template.return_value
            mock_instance.status_code = 200
            mock_instance.headers = {"content-type": "text/html; charset=utf-8"}
            mock_instance.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200
            
            # Verifying that TemplateResponse was called with the correct template
            mock_template.assert_called_once()
            assert "results.html" in mock_template.call_args[0] or "results.html" in str(mock_template.call_args)
    
    def test_session_handling(self):
        """Test session handling for results route"""
        # Testing with valid session
        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.extend_session"), \
             patch("app.templates.TemplateResponse") as mock_template:
            
            # Setting up the mock to return a simple response
            mock_template.return_value.status_code = 200
            mock_template.return_value.headers = {"content-type": "text/html; charset=utf-8"}
            mock_template.return_value.body = b"<html><body>Results</body></html>"
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200
        
        # Testing with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False), \
             patch("app.session_manager.get_user_session", side_effect=HTTPException(status_code=401, detail="Session expired")):
            
            response = client.get("/results", params={"results": "{}"})
            assert response.status_code == 200  # Due to the exception handler
            assert "session has expired" in response.text.lower()

class TestDownloadResults:
    """Test the download_results endpoint"""

    def test_successful_download(self):
        """Test successful download of visualization results"""
        # Create test files
        test_user_id = "test_user"
        test_visuals_dir = Path("static/visuals") / test_user_id
        os.makedirs(test_visuals_dir, exist_ok=True)
        test_files = ["test1.png", "test2.png"]
        
        try:
            # Create some test files
            for filename in test_files:
                with open(test_visuals_dir / filename, "wb") as f:
                    f.write(b"test content")

            with patch("app.session_manager.is_session_valid", return_value=True), \
                 patch("app.session_manager.get_user_session", return_value={"user_id": test_user_id}), \
                 patch("app.session_manager.get_user_directories", return_value={"visuals": test_visuals_dir}):

                response = client.get("/download_results")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/x-zip-compressed"
                assert response.headers["content-disposition"] == f'attachment; filename={test_user_id}_analysis_results.zip'
                
                # Verify the ZIP file contains our test files
                zip_content = io.BytesIO(response.content)
                with zipfile.ZipFile(zip_content) as zf:
                    assert sorted(zf.namelist()) == sorted(test_files)
                    
        finally:
            # Clean up test files
            if os.path.exists(test_visuals_dir):
                shutil.rmtree(test_visuals_dir)

    def test_download_with_invalid_session(self):
        """Test download attempt with invalid session"""
        with patch("app.session_manager.is_session_valid", return_value=False), \
             patch("app.session_manager.get_user_session", side_effect=HTTPException(status_code=401, detail="Session expired")):
            
            response = client.get("/download_results")
            assert response.status_code == 200  # Due to the exception handler
            assert "session has expired" in response.text.lower()

    def test_download_with_empty_directory(self):
        """Test download attempt with empty visuals directory"""
        test_user_id = "test_user"
        test_visuals_dir = Path("static/visuals") / test_user_id
        os.makedirs(test_visuals_dir, exist_ok=True)

        try:
            with patch("app.session_manager.is_session_valid", return_value=True), \
                 patch("app.session_manager.get_user_session", return_value={"user_id": test_user_id}), \
                 patch("app.session_manager.get_user_directories", return_value={"visuals": test_visuals_dir}):

                response = client.get("/download_results")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/x-zip-compressed"
                assert response.headers["content-disposition"] == f'attachment; filename={test_user_id}_analysis_results.zip'
                
                # Verify the ZIP file is empty
                zip_content = io.BytesIO(response.content)
                with zipfile.ZipFile(zip_content) as zf:
                    assert len(zf.namelist()) == 0

        finally:
            if os.path.exists(test_visuals_dir):
                shutil.rmtree(test_visuals_dir)

    def test_download_with_nonexistent_directory(self):
        """Test download attempt with non-existent visuals directory"""
        test_user_id = "nonexistent_user"
        test_visuals_dir = Path("static/visuals") / test_user_id

        with patch("app.session_manager.is_session_valid", return_value=True), \
             patch("app.session_manager.get_user_session", return_value={"user_id": test_user_id}), \
             patch("app.session_manager.get_user_directories", return_value={"visuals": test_visuals_dir}):

            response = client.get("/download_results")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-zip-compressed"
            assert response.headers["content-disposition"] == f'attachment; filename={test_user_id}_analysis_results.zip'
            
            # Verify the ZIP file is empty
            zip_content = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_content) as zf:
                assert len(zf.namelist()) == 0

class TestVisualsRoute:
    """Test the visuals route"""
    
    def test_serve_valid_file(self):
        """Test serving a valid visual file"""
        # Creating a test file
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
            # Cleaning up
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
        # Testing with invalid session
        with patch("app.session_manager.is_session_valid", return_value=False):
            response = client.get("/visuals/test.png")
            assert response.status_code == 404

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for tests"""
    # Setup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/visuals", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # User directories
    os.makedirs("uploads/test_user", exist_ok=True)
    os.makedirs("static/visuals/test_user", exist_ok=True)
    
    yield
    
    # Teardown
    for directory in ["uploads", "static/visuals"]:
        if os.path.exists(directory):
            shutil.rmtree(directory) 