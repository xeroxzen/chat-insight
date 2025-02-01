import sys
import os
import shutil
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from unittest.mock import Mock, patch
from datetime import datetime

from app import app

client = TestClient(app)

@pytest.fixture
def mock_upload_file():
    return Mock(spec=UploadFile, filename="test_chat.txt")

def test_index_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def create_mock_results():
    """Create standardized mock results for tests"""
    return {
        "is_group": True,
        "conversation_balance": {
            "message_ratio": 0.5
        },
        "group_dynamics": {
            "most_active_member": "John Doe",
            "least_active_member": "Jane Doe",
            "total_messages": 100
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
            "John": 50,
            "Jane": 30
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
        "avg_messages_per_day": 25.5
    }

def test_results_route():
    mock_results = create_mock_results()
    response = client.get("/results", params={"results": json.dumps(mock_results)})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_serve_visuals():
    # Create a test file in static/visuals
    os.makedirs("static/visuals", exist_ok=True)
    test_file_path = "static/visuals/test.png"
    
    # Create a binary file instead of text file
    with open(test_file_path, "wb") as f:
        f.write(b"test content")

    try:
        response = client.get("/visuals/test.png")
        assert response.status_code == 200
    finally:
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

@pytest.mark.asyncio
async def test_upload_invalid_file(mock_upload_file):
    mock_upload_file.filename = "invalid.zip"
    
    with patch("app.allowed_file", return_value=False):
        response = client.post(
            "/upload",
            files={"file": ("invalid.zip", b"content", "application/zip")}
        )
        
    assert response.status_code == 200
    assert "Invalid file format" in response.text

@pytest.mark.asyncio
async def test_upload_valid_file(mock_upload_file):
    mock_results = create_mock_results()
    
    with patch("app.allowed_file", return_value=True), \
         patch("app.handle_upload", return_value=("test.txt", "test.csv")), \
         patch("app.parse_chat_log"), \
         patch("app.analyze_chat_log", return_value=mock_results):

        response = client.post(
            "/upload",
            files={"file": ("chat.txt", b"chat content", "text/plain")}
        )
        
    assert response.status_code == 200
    # Check for specific content that should be in the results page
    assert "Analysis Results" in response.text  # Check for the title
    assert "Group Participation Distribution" in response.text  # Check for a section header
    assert "Group Interaction Network" in response.text  # Check for another section header
    
    # Verify some of the mock data appears in the response
    assert mock_results["group_dynamics"]["most_active_member"] in response.text
    assert mock_results["group_dynamics"]["least_active_member"] in response.text
    
    # For numeric values, we might need to handle different format possibilities
    avg_messages = mock_results["avg_messages_per_day"]
    possible_formats = [
        str(avg_messages),  # 25.5
        str(int(avg_messages)),  # 25
        f"{avg_messages:.1f}",  # 25.5
        f"{avg_messages:.0f}"   # 26
    ]
    assert any(format in response.text for format in possible_formats), \
        f"Could not find average messages per day in any format: {possible_formats}"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/visuals", exist_ok=True)
    
    yield
    
    # Teardown
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    if os.path.exists("static/visuals"):
        shutil.rmtree("static/visuals") 