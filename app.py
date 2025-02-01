import os
import json
import sys
from pathlib import Path

# Place the current directory to Python path to avoid dir related errors
sys.path.append(str(Path(__file__).parent))

from parser import parse_chat_log
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from datetime import datetime

from analyzer import analyze_chat_log
from file_handler import allowed_file, handle_upload

app = FastAPI()

# Ensure directories exist
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
VISUALS_FOLDER = "static/visuals"

for folder in [UPLOAD_FOLDER, STATIC_FOLDER, VISUALS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/visuals", StaticFiles(directory="static/visuals"), name="visuals")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

def standardize_results(results: dict) -> dict:
    """Standardize the results format for template rendering."""
    if not results:
        return {}
        
    # Convert all date strings to datetime objects
    date_fields = ["most_active_day", "first_message_date", "last_message_date"]
    for field in date_fields:
        if field in results:
            try:
                results[field] = datetime.strptime(results[field], "%Y-%m-%d")
            except ValueError:
                results[field] = None
            
    # Map emoji data to top-level emojis field
    if "content_analysis" in results and "emoji_usage" in results["content_analysis"]:
        results["emojis"] = results["content_analysis"]["emoji_usage"]
    else:
        results["emojis"] = {}
        
    # Required fields
    required_fields = {
        "links": {},
        "media": {"images": 0, "videos": 0, "documents": 0},
        "sentiment_analysis": {"positive": 0, "neutral": 0, "negative": 0},
        "response_patterns": {"average_response_time": "N/A", "peak_activity_day": "N/A"},
        "first_message_date": datetime.now(),
        "last_message_date": datetime.now(),
        "avg_messages_per_day": 0.0 
    }
    
    for field, default_value in required_fields.items():
        if field not in results:
            results[field] = default_value
            
    return results

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results")
async def read_root(request: Request):
    results_str = request.query_params.get("results", "{}")
    try:
        results = json.loads(results_str)
        results = standardize_results(results)
    except json.JSONDecodeError:
        results = {}
    return templates.TemplateResponse(request, "results.html", {"results": results})

@app.get("/visuals/{filename}")
async def serve_visuals(request: Request, filename: str):
    return FileResponse(f"static/visuals/{filename}")

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        if not allowed_file(file.filename):
            return templates.TemplateResponse(
                request, "index.html", {"error": "Invalid file format"}
            )

        txt_file_path, csv_file_path = handle_upload(file)
        
        # Add logging to check file contents
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return templates.TemplateResponse(
                    request, "index.html", 
                    {"error": "The uploaded file is empty"}
                )
        
        try:
            parse_chat_log(txt_file_path, csv_file_path)
        except Exception as e:
            return templates.TemplateResponse(
                request, "index.html",
                {"error": f"Error parsing chat log: {str(e)}"}
            )
            
        try:
            analysis_results = analyze_chat_log(csv_file_path)
        except ValueError as e:
            return templates.TemplateResponse(
                request, "index.html",
                {"error": f"Error analyzing chat log: {str(e)}"}
            )
        
        # Standardize results before sending to template
        standardized_results = standardize_results(analysis_results)
        
        return templates.TemplateResponse(request, "results.html", {
            "results": standardized_results,
            "request": request
        })
        
    except Exception as e:
        return templates.TemplateResponse(
            request, "index.html",
            {"error": f"An unexpected error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
