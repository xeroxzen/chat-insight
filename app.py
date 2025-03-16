import os
import json
import sys
import logging
from pathlib import Path
import secrets
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from contextlib import asynccontextmanager
import io
import zipfile
from fastapi.responses import StreamingResponse

# Adding the current directory to Python path to avoid directory related errors
sys.path.append(str(Path(__file__).parent))

from parser import parse_chat_log
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware

from analyzer import analyze_chat_log
from file_handler import allowed_file, handle_upload
from session_manager import SessionManager
from rate_limiter import RateLimiter

# Setting up logging
logger = logging.getLogger(__name__)

# Generating a secure secret key for session management
SECRET_KEY = secrets.token_urlsafe(32)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
        return response

# Initializing session manager and rate limiter
session_manager = SessionManager(SECRET_KEY)
rate_limiter = RateLimiter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown events"""
    # Startup: Clean up old sessions
    session_manager.cleanup_old_sessions()
    yield
    # Shutdown: Any cleanup code would go here

app = FastAPI(lifespan=lifespan)

# Adding middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="chat_insight_session",
    max_age=24 * 60 * 60,  # 24 hours
    same_site="lax",  # Protect against CSRF
    path="/"  # Make cookie available for all paths
)

# Adding CORS middleware with strict settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To be changed in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adding trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # To be changed in production
)

# Ensuring base directories exist
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
VISUALS_FOLDER = "static/visuals"

for folder in [UPLOAD_FOLDER, STATIC_FOLDER, VISUALS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Mounting static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount the visuals directory directly for easier access to visualization files
app.mount("/visuals", StaticFiles(directory="static/visuals"), name="visuals")

# Note: We're keeping both the direct mount and the route-based approach for compatibility
# The route-based approach (/visuals/{user_id}/{filename}) provides security checks
# while the direct mount is simpler but less secure

# Setting up Jinja2 templates
templates = Jinja2Templates(directory="templates")

def standardize_results(results: dict) -> dict:
    """Standardize the results format for template rendering."""
    if not results:
        return {"visualization_paths": {}}
        
    # Converting all date strings to datetime objects
    date_fields = ["most_active_day", "first_message_date", "last_message_date"]
    for field in date_fields:
        if field in results:
            try:
                results[field] = datetime.strptime(results[field], "%Y-%m-%d")
            except ValueError:
                results[field] = None
            
    # Mapping emoji data to top-level emojis field
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
    
    # Ensure visualization_paths exists
    if "visualization_paths" not in results:
        results["visualization_paths"] = {}
            
    return results

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return appropriate responses."""
    if exc.status_code == 401:
        # For session expired, redirect to home page
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Your session has expired. Please try again."}
        )
    elif exc.status_code == 429:
        # For rate limiting
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Too many requests. Please try again later."}
        )
    else:
        # For other errors
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": str(exc.detail)}
        )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        # Getting or creating user session
        session = session_manager.get_user_session(request)
        
        # Checking rate limit
        if not rate_limiter.check_rate_limit(request):
            raise HTTPException(status_code=429, detail="Too many requests")
        
        return templates.TemplateResponse(request, "index.html", {})
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "An unexpected error occurred. Please try again."}
        )

@app.get("/results")
async def read_root(request: Request):
    # Checking session validity
    if not session_manager.is_session_valid(request):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Checking rate limit
    if not rate_limiter.check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Extending session
    session_manager.extend_session(request)
    
    # Getting user session
    session = session_manager.get_user_session(request)
    user_id = session["user_id"]
    
    results_str = request.query_params.get("results", "{}")
    try:
        results = json.loads(results_str)
        results = standardize_results(results)
        
        # Ensure visualization_paths is properly set in the results
        if "visualization_paths" not in results:
            results["visualization_paths"] = {}
        
        # Update visualization paths to include user_id
        updated_paths = {}
        for key, path in results["visualization_paths"].items():
            if isinstance(path, str):
                if not path.startswith(f"/visuals/{user_id}/"):
                    # Extract filename if it's a path
                    if "/" in path:
                        filename = path.split("/")[-1]
                    else:
                        filename = path
                    # Use the same format as in upload_file
                    updated_paths[key] = f"/visuals/{user_id}/{filename}"
                else:
                    # Path already has the correct format
                    updated_paths[key] = path
            else:
                updated_paths[key] = path
        
        results["visualization_paths"] = updated_paths
            
    except json.JSONDecodeError:
        results = {"visualization_paths": {}}
        
    return templates.TemplateResponse(request, "results.html", {"results": results})

@app.get("/visuals/{user_id}/{filename}")
async def serve_visuals(request: Request, user_id: str, filename: str):
    # Checking session validity
    if not session_manager.is_session_valid(request):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Checking rate limit
    if not rate_limiter.check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Extending session
    session_manager.extend_session(request)
    
    # Getting user session
    session = session_manager.get_user_session(request)
    
    # Security check: Only allow access to the user's own files
    if user_id != session["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get user directories
    user_dirs = session_manager.get_user_directories(user_id)
    
    # Checking if file exists in user's directory
    file_path = user_dirs["visuals"] / filename
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path))

@app.get("/visuals/{filename}")
async def serve_visuals_legacy(request: Request, filename: str):
    # Checking session validity
    if not session_manager.is_session_valid(request):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Checking rate limit
    if not rate_limiter.check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Extending session
    session_manager.extend_session(request)
    
    # Getting user session
    session = session_manager.get_user_session(request)
    user_dirs = session_manager.get_user_directories(session["user_id"])
    
    # Checking if file exists in user's directory
    file_path = user_dirs["visuals"] / filename
    if not file_path.exists():
        logger.error(f"Legacy file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path))

@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Serve the privacy policy page."""
    try:
        # No need to check session validity or rate limit for public page
        return templates.TemplateResponse("privacy_policy.html", {"request": request})
    except Exception as e:
        logger.error(f"Error in privacy policy route: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "An unexpected error occurred. Please try again."}
        )

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # Checking session validity
        if not session_manager.is_session_valid(request):
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Checking rate limit
        if not rate_limiter.check_rate_limit(request):
            raise HTTPException(status_code=429, detail="Too many requests")
        
        # Extending session
        session_manager.extend_session(request)
        
        # Getting user session and file paths
        session = session_manager.get_user_session(request)
        file_paths = session_manager.get_user_file_paths(session["user_id"], file.filename)
        
        if not allowed_file(file.filename):
            return templates.TemplateResponse(
                request, "index.html", {"error": "Invalid file format"}
            )

        txt_file_path, csv_file_path = handle_upload(file, file_paths)
        
        # Adding logging to check file contents
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return templates.TemplateResponse(
                    request, "index.html", 
                    {"error": "The uploaded file is empty"}
                )
        
        try:
            parse_chat_log(str(txt_file_path), str(csv_file_path))
        except Exception as e:
            return templates.TemplateResponse(
                request, "index.html",
                {"error": f"Error parsing chat log: {str(e)}"}
            )
            
        try:
            analysis_results = analyze_chat_log(str(csv_file_path), file_paths["visuals"])
        except ValueError as e:
            return templates.TemplateResponse(
                request, "index.html",
                {"error": f"Error analyzing chat log: {str(e)}"}
            )
        
        # Standardizing results before sending to template
        standardized_results = standardize_results(analysis_results)
        
        # Log the original visualization paths from analysis_results
        logger.info("Original visualization paths from analysis_results:")
        for key, path in analysis_results.get("visualization_paths", {}).items():
            logger.info(f"Original {key}: {path}")
        
        # Getting the relative paths for visualizations and updating standardized_results
        visualization_paths = {}
        for key, path in standardized_results.get("visualization_paths", {}).items():
            if isinstance(path, (str, Path)):
                # Converting absolute path to relative path from visuals directory
                try:
                    # Extract the user_id and filename
                    user_id = session['user_id']
                    
                    # If path is a Path object, convert to string
                    path_str = str(path)
                    
                    # Extract the filename
                    filename = os.path.basename(path_str)
                    
                    # Create the direct path
                    direct_path = f"/visuals/{user_id}/{filename}"
                    
                    visualization_paths[key] = direct_path
                    logger.info(f"Visualization path for {key}: {direct_path}")
                except Exception as e:
                    logger.error(f"Error processing path for {key}: {path}, Error: {str(e)}")
                    # If we can't process the path, use a fallback
                    try:
                        filename = os.path.basename(str(path))
                        visualization_paths[key] = f"/visuals/{session['user_id']}/{filename}"
                        logger.info(f"Using fallback filename for {key}: {visualization_paths[key]}")
                    except:
                        visualization_paths[key] = str(path)
                        logger.error(f"Could not extract filename for {key}: {path}")
        
        # Update the visualization_paths in standardized_results
        standardized_results["visualization_paths"] = visualization_paths
        
        # Logging all visualization paths being passed to template
        logger.info("All visualization paths being passed to template:")
        for key, path in visualization_paths.items():
            logger.info(f"{key}: {path}")
            
        # Check specifically for group visualization paths
        if standardized_results.get('is_group', False):
            logger.info("Checking group-specific visualization paths:")
            if 'group_participation' in visualization_paths:
                logger.info(f"group_participation: {visualization_paths['group_participation']}")
            else:
                logger.error("group_participation is missing from visualization_paths!")
                
            if 'group_interaction_network' in visualization_paths:
                logger.info(f"group_interaction_network: {visualization_paths['group_interaction_network']}")
            else:
                logger.error("group_interaction_network is missing from visualization_paths!")
            
        # Log the is_group flag to verify it's being set correctly
        logger.info(f"Is group chat: {standardized_results.get('is_group', False)}")
        
        return templates.TemplateResponse(request, "results.html", {
            "results": standardized_results
        })
        
    except Exception as e:
        logger.error(f"Error in upload route: {str(e)}")
        return templates.TemplateResponse(
            request, "index.html",
            {"error": f"An unexpected error occurred: {str(e)}"}
        )

@app.get("/download_results")
async def download_results(request: Request):
    # Check session validity
    if not session_manager.is_session_valid(request):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user session and directories
    session = session_manager.get_user_session(request)
    user_id = session["user_id"]
    user_dirs = session_manager.get_user_directories(user_id)
    visuals_dir = user_dirs["visuals"]

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Loop through files in the visuals directory and add them to the ZIP
        for file_path in visuals_dir.glob("*"):
            if file_path.is_file():
                zip_file.write(file_path, arcname=file_path.name)
    zip_buffer.seek(0)

    # Return the ZIP file as a StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={user_id}_analysis_results.zip"}
    )


@app.get("/debug/files/{user_id}")
async def debug_files(request: Request, user_id: str):
    """Debug route to check the actual files in a user's directory."""
    try:
        # Checking session validity
        if not session_manager.is_session_valid(request):
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Getting user session
        session = session_manager.get_user_session(request)
        
        # Security check: Only allow access to the user's own files or admin
        if user_id != session["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get user directories
        user_dirs = session_manager.get_user_directories(user_id)
        
        # List files in the user's visuals directory
        visuals_dir = user_dirs["visuals"]
        files = []
        if visuals_dir.exists():
            files = [str(f.relative_to(visuals_dir)) for f in visuals_dir.glob("*") if f.is_file()]
        
        # Return the list of files
        return {
            "user_id": user_id,
            "visuals_dir": str(visuals_dir),
            "files": files,
            "file_count": len(files)
        }
    except Exception as e:
        logger.error(f"Error in debug_files route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/debug/paths")
async def debug_paths(request: Request):
    """Debug route to check the visualization paths being passed to the template."""
    try:
        # Checking session validity
        if not session_manager.is_session_valid(request):
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Getting user session
        session = session_manager.get_user_session(request)
        user_id = session["user_id"]
        
        # Get user directories
        user_dirs = session_manager.get_user_directories(user_id)
        
        # List files in the user's visuals directory
        visuals_dir = user_dirs["visuals"]
        files = []
        if visuals_dir.exists():
            files = [str(f.relative_to(visuals_dir)) for f in visuals_dir.glob("*") if f.is_file()]
        
        # Check for group-specific files
        group_files = [f for f in files if "group_" in f]
        
        # Create mock visualization paths
        visualization_paths = {}
        for file in files:
            key = Path(file).stem
            visualization_paths[key] = f"/visuals/{user_id}/{file}"
        
        return {
            "user_id": user_id,
            "visuals_dir": str(visuals_dir),
            "files": files,
            "group_files": group_files,
            "visualization_paths": visualization_paths
        }
    except Exception as e:
        logger.error(f"Error in debug_paths route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/debug/html")
async def debug_html(request: Request):
    """Debug route to check the actual HTML being rendered."""
    try:
        # Checking session validity
        if not session_manager.is_session_valid(request):
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Getting user session
        session = session_manager.get_user_session(request)
        user_id = session["user_id"]
        
        # Get user directories
        user_dirs = session_manager.get_user_directories(user_id)
        
        # List files in the user's visuals directory
        visuals_dir = user_dirs["visuals"]
        files = []
        if visuals_dir.exists():
            files = [str(f.relative_to(visuals_dir)) for f in visuals_dir.glob("*") if f.is_file()]
        
        # Create mock results with visualization paths
        results = {
            "is_group": True,
            "visualization_paths": {},
            "group_dynamics": {
                "most_active_member": "User1",
                "least_active_member": "User2"
            }
        }
        
        # Add visualization paths for all files
        for file in files:
            key = Path(file).stem
            results["visualization_paths"][key] = f"/visuals/{user_id}/{file}"
        
        # Render the template with the mock results
        html = templates.get_template("results.html").render(results=results)
        
        # Extract the group visualization sections
        import re
        group_sections = re.findall(r'<div class="chart-wrapper">\s*<h2>(Group [^<]+)</h2>.*?</div>\s*</div>', html, re.DOTALL)
        
        return {
            "user_id": user_id,
            "is_group": True,
            "group_files": [f for f in files if "group_" in f],
            "group_sections": group_sections,
            "visualization_paths": results["visualization_paths"]
        }
    except Exception as e:
        logger.error(f"Error in debug_html route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
