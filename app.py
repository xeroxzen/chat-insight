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
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Place the current directory to Python path to avoid dir related errors
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

# Set up logging
logger = logging.getLogger(__name__)

# Generate a secure secret key for session management
SECRET_KEY = secrets.token_urlsafe(32)

# Initialize Prometheus metrics
REQUESTS_TOTAL = Counter('app_requests_total', 'Total number of requests by endpoint', ['endpoint'])
FILE_UPLOAD_COUNT = Counter('app_file_uploads_total', 'Total number of file uploads')
FILE_SIZE_BYTES = Histogram('app_file_size_bytes', 'Distribution of uploaded file sizes in bytes', buckets=[1000, 10000, 100000, 1000000, 10000000])
ANALYSIS_DURATION = Histogram('app_analysis_duration_seconds', 'Time spent analyzing chat logs', buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
USER_SESSIONS = Counter('app_user_sessions_total', 'Total number of user sessions created')

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

# Initialize session manager and rate limiter
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

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="chat_insight_session",
    max_age=24 * 60 * 60,  # 24 hours
    same_site="lax",  # Protect against CSRF
    path="/"  # Make cookie available for all paths
)

# Add CORS middleware with strict settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, replace with specific domains
)

# Ensure base directories exist
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
    REQUESTS_TOTAL.labels(endpoint='/').inc()
    try:
        # Get or create user session
        session = session_manager.get_user_session(request)
        
        # Check rate limit
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
    REQUESTS_TOTAL.labels(endpoint='/results').inc()
    # Check session validity
    if not session_manager.is_session_valid(request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Your session has expired. Please upload your chat file again."}
        )
    
    # Check rate limit
    if not rate_limiter.check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Extend session
    session_manager.extend_session(request)
    
    results_str = request.query_params.get("results", "{}")
    try:
        results = json.loads(results_str)
        results = standardize_results(results)
    except json.JSONDecodeError:
        results = {}
    return templates.TemplateResponse(request, "results.html", {"results": results})

@app.get("/visuals/{filename}")
async def serve_visuals(request: Request, filename: str):
    # Check session validity
    if not session_manager.is_session_valid(request):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Check rate limit
    if not rate_limiter.check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Extend session
    session_manager.extend_session(request)
    
    # Get user session
    session = session_manager.get_user_session(request)
    user_dirs = session_manager.get_user_directories(session["user_id"])
    
    # Check if file exists in user's directory
    file_path = user_dirs["visuals"] / filename
    if not file_path.exists():
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
    REQUESTS_TOTAL.labels(endpoint='/upload').inc()
    FILE_UPLOAD_COUNT.inc()
    
    try:
        # Check session validity
        if not session_manager.is_session_valid(request):
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(request):
            raise HTTPException(status_code=429, detail="Too many requests")
        
        # Extend session
        session_manager.extend_session(request)
        
        # Get user session and file paths
        session = session_manager.get_user_session(request)
        file_paths = session_manager.get_user_file_paths(session["user_id"], file.filename)
        
        if not allowed_file(file.filename):
            return templates.TemplateResponse(
                request, "index.html", {"error": "Invalid file format"}
            )

        txt_file_path, csv_file_path = handle_upload(file, file_paths)
        
        # Add logging to check file contents
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
        
        # Standardize results before sending to template
        standardized_results = standardize_results(analysis_results)
        
        # Get the relative paths for visualizations
        visualization_paths = {}
        for key, path in standardized_results.get("visualization_paths", {}).items():
            if isinstance(path, (str, Path)):
                # Convert absolute path to relative path from visuals directory
                try:
                    relative_path = Path(path).relative_to(file_paths["visuals"])
                    visualization_paths[key] = str(relative_path)
                    logger.info(f"Visualization path for {key}: {relative_path}")
                except ValueError:
                    logger.error(f"Could not get relative path for {key}: {path}")
                    visualization_paths[key] = str(path)
        
        # Log all visualization paths being passed to template
        logger.info("All visualization paths being passed to template:")
        for key, path in visualization_paths.items():
            logger.info(f"{key}: {path}")
        
        return templates.TemplateResponse(request, "results.html", {
            "results": standardized_results,
            "visualization_paths": visualization_paths
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
