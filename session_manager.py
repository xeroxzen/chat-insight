import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import logging
from fastapi import Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.base_upload_dir = Path("uploads")
        self.base_visuals_dir = Path("static/visuals")
        self.session_timeout = timedelta(hours=24)  # 24 hour session timeout
        self.cleanup_threshold = timedelta(days=7)  # Clean up files older than 7 days
        
    def get_user_session(self, request: Request) -> Dict[str, Any]:
        """Get or create a user session."""
        if not request.session.get("user_id"):
            request.session["user_id"] = str(uuid.uuid4())
            request.session["created_at"] = datetime.now().isoformat()
            logger.info(f"Created new session for user: {request.session['user_id']}")
        return request.session
    
    def get_user_directories(self, user_id: str) -> Dict[str, Path]:
        """Get user-specific directories for uploads and visuals."""
        user_upload_dir = self.base_upload_dir / user_id
        user_visuals_dir = self.base_visuals_dir / user_id
        
        # Create directories if they don't exist
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        user_visuals_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "upload": user_upload_dir,
            "visuals": user_visuals_dir
        }
    
    def cleanup_old_sessions(self):
        """Clean up files from expired sessions."""
        try:
            current_time = datetime.now()
            
            # Clean up uploads
            for user_dir in self.base_upload_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                    
                # Check if directory is older than cleanup threshold
                dir_time = datetime.fromtimestamp(user_dir.stat().st_mtime)
                if current_time - dir_time > self.cleanup_threshold:
                    shutil.rmtree(user_dir)
                    logger.info(f"Cleaned up old upload directory: {user_dir}")
            
            # Clean up visuals
            for user_dir in self.base_visuals_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                    
                # Check if directory is older than cleanup threshold
                dir_time = datetime.fromtimestamp(user_dir.stat().st_mtime)
                if current_time - dir_time > self.cleanup_threshold:
                    shutil.rmtree(user_dir)
                    logger.info(f"Cleaned up old visuals directory: {user_dir}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def is_session_valid(self, request: Request) -> bool:
        """Check if the current session is valid."""
        if not request.session.get("created_at"):
            return False
            
        try:
            created_at = datetime.fromisoformat(request.session["created_at"])
            is_valid = datetime.now() - created_at < self.session_timeout
            if not is_valid:
                logger.info(f"Session expired for user: {request.session.get('user_id')}")
            return is_valid
        except (ValueError, TypeError) as e:
            logger.error(f"Error validating session: {str(e)}")
            return False
    
    def extend_session(self, request: Request):
        """Extend the session lifetime."""
        if request.session.get("created_at"):
            request.session["created_at"] = datetime.now().isoformat()
            logger.debug(f"Extended session for user: {request.session.get('user_id')}")
    
    def get_user_file_paths(self, user_id: str, filename: str) -> Dict[str, Path]:
        """Get user-specific file paths for upload and analysis."""
        user_dirs = self.get_user_directories(user_id)
        
        # Create unique filenames using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        
        return {
            "upload": user_dirs["upload"] / f"{base_name}_{timestamp}.zip",
            "extracted": user_dirs["upload"] / f"{base_name}_{timestamp}_extracted",
            "txt": user_dirs["upload"] / f"{base_name}_{timestamp}.txt",
            "csv": user_dirs["upload"] / f"{base_name}_{timestamp}.csv",
            "visuals": user_dirs["visuals"]
        } 