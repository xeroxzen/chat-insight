import os
import re
import zipfile
from fastapi import UploadFile
from pathlib import Path
from typing import Dict, Tuple
import logging
import shutil

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename: str) -> bool:
    """Check if the file is allowed based on its extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_upload(file: UploadFile, file_paths: Dict[str, Path]) -> Tuple[Path, Path]:
    """
    Handle the upload of a file to user-specific directories.
    
    Args:
        file: The uploaded file
        file_paths: Dictionary containing user-specific file paths
        
    Returns:
        Tuple of (txt_file_path, csv_file_path)
    """
    try:
        # Save the uploaded zip file
        with open(file_paths["upload"], "wb") as buffer:
            buffer.write(file.file.read())
        
        # Extract the zip file
        with zipfile.ZipFile(file_paths["upload"], 'r') as zip_ref:
            zip_ref.extractall(file_paths["extracted"])
        
        # Find the extracted .txt file
        txt_file = None
        for root, dirs, files in os.walk(file_paths["extracted"]):
            for file in files:
                if file.endswith('.txt'):
                    txt_file = Path(root) / file
                    break
            if txt_file:
                break
        
        if not txt_file:
            raise FileNotFoundError("No .txt file found in the extracted zip")
        
        # Move the txt file to user's directory
        shutil.move(str(txt_file), str(file_paths["txt"]))
        
        # Clean up extracted directory
        shutil.rmtree(file_paths["extracted"])
        
        return file_paths["txt"], file_paths["csv"]
        
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        # Clean up any partially created files
        for path in file_paths.values():
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        raise