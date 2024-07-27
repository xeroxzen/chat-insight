import os
import re
import zipfile
from fastapi import UploadFile

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_upload(file: UploadFile):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())
    
    # Get the original filename without extension and remove emojis
    original_filename = os.path.splitext(file.filename)[0]
    original_filename = re.sub(r'[^\w\-_\. ]', '', original_filename)
    
    extract_path = os.path.join(UPLOAD_FOLDER, 'extracted')
    with zipfile.ZipFile(file_location, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Find the extracted .txt file
    txt_file = None
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.txt'):
                txt_file = os.path.join(root, file)
                break
        if txt_file:
            break
    
    if not txt_file:
        raise FileNotFoundError("No .txt file found in the extracted zip")
    
    txt_file_path = txt_file
    csv_file_path = os.path.join(UPLOAD_FOLDER, f'{original_filename}_chat.csv')
    
    return txt_file_path, csv_file_path