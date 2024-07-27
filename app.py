import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from file_handler import allowed_file, handle_upload
from parser import parse_chat_log
from analyzer import analyze_chat_log

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid file format"})
    
    txt_file_path, csv_file_path = handle_upload(file)
    parse_chat_log(txt_file_path, csv_file_path)
    analysis_results = analyze_chat_log(csv_file_path)
    return templates.TemplateResponse("results.html", {"request": request, "results": analysis_results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)