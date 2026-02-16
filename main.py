import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from text_cleaner import extract_and_clean

# Create app
app = FastAPI(title="Document Cleaner")

# Templates
templates = Jinja2Templates(directory="templates")

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------------
# Home Page (UI)
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------
# Upload & Process File
# -------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    # Save uploaded file
    file_id = str(uuid.uuid4())
    upload_path = f"uploads/{file_id}_{file.filename}"

    with open(upload_path, "wb") as buffer:
        buffer.write(await file.read())

    # Output file path
    output_path = f"outputs/cleaned_{file_id}.txt"

    # Run your function
    cleaned_text = extract_and_clean(
        file_path=upload_path,
        output_path=output_path,
        remove_pii_flag=True
    )

    # Show only first 2000 chars in UI
    preview = cleaned_text[:2000]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "preview": preview,
            "download_link": f"/download/{file_id}"
        }
    )


# -------------------------
# Download Cleaned File
# -------------------------
@app.get("/download/{file_id}")
def download_file(file_id: str):
    file_path = f"outputs/cleaned_{file_id}.txt"
    return FileResponse(
        path=file_path,
        filename="cleaned_output.txt",
        media_type="text/plain"
    )
