from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Dict
import shutil
import os
from uuid import uuid4
from models.traditional_model import compute_tfidf_score
from models.gnn_model import gnn_score
from models.rl_model import reinforcement_learning_score
from utils.text_extraction import extract_text

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mock database
jobs: Dict[str, Dict] = {}
cvs: Dict[str, Dict] = {}

@app.post("/add-job")
def add_job(job_title: str = Form(...), job_description: str = Form(...)):
    job_id = str(uuid4())
    jobs[job_id] = {"title": job_title, "description": job_description}
    
    print(f"Job added: {job_id} -> {job_title}")
    
    return {"job_id": job_id, "message": "Job added successfully."}

@app.post("/upload-cvs")
def upload_cvs(files: List[UploadFile] = File(...)):
    uploaded_files = []
    
    for file in files:
        if file.filename.endswith((".pdf", ".docx")):
            file_id = str(uuid4())
            file_path = os.path.join(UPLOAD_DIR, file_id + os.path.splitext(file.filename)[1])
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            text = extract_text(file_path)
            cvs[file_id] = {"path": file_path, "text": text}
            uploaded_files.append({"file_id": file_id, "filename": file.filename})
            
            print(f"Uploaded: {file.filename} -> {file_id}")

        else:
            print("Invalid file format:", file.filename)
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF and DOCX allowed.")
    
    print("Current uploaded CVs:", cvs.keys())
    
    return {"message": "Files uploaded successfully.", "files": uploaded_files}

@app.post("/score-cvs")
def score_cvs(job_id: str):
    print("Incoming job_id:", job_id)
    print("Existing jobs:", jobs)
    
    if job_id not in jobs:
        print("Job not found:", job_id)
        raise HTTPException(status_code=404, detail="Job not found.")
    
    job_text = jobs[job_id]["description"]
    print("Job description:", job_text)
    
    cv_texts = {cv_id: cvs[cv_id]["text"] for cv_id in cvs}
    print("Extracted CV texts:", cv_texts)

    if not cv_texts:
        print("No CVs uploaded yet!")
        raise HTTPException(status_code=400, detail="No CVs uploaded.")

    # Scoring models
    print("Running TF-IDF scoring...")
    tfidf_scores = tfidf_scoring(job_text, cv_texts)
    print("TF-IDF Scores:", tfidf_scores)

    print("Running GNN scoring...")
    gnn_scores, graph_image = gnn_scoring(job_text, cv_texts)
    print("GNN Scores:", gnn_scores)

    print("Running RL scoring...")
    rl_scores = rl_scoring(job_text, cv_texts)
    print("RL Scores:", rl_scores)

    # Final combined scores
    final_scores = {
        cv_id: (0.4 * tfidf_scores[cv_id]) + (0.3 * gnn_scores[cv_id]) + (0.3 * rl_scores[cv_id])
        for cv_id in tfidf_scores
    }
    
    print("Final Scores:", final_scores)

    ranked_cvs = sorted(final_scores.keys(), key=lambda cv_id: final_scores[cv_id], reverse=True)

    return {
        "ranked_candidates": [
            {
                "cv_id": cv_id,
                "Final Score": round(final_scores[cv_id], 4),
                "TF-IDF Score": round(tfidf_scores[cv_id], 4),
                "GNN Score": round(gnn_scores[cv_id], 4),
                "RL Score": round(rl_scores[cv_id], 4)
            } for cv_id in ranked_cvs
        ],
        "graph_visualization": graph_image
    }
