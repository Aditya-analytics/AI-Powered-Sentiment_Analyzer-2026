from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
import pandas as pd
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware
from pipeline import Pipeline   # your pipeline file

app = FastAPI()

# 🔥 CORS setup
origins = [
    "http://localhost:3000",   # React default
    "http://127.0.0.1:3000",
    "*"   # ⚠️ allow all (for development only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    top_n: int = Query(3, description="Number of negative reviews to return"),
    column_name : str = Query(..., description="Choose name of column to get sentiment result"),
):
    # 🔹 Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    input_path = f"input_{file_id}.csv"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 🔹 Run your pipeline
    pipe = Pipeline(input_path)

    df = pipe.fit(column_name)

    # 🔹 Get top negative reviews
    top_reviews = pipe.top(df, top_n)

    # 🔹 Save processed output
    output_path = f"processed_{file_id}.csv"
    df.to_csv(output_path, index=False)

    # 🔹 Clean input file (optional)
    os.remove(input_path)

    return {
        "message": "Analysis completed",
        "top_negative_reviews": top_reviews,
        "download_url": f"/download/{file_id}"
    }

@app.get("/download/{file_id}")
def download_file(file_id: str):
    file_path = f"processed_{file_id}.csv"

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename="processed_reviews.csv"
    )

# Optional health check
@app.get("/")
def home():
    return {"status": "API running"}