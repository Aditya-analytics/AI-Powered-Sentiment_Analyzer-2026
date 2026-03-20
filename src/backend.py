from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid, os, re
import groq
from dotenv import load_dotenv
from pipeline import Pipeline
from prompts import SYSTEM_PROMPT
from fastapi.responses import HTMLResponse
import io
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd


load_dotenv()

# ── Clients & constants ───────────────────────────────────────────────────────
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

# current - breaks if Render adds spaces
# ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

#fix
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
SAFE_ID = re.compile(r'^[a-f0-9\-]{36}$')   # strict UUID-only for file_id

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Review Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # set * in .env only for dev
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── LLM helper ────────────────────────────────────────────────────────────────
GROQ_ERRORS = {
    groq.RateLimitError:      "Rate limit hit — try again shortly.",
    groq.AuthenticationError: "Invalid API key.",
    groq.APIConnectionError:  "Network issue — could not reach Groq.",
}

def analyze_with_llm(reviews: list[str]) -> str | None:
    formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Analyze these negative reviews and provide solutions:\n\n{formatted}"},
            ],
            timeout=20.0,
        )
        return res.choices[0].message.content
    except groq.APIStatusError as e:
        raise HTTPException(502, f"Groq error ({e.status_code}): {e.message}")
    except tuple(GROQ_ERRORS) as e:
        raise HTTPException(503, GROQ_ERRORS[type(e)])
    except Exception as e:
        raise HTTPException(500, f"Unexpected LLM error: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
def serve_landing():
    with open(os.path.join(BASE_DIR, "landing.html"), encoding="utf-8") as f:
        return f.read()
    
@app.get("/app", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()
    
# In-memory store: { file_id: bytes }
csv_store: dict[str, bytes] = {}

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    top_n: int = Query(3, ge=1, le=50,description="Number of negative reviews to return"),
    column_name: str = Query(..., description="Choose name of column to get sentiment result"),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")
    
    file_id    = str(uuid.uuid4())
    input_path = f"input_{file_id}.csv"

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        pipe        = Pipeline(input_path)
        temp_df = pd.read_csv(input_path)
        if column_name not in temp_df.columns:
            raise HTTPException(400, f"Column '{column_name}' not found. Available columns: {list(temp_df.columns)}")
        df          = pipe.fit(column_name)
        top_reviews = pipe.top(df, top_n)

        # Add this — count sentiments
        sentiment_counts = df['sentiment'].value_counts().to_dict()

        # ✅ Store CSV bytes in memory — no output file written
        csv_store[file_id] = df.to_csv(index=False).encode("utf-8")

        # Extract just the text column for the LLM payload
        # Extract plain text for top reviews
        review_texts = [str(r[column_name]) for r in top_reviews]

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return {
        "message":              "Analysis completed",
        "top_negative_reviews": review_texts,
        "sentiment_counts":     sentiment_counts,   # ← new
        "download_url":         f"/download/{file_id}",
    }

@app.get("/download/{file_id}")
def download_file(file_id: str):
    if not SAFE_ID.match(file_id):
        raise HTTPException(400, "Invalid file ID.")

    csv_bytes = csv_store.pop(file_id, None)   # pop = delete after serving
    if csv_bytes is None:
        raise HTTPException(404, "File not found or already downloaded.")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=processed_reviews.csv"},
    )

class SolveRequest(BaseModel):
    reviews: list[str]

@app.post("/solve")
async def solve_reviews(body: SolveRequest):
    if not body.reviews:
        raise HTTPException(400, "No reviews provided.")
    llm_analysis = analyze_with_llm(body.reviews)
    return {"llm_analysis": llm_analysis}