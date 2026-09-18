from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .matching_engine import Resume, JobPosting, get_matching_engine, MatchMode
app = FastAPI(title="AI Job Matching Finder")


class MatchRequest(BaseModel):
    resume_text: str
    resume_skills: list[str] = []
    job_text: str
    job_required_skills: list[str] = []
    mode: MatchMode = "keyword"  # "keyword" | "ai" | "hybrid"


class MatchResponse(BaseModel):
    score: float
    explanation: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest):
    resume = Resume(text=request.resume_text, skills=request.resume_skills)
    job = JobPosting(text=request.job_text, required_skills=request.job_required_skills)

    try:
        engine = get_matching_engine(request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = engine.score(resume, job)
    return MatchResponse(score=result.score, explanation=result.explanation)


# Run locally with: uvicorn main:app --reload
