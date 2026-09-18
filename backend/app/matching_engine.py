"""
Matching engine interface for the job-match-finder project.

The idea: define one abstract interface (MatchingEngine) that both the
keyword-based implementation (phase 1) and the embeddings-based
implementation (phase 2) satisfy. The rest of the app (API routes,
scoring pipeline, etc.) only ever talks to the interface, so swapping
phase 1 for phase 2 later is a one-line change, not a rewrite.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re


@dataclass
class Resume:
    text: str
    skills: list[str] = field(default_factory=list)


@dataclass
class JobPosting:
    text: str
    required_skills: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    score: float  # normalized 0.0 - 1.0
    explanation: str  # human-readable reason for the score


class MatchingEngine(ABC):
    """Common interface every matching implementation must follow."""

    @abstractmethod
    def score(self, resume: Resume, job: JobPosting) -> MatchResult:
        ...


# ---------------------------------------------------------------------
# Phase 1: keyword-based implementation
# ---------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
    "is", "are", "on", "at", "as", "by",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


class KeywordMatchingEngine(MatchingEngine):
    """Jaccard similarity over resume text/skills vs. job text/skills."""

    def score(self, resume: Resume, job: JobPosting) -> MatchResult:
        resume_terms = _tokenize(resume.text) | {s.lower() for s in resume.skills}
        job_terms = _tokenize(job.text) | {s.lower() for s in job.required_skills}

        if not job_terms:
            return MatchResult(score=0.0, explanation="Job posting has no extractable terms.")

        overlap = resume_terms & job_terms
        jaccard = len(overlap) / len(resume_terms | job_terms) if (resume_terms | job_terms) else 0.0

        # Weight required-skill hits more heavily than general text overlap.
        required = {s.lower() for s in job.required_skills}
        required_hits = required & resume_terms
        required_ratio = len(required_hits) / len(required) if required else 0.0

        final_score = 0.4 * jaccard + 0.6 * required_ratio

        explanation = (
            f"Matched {len(required_hits)}/{len(required)} required skills "
            f"({', '.join(sorted(required_hits)) or 'none'}); "
            f"general term overlap: {len(overlap)} shared terms."
        )
        return MatchResult(score=round(final_score, 3), explanation=explanation)


# ---------------------------------------------------------------------
# Phase 2: embeddings-based implementation (stub — fill in later)
# ---------------------------------------------------------------------

class EmbeddingMatchingEngine(MatchingEngine):
    """
    Semantic similarity using vector embeddings.

    To implement:
      1. Pick an embeddings provider (e.g. an embeddings API, or a
         local sentence-transformers model).
      2. Embed resume.text and job.text into vectors.
      3. Compute cosine similarity between the two vectors.
      4. Optionally blend with a KeywordMatchingEngine score for a
         hybrid result (e.g. 0.6 * embedding_score + 0.4 * keyword_score).
    """

    def __init__(self, embed_fn):
        # embed_fn: Callable[[str], list[float]] — injected so this class
        # doesn't hardcode a specific embeddings provider.
        self._embed_fn = embed_fn

    def score(self, resume: Resume, job: JobPosting) -> MatchResult:
        raise NotImplementedError("Wire up an embeddings provider here in phase 2.")


# ---------------------------------------------------------------------
# Hybrid implementation — runs both and blends the scores
# ---------------------------------------------------------------------

class HybridMatchingEngine(MatchingEngine):
    """Blends keyword and embedding scores. Falls back to keyword-only
    if the embedding engine isn't ready yet (e.g. no API key configured)."""

    def __init__(self, keyword_engine: KeywordMatchingEngine, embedding_engine: EmbeddingMatchingEngine, embedding_weight: float = 0.6):
        self._keyword_engine = keyword_engine
        self._embedding_engine = embedding_engine
        self._embedding_weight = embedding_weight

    def score(self, resume: Resume, job: JobPosting) -> MatchResult:
        keyword_result = self._keyword_engine.score(resume, job)
        try:
            embedding_result = self._embedding_engine.score(resume, job)
        except NotImplementedError:
            return keyword_result  # graceful fallback until phase 2 is wired up

        w = self._embedding_weight
        blended = w * embedding_result.score + (1 - w) * keyword_result.score
        explanation = f"Hybrid: {embedding_result.explanation} | {keyword_result.explanation}"
        return MatchResult(score=round(blended, 3), explanation=explanation)


# ---------------------------------------------------------------------
# Switcher — lets the caller (e.g. an API request param, or a user
# setting) pick which engine to use at runtime, without touching any
# other code.
# ---------------------------------------------------------------------

MatchMode = str  # "keyword" | "ai" | "hybrid"


def get_matching_engine(mode: MatchMode = "keyword", embed_fn=None) -> MatchingEngine:
    """
    Factory: returns the engine that matches the user's chosen mode.

    Example (FastAPI route):
        @app.get("/match")
        def match(job_id: str, mode: MatchMode = "keyword"):
            engine = get_matching_engine(mode, embed_fn=my_embed_fn)
            return engine.score(resume, job)
    """
    keyword_engine = KeywordMatchingEngine()

    if mode == "keyword":
        return keyword_engine

    if mode in ("ai", "embeddings"):
        if embed_fn is None:
            raise ValueError("mode='ai' requires an embed_fn (phase 2 provider).")
        return EmbeddingMatchingEngine(embed_fn)

    if mode == "hybrid":
        if embed_fn is None:
            raise ValueError("mode='hybrid' requires an embed_fn (phase 2 provider).")
        return HybridMatchingEngine(keyword_engine, EmbeddingMatchingEngine(embed_fn))

    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    resume = Resume(
        text="Backend developer experienced with Node.js, Express, REST APIs.",
        skills=["Node.js", "Express", "PostgreSQL"],
    )
    job = JobPosting(
        text="Looking for a backend engineer familiar with Python and FastAPI.",
        required_skills=["Python", "FastAPI"],
    )

    # This is the "switch": the user (or the frontend) decides the mode.
    user_chosen_mode: MatchMode = "keyword"
    engine = get_matching_engine(user_chosen_mode)

    result = engine.score(resume, job)
    print(result)
