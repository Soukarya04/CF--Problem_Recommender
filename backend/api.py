import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_fetcher import (
    get_user_submissions,
    get_all_problems,
    get_user_rating,
    extract_solved_problems,
    get_recent_solved,
    compute_recent_cf_rating,
    build_problemset,
    build_tag_index,
    build_rating_index,
    generate_candidates,
    build_tag_idf,
)
from recommender import (
    build_user_profile,
    recommend_problems,
    UCB1Bandit,
)
from models import TARGET_RATING_OFFSET

# -------------------------
# BANDIT PERSISTENCE
# -------------------------

BANDIT_DIR = "bandit_states"
os.makedirs(BANDIT_DIR, exist_ok=True)


def load_bandit(handle: str) -> UCB1Bandit:
    """Load UCB1 state for a user from disk, or return a fresh one."""
    path = os.path.join(BANDIT_DIR, f"{handle}.json")
    bandit = UCB1Bandit()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        bandit.counts = data["counts"]
        bandit.values = data["values"]
        bandit.t      = data["t"]
    return bandit


def save_bandit(handle: str, bandit: UCB1Bandit):
    """Persist UCB1 state for a user to disk."""
    path = os.path.join(BANDIT_DIR, f"{handle}.json")
    with open(path, "w") as f:
        json.dump({
            "counts": bandit.counts,
            "values": bandit.values,
            "t":      bandit.t,
        }, f)


# -------------------------
# STARTUP — load problemset once
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the full Codeforces problemset once when the server starts.
    All requests reuse this — no repeated API calls for the problemset.
    """
    print("Loading Codeforces problemset...")
    problems_raw = get_all_problems()
    if problems_raw is None:
        raise RuntimeError("Failed to fetch problemset on startup.")

    app.state.problemset                             = build_problemset(problems_raw)
    app.state.tag_index                              = build_tag_index(app.state.problemset)
    app.state.tag_idf                                = build_tag_idf(app.state.problemset)
    app.state.sorted_ratings, app.state.rating_to_pids = build_rating_index(app.state.problemset)

    print(f"Loaded {len(app.state.problemset)} problems, {len(app.state.tag_index)} tags.")
    yield


# -------------------------
# APP
# -------------------------

app = FastAPI(
    title="Codeforces Problem Recommender",
    description="Recommends Codeforces problems based on user history, tag weakness, and UCB1 bandit.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this when you have a real frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# SCHEMAS
# -------------------------

class RecommendedProblem(BaseModel):
    pid:    str
    name:   str
    rating: int
    tags:   list[str]
    score:  float
    mode:   str         # "exploit" or "explore"
    url:    str


class FeedbackRequest(BaseModel):
    handle: str
    pid:    str
    solved: bool        # True = user solved it, False = skipped/failed


class TagStat(BaseModel):
    tag:      str
    accuracy: float
    attempts: int


class ProfileResponse(BaseModel):
    handle:        str
    current_rating: float
    target_rating:  float
    weakest_tags:   list[TagStat]
    unsolved_tag_count: int
    unsolved_tags:      list[str]


# -------------------------
# HELPERS
# -------------------------

def pid_to_url(pid: str) -> str:
    """Convert a pid like '1536-F' to a Codeforces problem URL."""
    contest_id, index = pid.split("-", 1)
    return f"https://codeforces.com/problemset/problem/{contest_id}/{index}"


def build_unsolved(handle: str, app_state) -> tuple:
    """
    Shared logic to fetch submissions, build profile, and get unsolved candidates.
    Returns (submissions, solved, profile, unsolved) or raises HTTPException.
    """
    submissions = get_user_submissions(handle)
    if submissions is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch submissions for '{handle}'. Check the handle.")

    rating_history = get_user_rating(handle)
    if rating_history is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch rating history for '{handle}'.")

    solved         = extract_solved_problems(submissions)
    current_rating = compute_recent_cf_rating(rating_history)
    profile        = build_user_profile(submissions, solved, app_state.problemset, current_rating)
    profile.handle = handle

    recent_solved = get_recent_solved(submissions, limit=50)
    candidates    = generate_candidates(recent_solved, app_state.problemset, app_state.tag_index)
    unsolved      = {
        pid: app_state.problemset[pid]
        for pid in candidates
        if pid not in solved
    }

    if not unsolved:
        raise HTTPException(status_code=404, detail="No unsolved candidates found for this user.")

    return submissions, solved, profile, unsolved


# -------------------------
# ENDPOINTS
# -------------------------

@app.get("/recommend/{handle}", response_model=list[RecommendedProblem])
def recommend(handle: str, k: int = 10):
    """
    Get top-k recommended problems for a Codeforces user.
    - 90% exploit: best scored problems via UCB1
    - 10% explore: random problems in difficulty range outside the scored pool
    """
    _, _, profile, unsolved = build_unsolved(handle, app.state)

    scored = recommend_problems(unsolved, profile, app.state.tag_idf, k=k)

    bandit = load_bandit(handle)
    final  = bandit.recommend(
        scored=scored,
        unsolved=unsolved,
        profile=profile,
        sorted_ratings=app.state.sorted_ratings,
        rating_to_pids=app.state.rating_to_pids,
        k=k,
    )
    save_bandit(handle, bandit)

    return [
        RecommendedProblem(
            pid=sp.pid,
            name=sp.problem.name,
            rating=sp.problem.rating,
            tags=sp.problem.tags,
            score=round(sp.score, 4),
            mode=sp.mode,
            url=pid_to_url(sp.pid),
        )
        for sp in final
    ]


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Submit feedback for a recommended problem.
    Updates the UCB1 bandit state so future recommendations improve.

    solved=true  -> reward 1.0 (user solved it)
    solved=false -> reward 0.0 (user skipped or failed)
    """
    bandit = load_bandit(req.handle)
    bandit.update(req.pid, 1.0 if req.solved else 0.0)
    save_bandit(req.handle, bandit)
    return {"status": "ok", "pid": req.pid, "reward": 1.0 if req.solved else 0.0}


@app.get("/profile/{handle}", response_model=ProfileResponse)
def profile(handle: str):
    """
    Get a user's tag accuracy profile.
    Shows their weakest tags (lowest accuracy) and count of tags never attempted.
    """
    _, _, user_profile, _ = build_unsolved(handle, app.state)

    sorted_tags = sorted(user_profile.tag_accuracy.items(), key=lambda x: x[1])
    weakest = [
        TagStat(
            tag=tag,
            accuracy=round(acc, 4),
            attempts=user_profile.tag_attempts.get(tag, 0),
        )
        for tag, acc in sorted_tags[:10]
    ]

    return ProfileResponse(
        handle=handle,
        current_rating=round(user_profile.current_rating, 1),
        target_rating=round(user_profile.target_rating, 1),
        weakest_tags=weakest,
        unsolved_tag_count=len(user_profile.unsolved_tags),
        unsolved_tags=user_profile.unsolved_tags,
    )