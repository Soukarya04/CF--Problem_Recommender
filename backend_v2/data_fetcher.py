import time
import math
import requests
from collections import defaultdict
from bisect import bisect_left, bisect_right

from models import Problem, Submission, RatingEvent

# -------------------------
# CONSTANTS
# -------------------------

BASE_URL  = "https://codeforces.com/api"
CACHE_TTL = 3600    # seconds — cached API responses expire after 1 hour

# -------------------------
# TTL CACHE
# -------------------------

_cache: dict[str, tuple[dict, float]] = {}


def cached_get(url: str) -> dict:
    """
    GET a URL and cache the response for CACHE_TTL seconds.
    On a repeated run within the TTL window the API is not hit again.
    Raises requests.HTTPError on non-2xx responses.
    """
    now = time.time()

    if url in _cache:
        data, ts = _cache[url]
        if now - ts < CACHE_TTL:
            return data

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    _cache[url] = (data, now)
    return data


# -------------------------
# API FETCHERS
# -------------------------

def get_user_submissions(handle: str) -> list[Submission] | None:
    """
    Fetch all submissions for a user and return them as Submission dataclasses.
    Returns None if the API call fails.
    """
    url = f"{BASE_URL}/user.status?handle={handle}"

    try:
        data = cached_get(url)
    except (requests.RequestException, ValueError) as e:
        print(f"[ERROR] Failed to fetch submissions for '{handle}': {e}")
        return None

    if data.get("status") != "OK":
        print(f"[ERROR] API returned status: {data.get('status')}")
        return None

    submissions = []
    for raw in data["result"]:
        prob = raw["problem"]

        # skip problems without a contest ID (rare but happens)
        if "contestId" not in prob:
            continue

        problem = Problem(
            contest_id=prob["contestId"],
            index=prob["index"],
            rating=prob.get("rating", 0),
            tags=prob.get("tags", []),
            name=prob.get("name", ""),
        )
        submissions.append(Submission(
            problem=problem,
            verdict=raw.get("verdict", "UNKNOWN"),
            timestamp=raw.get("creationTimeSeconds", 0),
        ))

    return submissions


def get_all_problems() -> list[Problem] | None:
    """
    Fetch the full Codeforces problemset.
    Returns None if the API call fails.
    """
    url = f"{BASE_URL}/problemset.problems"

    try:
        data = cached_get(url)
    except (requests.RequestException, ValueError) as e:
        print(f"[ERROR] Failed to fetch problemset: {e}")
        return None

    if data.get("status") != "OK":
        print(f"[ERROR] API returned status: {data.get('status')}")
        return None

    problems = []
    for prob in data["result"]["problems"]:
        if "rating" not in prob or "contestId" not in prob:
            continue
        problems.append(Problem(
            contest_id=prob["contestId"],
            index=prob["index"],
            rating=prob["rating"],
            tags=prob.get("tags", []),
            name=prob.get("name", ""),
        ))

    return problems


def get_user_rating(handle: str) -> list[RatingEvent] | None:
    """
    Fetch the rating history for a user.
    Returns None if the API call fails.
    """
    url = f"{BASE_URL}/user.rating?handle={handle}"

    try:
        data = cached_get(url)
    except (requests.RequestException, ValueError) as e:
        print(f"[ERROR] Failed to fetch rating history for '{handle}': {e}")
        return None

    if data.get("status") != "OK":
        print(f"[ERROR] API returned status: {data.get('status')}")
        return None

    return [
        RatingEvent(
            contest_id=entry["contestId"],
            new_rating=entry["newRating"],
            timestamp=entry["ratingUpdateTimeSeconds"],
        )
        for entry in data["result"]
    ]


# -------------------------
# SUBMISSION HELPERS
# -------------------------

def extract_solved_problems(submissions: list[Submission]) -> set[str]:
    """Return a set of problem IDs the user has solved at least once."""
    return {
        sub.problem.pid
        for sub in submissions
        if sub.verdict == "OK"
    }


def extract_problem_details(submissions: list[Submission]) -> dict[str, Problem]:
    """
    Return a dict of pid -> Problem for every problem the user has touched.
    If a problem appears multiple times (multiple attempts), the last entry wins
    but since problem metadata is static that doesn't matter.
    """
    return {sub.problem.pid: sub.problem for sub in submissions}


def get_recent_solved(submissions: list[Submission], limit: int = 50) -> list[str]:
    """
    Return up to `limit` unique problem IDs the user solved most recently,
    ordered newest first.
    """
    seen: set[str] = set()
    recent: list[str] = []

    for sub in submissions:
        if sub.verdict != "OK":
            continue
        pid = sub.problem.pid
        if pid not in seen:
            seen.add(pid)
            recent.append(pid)
        if len(recent) >= limit:
            break

    return recent


# -------------------------
# RATING HELPERS
# -------------------------

def compute_recent_cf_rating(rating_history: list[RatingEvent], days: int = 30) -> float:
    """
    Average of contest ratings from the last `days` days.
    Falls back to the most recent rating if no contests in that window.
    """
    if not rating_history:
        return 0.0

    now   = int(time.time())
    cutoff = now - days * 24 * 3600

    recent = [e.new_rating for e in rating_history if e.timestamp >= cutoff]

    if not recent:
        return float(rating_history[-1].new_rating)

    return sum(recent) / len(recent)


# -------------------------
# PROBLEMSET STRUCTURES
# -------------------------

def build_problemset(problems: list[Problem]) -> dict[str, Problem]:
    """pid -> Problem lookup table."""
    return {p.pid: p for p in problems}


def build_tag_index(problemset: dict[str, Problem]) -> dict[str, list[str]]:
    """
    Inverted index: tag -> [pid, pid, ...]

    Used to jump from a tag directly to all problems that have it,
    replacing the O(n * m) brute-force similarity scan with a fast
    set-union over tag buckets.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for pid, prob in problemset.items():
        for tag in prob.tags:
            index[tag].append(pid)
    return dict(index)


def build_rating_index(problemset: dict[str, Problem]) -> tuple[list[int], dict[int, list[str]]]:
    """
    Two structures for O(log n) rating-range queries:
      - sorted_ratings : sorted list of all unique ratings
      - rating_to_pids : rating -> [pid, ...]

    Usage:
        lo, hi = 1400, 1600
        i = bisect_left(sorted_ratings, lo)
        j = bisect_right(sorted_ratings, hi)
        pids_in_range = [pid for r in sorted_ratings[i:j] for pid in rating_to_pids[r]]
    """
    rating_to_pids: dict[int, list[str]] = defaultdict(list)
    for pid, prob in problemset.items():
        rating_to_pids[prob.rating].append(pid)

    sorted_ratings = sorted(rating_to_pids.keys())
    return sorted_ratings, dict(rating_to_pids)


def get_problems_in_rating_range(
    lo: int,
    hi: int,
    sorted_ratings: list[int],
    rating_to_pids: dict[int, list[str]],
) -> list[str]:
    """Binary search to fetch all pids whose rating falls in [lo, hi]."""
    i = bisect_left(sorted_ratings,  lo)
    j = bisect_right(sorted_ratings, hi)
    result = []
    for r in sorted_ratings[i:j]:
        result.extend(rating_to_pids[r])
    return result


# -------------------------
# CANDIDATE GENERATION
# -------------------------

def generate_candidates(
    recent_solved: list[str],
    problemset: dict[str, Problem],
    tag_index: dict[str, list[str]],
) -> set[str]:
    """
    Build a candidate pool using the inverted tag index.

    For each recently solved problem, look up all problems sharing at
    least one tag. This is O(recent * avg_tags * bucket_size) instead
    of the previous O(recent * |problemset|) brute-force scan.
    """
    candidates: set[str] = set()
    recent_set  = set(recent_solved)

    for pid in recent_solved:
        if pid not in problemset:
            continue
        for tag in problemset[pid].tags:
            candidates.update(tag_index.get(tag, []))

    # don't recommend things already in the recent solved window
    candidates -= recent_set
    return candidates


# -------------------------
# TF-IDF TAG WEIGHTS
# -------------------------

def build_tag_idf(problemset: dict[str, Problem]) -> dict[str, float]:
    """
    IDF (inverse document frequency) for each tag.

    Common tags like 'implementation' or 'greedy' appear in thousands of
    problems and are not very discriminating — their IDF will be low.
    Rare tags like 'centroid decomposition' appear in few problems — their
    IDF will be high, so they contribute more to scoring.

    Formula: idf(tag) = log(N / df(tag))
    """
    N = len(problemset)
    tag_doc_count: dict[str, int] = defaultdict(int)

    for prob in problemset.values():
        for tag in set(prob.tags):   # set() so we count each tag once per problem
            tag_doc_count[tag] += 1

    return {
        tag: math.log(N / count)
        for tag, count in tag_doc_count.items()
        if count > 0
    }
