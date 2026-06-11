# CF Problem Recommender

A personalized Codeforces problem recommendation engine that uses tag-based user profiling, TF-IDF scoring, and a UCB1 multi-armed bandit to suggest the right problems at the right difficulty — balancing exploitation of known weaknesses with exploration of unseen problem types.

---

## How It Works

### 1. User Profiling
Fetches the user's full submission history from the Codeforces API and builds a profile:
- **Tag accuracy** — for each tag, computes a weighted fraction of problems solved, where older solves decay exponentially (half-life: 90 days). Recent practice matters more than problems solved two years ago.
- **Tag attempts** — tracks how many unique problems per tag the user has encountered.
- **Unsolved tags** — tags that exist in the full problemset but the user has never attempted.

### 2. Candidate Generation
Instead of scanning the full problemset (~10,000+ problems) for every request, an inverted tag index is built at startup. For each of the user's 50 most recently solved problems, the engine looks up all problems sharing at least one tag — generating a focused candidate pool in O(recent × avg\_tags × bucket\_size).

### 3. Scoring
Each candidate problem is scored on two components:

- **Difficulty match** — problems close to the user's target rating (current rating − 100) score higher. Normalized over a ±500 window.
- **Weak tag boost** — tags the user struggles with score higher, weighted by IDF so rare/specific tags (e.g. `centroid decomposition`) outweigh common ones (e.g. `greedy`, `implementation`).

Scores are normalized and a 20% recency bonus is added based on how new the problem's contest is.

### 4. UCB1 Bandit Selection
The top scored problems are passed to a UCB1 multi-armed bandit for final selection:

- **90% exploit** — picks from the scored pool using UCB1, balancing high-scoring problems with ones that haven't been recommended much yet.
- **10% explore** — randomly samples problems in the right difficulty range that are completely outside the scored pool, preventing the recommender from getting stuck in a tag bubble.

Bandit state (counts, average rewards) is persisted per user to disk as JSON, so recommendations improve over time as feedback is provided.

---

## Project Structure

```
cf_recommender/
├── backend_v2/
│   ├── main.py          # CLI entry point
│   ├── api.py           # FastAPI server with REST endpoints
│   ├── recommender.py   # Scoring, UCB1 bandit, user profile builder
│   ├── data_fetcher.py  # Codeforces API calls, indexes, candidate generation
│   └── models.py        # Dataclasses: Problem, Submission, UserProfile, etc.
└── frontend/
    └── cf_recommender_frontend.html   # Simple HTML frontend
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recommend/{handle}?k=10` | Get top-k recommended problems for a user |
| POST | `/feedback` | Submit solve/skip feedback to update the bandit |
| GET | `/profile/{handle}` | Get tag accuracy profile and weakest tags |

### Example: Get Recommendations
```
GET /recommend/tourist?k=10
```

### Example: Submit Feedback
```json
POST /feedback
{
  "handle": "tourist",
  "pid": "1536-F",
  "solved": true
}
```

---

## Running Locally

### CLI
```bash
cd backend_v2
pip install requests
python main.py <codeforces_handle> --top-k 10
```

### API Server
```bash
cd backend_v2
pip install fastapi uvicorn requests
uvicorn api:app --reload
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Tech Stack

- **Python 3.12+**
- **FastAPI** — REST API
- **Codeforces API** — live problem and submission data
- **Algorithms** — UCB1 bandit, TF-IDF, exponential decay, inverted index, binary search rating range queries
