import math
import time
import heapq
import random
from collections import defaultdict

from models import (
    Problem,
    Submission,
    UserProfile,
    ScoredProblem,
    TARGET_RATING_OFFSET,
    RATING_SCALE,
    RECENCY_HALF_LIFE_DAYS,
)

# -------------------------
# DECAY WEIGHT
# -------------------------

def decay_weight(submission_timestamp: int, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """
    Exponential decay weight for a submission based on how long ago it happened.

    A solve from today       -> weight 1.0
    A solve from 60 days ago -> weight 0.5   (one half-life)
    A solve from 120 days ago-> weight 0.25  (two half-lives)

    This means recent practice has more influence on your tag accuracy
    than problems you solved two years ago and may be rusty on.
    """
    now      = time.time()
    age_days = (now - submission_timestamp) / 86400
    return math.exp(-0.693 * age_days / half_life_days)   # 0.693 = ln(2)


# -------------------------
# USER PROFILE BUILDER
# -------------------------

def build_user_profile(
    submissions: list[Submission],
    solved: set[str],
    problemset: dict[str, Problem],
    user_rating: float,
) -> UserProfile:
    """
    Build a UserProfile from the user's submission history.

    Key improvements over v1:
    - Solved problems are weighted by exponential decay (recent = more weight)
    - Tags with 0 accuracy (attempted but never solved) are kept, not dropped —
      these are actually your weakest tags and should be boosted the most
    - tag_attempts tracks how many times you've seen each tag (for UCB1 use)
    """
    handle = ""  # populated by caller if needed

    # count wrong submissions per problem
    wrong_counts:        dict[str, int]   = defaultdict(int)
    problem_verdicts:    dict[str, str]   = {}
    problem_timestamps:  dict[str, int]   = {}

    for sub in submissions:
        pid = sub.problem.pid
        if sub.verdict == "OK":
            problem_verdicts[pid]   = "OK"
            problem_timestamps[pid] = sub.timestamp
        else:
            wrong_counts[pid] += 1
            if pid not in problem_verdicts:
                problem_verdicts[pid] = sub.verdict

    # weighted solved count and raw attempt count per tag
    tag_weighted_solved: dict[str, float] = defaultdict(float)
    tag_attempts:        dict[str, int]   = defaultdict(int)

    for pid, verdict in problem_verdicts.items():
        prob = problemset.get(pid)
        if prob is None:
            continue
        tags = prob.tags

        for tag in tags:
            tag_attempts[tag] += 1   # unique problems only, not submissions

            if verdict == "OK":
                wrongs  = wrong_counts.get(pid, 0)
                penalty = 1 / (1 + 0.3 * wrongs)   # 0.3 penalty for wrong submissions
                w       = decay_weight(problem_timestamps[pid]) * penalty
                tag_weighted_solved[tag] += w

    tag_accuracy: dict[str, float] = {}
    solved_tags:  set[str]         = set()

    for tag, attempts in tag_attempts.items():
        weighted_solved = tag_weighted_solved.get(tag, 0.0)
        acc = weighted_solved / attempts  # in [0, 1]
        if acc == 0.0:
            pass # goes into unsolved tags
        else:
            tag_accuracy[tag] = acc
            solved_tags.add(tag)

    # tags that exist in the full problemset but user has never even attempted
    all_tags = {tag for prob in problemset.values() for tag in prob.tags}
    unsolved_tags = sorted(all_tags - solved_tags)

    return UserProfile(
        handle=handle,
        current_rating=user_rating,
        tag_accuracy=tag_accuracy,
        tag_attempts=dict(tag_attempts),
        unsolved_tags=unsolved_tags,
    )


# -------------------------
# PROBLEM SCORING
# -------------------------

def score_problem(
    problem: Problem,
    profile: UserProfile,
    tag_idf: dict[str, float],
) -> float:
    """
    Score a candidate problem for a user.

    Components:
    1. Difficulty match  — problems close to target rating score higher
    2. Weak tag boost    — tags the user is weak at score higher
                          multiplied by IDF so rare/specific tags matter more
                          than common ones like 'greedy' or 'implementation'
    """
    target_rating = profile.target_rating
    tag_accuracy  = profile.tag_accuracy

    # 1. difficulty match
    diff            = abs(problem.rating - target_rating)
    relevance_score = max(0.0, 1.0 - diff / RATING_SCALE)

    # 2. weak tag boost (IDF-weighted)
    max_idf = max(tag_idf.values()) if tag_idf else 1.0

    weak_score = sum(
        (1 - tag_accuracy.get(tag, 0.0)) *    (tag_idf.get(tag, 1.0) / max_idf)
        for tag in problem.tags
    )
    return relevance_score + weak_score


# -------------------------
# RECOMMENDATION — MAX-HEAP (O(n log k))
# -------------------------

def recommend_problems(
    unsolved: dict[str, Problem],
    profile: UserProfile,
    tag_idf: dict[str, float],
    k: int = 10,
) -> list[ScoredProblem]:
    """
    Score every candidate problem and return the top-k using a min-heap of
    size k. This is O(n log k) rather than O(n log n) — we never sort the
    full list.

    Final score = base_score + 0.2 * recency_score

    recency_score measures how new the problem's contest is, normalized
    against the actual max contest ID in the candidate pool so the score
    is always in (0, 1] regardless of how large contest IDs grow over time.
    This is different from exponential decay (which is user-side) — this is
    purely about how recent the problem itself is in the CF problemset.
    """
    if not unsolved:
        return []
    

    target   = profile.target_rating
    unsolved = {
        pid: prob for pid, prob in unsolved.items()
        if abs(prob.rating - target) <= 400
    }

    if not unsolved:
        return []

    # compute max contest_id once so normalization is always in (0, 1]
    max_contest_id = max(prob.contest_id for prob in unsolved.values())

    # min-heap: (score, pid) — we want the k largest scores
    heap: list[tuple[float, str]] = []

    # first pass — compute all base scores
    scored_raw = {}
    for pid, prob in unsolved.items():
        scored_raw[pid] = score_problem(prob, profile, tag_idf)

    # normalize base scores to 0-1
    max_base = max(scored_raw.values()) if scored_raw else 1.0

    # second pass — add recency and push to heap
    for pid, prob in unsolved.items():
        base_score    = scored_raw[pid] / max_base
        recency_score = prob.contest_id / max_contest_id
        s             = base_score + 0.2 * recency_score

        if len(heap) < k:
            heapq.heappush(heap, (s, pid))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, pid))

    # sort descending so index 0 is the best
    top = sorted(heap, reverse=True)

    return [
        ScoredProblem(pid=pid, problem=unsolved[pid], score=s, mode="exploit")
        for s, pid in top
    ]


# -------------------------
# UCB1 BANDIT
# -------------------------

class UCB1Bandit:
    """
    Upper Confidence Bound 1 bandit for problem recommendation.

    Replaces epsilon-greedy (which was implemented before). Instead of 
    randomly exploring with probability epsilon (stateless), UCB1 tracks
    how many times each problem has been recommended and how well it performed,
    then adds an exploration bonus that shrinks as a problem is recommended more.

    Formula per arm:  UCB = avg_reward + sqrt(2 * ln(t) / n_i)
                      where t = total recommendations, n_i = times arm i was chosen

    Problems never recommended yet are always prioritised (infinite UCB).

    How reward works in this project:
      - When you call select(), it picks the best problem to show.
      - When the user solves a recommended problem, call update(pid, 1.0).
      - When they skip or fail it, call update(pid, 0.0).
      - Over time UCB1 learns which types of problems actually get solved.
    """

    def __init__(self):
        self.counts: dict[str, int]   = {}   # pid -> times recommended
        self.values: dict[str, float] = {}   # pid -> running average reward
        self.t: int = 0                      # total recommendations so far

    def select(self, candidates: list[str]) -> str:
        """Pick the candidate with the highest UCB score."""
        self.t += 1
        best_pid = None
        best_ucb = -1.0

        for pid in candidates:
            n = self.counts.get(pid, 0)

            if n == 0:
                # never recommended — infinite exploration bonus, pick immediately
                return pid

            avg = self.values[pid]
            exploration_bonus = math.sqrt(2 * math.log(self.t) / n)
            ucb = avg + exploration_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_pid = pid

        return best_pid

    def update(self, pid: str, reward: float):
        """
        Update the running average reward for a problem after feedback.
        reward = 1.0 if the user solved it, 0.0 if they skipped/failed.
        """
        n = self.counts.get(pid, 0)
        v = self.values.get(pid, 0.0)
        self.counts[pid] = n + 1
        self.values[pid] = (v * n + reward) / (n + 1)
    
    def recommend(
        self,
        scored: list[ScoredProblem],
        unsolved: dict[str, Problem],
        profile: UserProfile,
        sorted_ratings: list[int],
        rating_to_pids: dict[int, list[str]],
        k: int = 10,
    ) -> list[ScoredProblem]:
        """
        Build a final recommendation list of k problems.

        - Exploit (80%): pull from the scored top-k list using UCB1 selection.
        UCB1 balances between problems with high scores (exploitation) and
        problems that haven't been recommended much yet (exploration bonus).

        - Explore (20%): use the rating index to find problems in the right
        difficulty range that are completely outside the scored list —
        true exploration beyond the user's tag bubble.

        """
        from data_fetcher import get_problems_in_rating_range

        target = profile.target_rating
        lo, hi = int(target), int(target + 200)

        exploit_pids = [sp.pid for sp in scored]
        scored_map   = {sp.pid: sp for sp in scored}

        # explore pool — problems in rating range not already in scored list
        explore_pool = [
            pid for pid in get_problems_in_rating_range(lo, hi, sorted_ratings, rating_to_pids)
            if pid in unsolved and pid not in set(exploit_pids)
        ]
        if not explore_pool:
            explore_pool = list(unsolved.keys())

        final: list[ScoredProblem] = []

        # fixed split — 90% exploit, 10% explore, always guaranteed
        n_exploit = int(k * 0.9)   # 9 out of 10
        n_explore  = k - n_exploit  # 1 out of 10

        remaining_exploit = list(exploit_pids)

        for _ in range(n_exploit):
            if remaining_exploit:
                pid = self.select(remaining_exploit)
                remaining_exploit.remove(pid)
                sp  = scored_map[pid]
                final.append(ScoredProblem(pid=pid, problem=sp.problem, score=sp.score, mode="exploit"))

        for _ in range(n_explore):
            if explore_pool:
                pid = random.choice(explore_pool)
                explore_pool.remove(pid)
                final.append(ScoredProblem(pid=pid, problem=unsolved[pid], score=0.0, mode="explore"))

        return final
