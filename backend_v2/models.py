from dataclasses import dataclass, field


@dataclass
class Problem:
    contest_id: int
    index: str          # "A", "B", "1A", etc.
    rating: int
    tags: list[str]
    name: str = ""

    @property
    def pid(self) -> str:
        return f"{self.contest_id}-{self.index}"


@dataclass
class Submission:
    problem: Problem
    verdict: str        # "OK", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", etc.
    timestamp: int      # Unix epoch seconds


@dataclass
class RatingEvent:
    contest_id: int
    new_rating: int
    timestamp: int      # Unix epoch seconds


@dataclass
class UserProfile:
    handle: str
    current_rating: float
    tag_accuracy: dict[str, float]      # tag -> weighted fraction solved (with decay)
    tag_attempts: dict[str, int]        # tag -> total times attempted
    unsolved_tags: list[str]            # tags never solved at all

    @property
    def target_rating(self) -> float:
        """Rating to target for recommendations — slightly below current."""
        return self.current_rating - TARGET_RATING_OFFSET


@dataclass
class ScoredProblem:
    pid: str
    problem: Problem
    score: float
    mode: str = "exploit"   # "exploit" or "explore"


# -------------------------
# CONSTANTS
# -------------------------

TARGET_RATING_OFFSET = 100      # recommend problems slightly below user rating
RATING_SCALE          = 400     # normalization window for rating distance
RECENCY_HALF_LIFE_DAYS = 90     # solved problems lose half their weight every 90 days
