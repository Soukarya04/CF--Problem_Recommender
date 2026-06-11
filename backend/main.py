import argparse

from data_fetcher import (
    get_user_submissions,
    get_all_problems,
    get_user_rating,
    extract_solved_problems,
    extract_problem_details,
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


def main():
    # -------------------------
    # ARGS
    # -------------------------
    parser = argparse.ArgumentParser(description="Codeforces problem recommender")
    parser.add_argument("handle",          type=str,            help="Codeforces handle")
    parser.add_argument("--top-k",         type=int, default=10, help="Number of recommendations")
    parser.add_argument("--recent-limit",  type=int, default=50, help="Recent solved problems to use as seed")
    args = parser.parse_args()

    handle = args.handle

    # -------------------------
    # STEP 1: Fetch submissions
    # -------------------------
    print(f"Fetching submissions for '{handle}'...")
    submissions = get_user_submissions(handle)

    if submissions is None:
        print("[ERROR] Could not fetch submissions. Exiting.")
        return

    print(f"  Total submissions fetched: {len(submissions)}")

    # -------------------------
    # STEP 2: Solved set + problem details
    # -------------------------
    solved   = extract_solved_problems(submissions)
    problems = extract_problem_details(submissions)
    print(f"  Unique problems solved: {len(solved)}")

    # -------------------------
    # STEP 3: Fetch full problemset
    # -------------------------
    print("\nFetching full Codeforces problemset...")
    all_problems_raw = get_all_problems()

    if all_problems_raw is None:
        print("[ERROR] Could not fetch problemset. Exiting.")
        return

    problemset = build_problemset(all_problems_raw)
    print(f"  Total rated problems in problemset: {len(problemset)}")

    # -------------------------
    # STEP 4: Build indexes
    # -------------------------
    print("\nBuilding indexes...")
    tag_index                      = build_tag_index(problemset)
    sorted_ratings, rating_to_pids = build_rating_index(problemset)
    tag_idf                        = build_tag_idf(problemset)
    print(f"  Unique tags indexed: {len(tag_index)}")
    print(f"  Unique ratings indexed: {len(sorted_ratings)}")

    # -------------------------
    # STEP 5: Fetch rating history
    # -------------------------
    print(f"\nFetching rating history for '{handle}'...")
    rating_history = get_user_rating(handle)

    if rating_history is None:
        print("[ERROR] Could not fetch rating history. Exiting.")
        return

    current_rating = compute_recent_cf_rating(rating_history)
    print(f"  Current rating (recent avg): {round(current_rating, 1)}")
    print(f"  Target rating for recommendations: {round(current_rating - TARGET_RATING_OFFSET, 1)}")

    # -------------------------
    # STEP 6: Build user profile
    # -------------------------
    print("\nBuilding user profile...")
    profile = build_user_profile(submissions, solved, problemset, current_rating)
    profile.handle = handle

    # show top 5 weakest tags (lowest accuracy)
    sorted_tags = sorted(profile.tag_accuracy.items(), key=lambda x: x[1])
    print(f"\n  Top 5 weakest tags (by accuracy):")
    for tag, acc in sorted_tags[:5]:
        attempts = profile.tag_attempts.get(tag, 0)
        print(f"    {tag:<35} accuracy= {round(acc * 100, 2):<5}%    attempts= {attempts}")

    print(f"\n  Unsolved tag count: {len(profile.unsolved_tags)}")

    print("\nUnsolved Tags:")
    print(", ".join(profile.unsolved_tags))


    # -------------------------
    # STEP 7: Generate candidate pool
    # -------------------------
    print("\nGenerating candidate pool (inverted tag index)...")
    recent_solved = get_recent_solved(submissions, limit=args.recent_limit)
    candidates    = generate_candidates(recent_solved, problemset, tag_index)

    unsolved = {
        pid: problemset[pid]
        for pid in candidates
        if pid not in solved
    }

    print(f"  Recent solved seed size: {len(recent_solved)}")
    print(f"  Candidate pool size:     {len(candidates)}")
    print(f"  Unsolved in pool:        {len(unsolved)}")

    if not unsolved:
        print("[ERROR] No unsolved candidates found. Try increasing --recent-limit.")
        return

    # -------------------------
    # STEP 8: Score + recommend
    # -------------------------
    print(f"\nScoring candidates and selecting top {args.top_k}...")
    scored = recommend_problems(unsolved, profile, tag_idf, k=args.top_k)

    # -------------------------
    # STEP 9: UCB1 final selection
    # -------------------------
    bandit = UCB1Bandit()
    final  = bandit.recommend(
        scored=scored,
        unsolved=unsolved,
        profile=profile,
        sorted_ratings=sorted_ratings,
        rating_to_pids=rating_to_pids,
        k=args.top_k,
    )

    # -------------------------
    # STEP 10: Display results
    # -------------------------
    print(f"\n{'='*65}")
    print(f"  Top {args.top_k} Recommended Problems for {handle}")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'Problem ID':<14} {'Rating':<8} {'Mode':<10} {'Score':<8} Tags")
    print(f"  {'-'*60}")

    for i, sp in enumerate(final, 1):
        score_str = f"{sp.score:.3f}" if sp.mode == "exploit" else "  —"
        tags_str  = ", ".join(sp.problem.tags[:3])
        if len(sp.problem.tags) > 3:
            tags_str += f" (+{len(sp.problem.tags) - 3})"
        print(f"  {i:<4} {sp.pid:<14} {sp.problem.rating:<8} {sp.mode:<10} {score_str:<8} {tags_str}")

    print(f"{'='*65}")
    print("\nDone.")
    print("\nNOTE: Call bandit.update(pid, reward) after each session to improve future recommendations.")
    print("      reward=1.0 if the user solved it, 0.0 if they skipped or failed.")


if __name__ == "__main__":
    main()
