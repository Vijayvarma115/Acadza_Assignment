"""
adaptive_planner.py
─────────────────────────────────────────────────────────────────────
Builds the step-by-step DOST plan from risk profiles.

Key difference from naive approach:
─────────────────────────────────────────────────────────────────────
Naive:  sort by avg_pct → assign concept/formula/assignment per band
Here:   sort by exam_risk_score → assign DOST based on WHAT the
        risk is driven by (low P_known? low stability? time decay?)

The DOST type isn't just determined by score band — it's determined
by WHY the chapter is risky:

  Low P(known) AND low stability  → concept + formula + assignment
  Low P(known) BUT good stability → student knew it, but performance
                                    dropped — likely needs formula
                                    refresh + medium assignment
  OK P(known) BUT low stability   → knowledge exists but unstable —
                                    revision + spaced practice
  OK P(known) AND good stability  → safe, use for speedRace or
                                    exam simulation only

This is called "differential diagnosis" — same symptom (fail) can
have different causes, and the treatment should match the cause.
"""

from data_loader import chapter_to_topic
from forgetting import days_to_exam


DIFF_NUM_TO_LABEL = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}


def _pick_questions(questions: list[dict], topic_key: str,
                    difficulty: str, n: int = 8) -> list[str]:
    """Pick up to n questions for a topic at given difficulty."""
    diff_map = {"easy": (1, 2), "medium": (3,), "hard": (4, 5)}
    pool = [q for q in questions if q.get("topic") == topic_key]
    preferred = [q for q in pool if q.get("difficulty") in diff_map.get(difficulty, (3,))]
    result = (preferred + [q for q in pool if q not in preferred])[:n]
    return [q["qid"] for q in result if q.get("qid")]


def _diagnose(risk_profile: dict) -> str:
    """
    Diagnose WHY a chapter is risky.
    Returns one of: NEVER_LEARNED, LEARNED_UNSTABLE, STABLE_DECAYING, EXAM_PRESSURE
    """
    p = risk_profile["p_known_now"]
    s = risk_profile["stability_days"]
    r = risk_profile["retrievability_now"]
    attempts = risk_profile["attempts"]

    if p < 0.25 and attempts <= 1:
        return "NEVER_LEARNED"          # barely encountered
    if p < 0.35:
        return "DEEPLY_UNKNOWN"         # tried multiple times, still not mastered
    if p < 0.55 and s < 5:
        return "LEARNED_UNSTABLE"       # partial knowledge, weak encoding
    if p >= 0.55 and r < 0.50:
        return "STABLE_DECAYING"        # knew it well, but time has eroded it
    if p >= 0.55 and r >= 0.50:
        return "EXAM_PRESSURE"          # ready but needs exam conditioning
    return "LEARNED_UNSTABLE"


def _build_step(step_num: int, dost_type: str, chapter: str,
                topic_key: str, questions: list[dict],
                difficulty: str, params: dict,
                reasoning: str, message: str) -> dict:
    qids = []
    if dost_type in ("practiceAssignment", "clickingPower", "pickingPower", "practiceTest"):
        qids = _pick_questions(questions, topic_key, difficulty)

    return {
        "step":           step_num,
        "dost_type":      dost_type,
        "target_chapter": chapter,
        "parameters":     params,
        "question_ids":   qids,
        "diagnosis":      "",   # filled by caller
        "reasoning":      reasoning,
        "student_message": message,
    }


def _steps_for_never_learned(chapter: str, topic: str,
                              questions: list[dict], n: int) -> list[dict]:
    steps = []
    steps.append(_build_step(n, "concept", chapter, topic, questions, "easy", {},
        f"P(known)≈0 with very few attempts — student has barely encountered {chapter}. "
        "No point practicing without conceptual foundation.",
        f"Let's start fresh on {chapter}. Read through the concept module fully "
        "— don't skip anything."))
    steps.append(_build_step(n+1, "formula", chapter, topic, questions, "easy", {},
        f"Formulas for {chapter} need to be locked in before any problem solving.",
        f"Memorize the key formulas for {chapter}. Write them out. "
        "You'll thank yourself in the exam."))
    steps.append(_build_step(n+2, "practiceAssignment", chapter, topic, questions, "easy",
        {"difficulty": "easy", "type_split": {"scq": 8, "integer": 2}},
        f"First contact with {chapter} problems at easy level — "
        "build pattern recognition before difficulty increases.",
        f"Your first real practice on {chapter}. No pressure — "
        "focus on getting the approach right, not the speed."))
    steps.append(_build_step(n+3, "clickingPower", chapter, topic, questions, "easy",
        {"total_questions": 10},
        f"Reinforce {chapter} recall speed after initial concept+practice cycle.",
        f"Speed round! 10 quick-fire questions on {chapter}. "
        "Build the reflex."))
    return steps


def _steps_for_deeply_unknown(chapter: str, topic: str,
                               questions: list[dict], n: int,
                               skip_rate: float) -> list[dict]:
    steps = []
    steps.append(_build_step(n, "formula", chapter, topic, questions, "easy", {},
        f"{chapter} has been attempted multiple times with P(known) still below 0.35. "
        "Core equations are likely not internalized — formula revision before more practice.",
        f"You've tried {chapter} before but it's not clicking. "
        "Let's go back to the formulas — that's usually where the gap is."))
    if skip_rate > 0.25:
        steps.append(_build_step(n+1, "pickingPower", chapter, topic, questions, "medium", {},
            f"High skip rate ({skip_rate:.0%}) detected. "
            "MCQ elimination practice builds confidence to attempt instead of skip.",
            f"Stop skipping {chapter} questions! Learn to eliminate wrong options "
            "— even a 50/50 guess beats a skip."))
    steps.append(_build_step(n+2, "practiceAssignment", chapter, topic, questions, "easy",
        {"difficulty": "easy", "type_split": {"scq": 7, "integer": 3}},
        f"Untimed practice on {chapter} after formula revision. "
        "Accuracy before speed for deeply unknown chapters.",
        f"Work through these {chapter} problems carefully. No timer. "
        "Every correct step matters."))
    steps.append(_build_step(n+3, "clickingPower", chapter, topic, questions, "easy",
        {"total_questions": 10},
        f"Speed reinforcement after accuracy is built for {chapter}.",
        f"Good work! Now make it fast — 10 questions, {chapter}, go!"))
    return steps


def _steps_for_learned_unstable(chapter: str, topic: str,
                                  questions: list[dict], n: int) -> list[dict]:
    steps = []
    steps.append(_build_step(n, "practiceAssignment", chapter, topic, questions, "medium",
        {"difficulty": "medium", "type_split": {"scq": 6, "mcq": 2, "integer": 2}},
        f"{chapter}: partial knowledge exists (P(known)~0.35–0.55) but encoding is weak "
        "(stability < 5 days). Medium-difficulty practice consolidates the encoding.",
        f"You know some of {chapter} — let's solidify it. "
        "These medium-level problems will cement your understanding."))
    steps.append(_build_step(n+1, "clickingPower", chapter, topic, questions, "medium",
        {"total_questions": 10},
        f"Speed drill after consolidation practice for {chapter}. "
        "Improves retrieval speed which directly increases stability.",
        f"Sharpen your {chapter} reflexes — speed and recall go together."))
    return steps


def _steps_for_stable_decaying(chapter: str, topic: str,
                                 questions: list[dict], n: int,
                                 days_since: float) -> list[dict]:
    steps = []
    # How many review days to allocate based on decay
    review_days = 2 if days_since < 20 else 3
    steps.append(_build_step(n, "revision", chapter, topic, questions, "medium",
        {"alloted_days": review_days, "strategy": 2, "daily_time_minutes": 60},
        f"{chapter}: P(known) is good (≥0.55) but {days_since:.0f} days have passed "
        f"since last review and retrievability has dropped below 50%. "
        "Spaced revision restores stability without re-learning from scratch.",
        f"It's been a while since you last touched {chapter}. "
        f"A {review_days}-day revision plan will bring it back — "
        "you haven't lost it, just need a refresh."))
    steps.append(_build_step(n+1, "practiceAssignment", chapter, topic, questions, "medium",
        {"difficulty": "medium", "type_split": {"scq": 5, "mcq": 3, "integer": 2}},
        f"Practice after revision for {chapter} to re-anchor the recovered knowledge.",
        f"Follow-up practice after your {chapter} revision. "
        "This locks the refreshed knowledge in."))
    return steps


def _steps_for_exam_pressure(chapter: str, topic: str,
                              questions: list[dict], n: int,
                              exam_pattern: str) -> list[dict]:
    steps = []
    steps.append(_build_step(n, "speedRace", chapter, topic, questions, "hard",
        {"rank": 50, "opponent_type": "bot"},
        f"{chapter} is well-consolidated. Exam-condition simulation "
        "pushes the student to perform under competition pressure.",
        f"You're solid on {chapter} — now prove it in a race. "
        "Beat the bot!"))
    return steps


def build_dost_plan(student: dict, risk_profiles: list[dict],
                    questions: list[dict],
                    abort_rate: float, skip_rate: float) -> dict:
    """
    Build the full step-by-step DOST plan.
    Risk profiles are already sorted by exam_risk_score desc.
    """
    steps = []
    n = 1

    # Infer exam pattern
    exam_pattern = "Mains"
    for att in reversed(student["attempts"]):
        ep = att.get("exam_pattern", "")
        if "advanced" in ep.lower():
            exam_pattern = "Advanced"
            break

    # ── If high abort rate, lead with revision on the riskiest chapters ──
    if abort_rate >= 0.33:
        top_risky = [r["chapter"] for r in risk_profiles[:3]
                     if r["urgency"] in ("CRITICAL", "HIGH")]
        if top_risky:
            steps.append({
                "step": n,
                "dost_type": "revision",
                "target_chapter": ", ".join(top_risky),
                "parameters": {
                    "alloted_days": 3,
                    "strategy": 1,
                    "daily_time_minutes": 90,
                },
                "question_ids": [],
                "diagnosis": "HIGH_ABORT_RATE",
                "reasoning": (
                    f"Abort rate {abort_rate:.0%} detected. Student is quitting tests before "
                    "completion — a structured revision plan rebuilds confidence before "
                    "more test exposure."
                ),
                "student_message": (
                    "You've been stopping tests halfway. Before we add more practice, "
                    f"let's get comfortable with {', '.join(top_risky[:2])} through "
                    "a structured revision plan. Finish every session."
                ),
            })
            n += 1

    # ── Main loop: process each chapter by risk order ──────────────────
    test_chapters = []  # collect for final practice test

    for profile in risk_profiles:
        chapter  = profile["chapter"]
        topic    = profile["topic_key"]
        urgency  = profile["urgency"]
        diag     = _diagnose(profile)
        days_ago = profile["days_since_review"]

        if urgency == "SAFE":
            # Exam-pressure conditioning for safe chapters
            new_steps = _steps_for_exam_pressure(chapter, topic, questions, n, exam_pattern)
        elif diag == "NEVER_LEARNED":
            new_steps = _steps_for_never_learned(chapter, topic, questions, n)
            test_chapters.append(chapter)
        elif diag == "DEEPLY_UNKNOWN":
            new_steps = _steps_for_deeply_unknown(chapter, topic, questions, n, skip_rate)
            test_chapters.append(chapter)
        elif diag == "LEARNED_UNSTABLE":
            new_steps = _steps_for_learned_unstable(chapter, topic, questions, n)
            test_chapters.append(chapter)
        elif diag == "STABLE_DECAYING":
            new_steps = _steps_for_stable_decaying(chapter, topic, questions, n, days_ago)
            test_chapters.append(chapter)
        else:
            new_steps = _steps_for_exam_pressure(chapter, topic, questions, n, exam_pattern)

        for idx, s in enumerate(new_steps):
            s["diagnosis"] = diag
            s["step"] = n + idx
        steps.extend(new_steps)
        n += len(new_steps)

    # ── Final exam simulation: combine the riskiest chapters ───────────
    if test_chapters:
        test_topics = [chapter_to_topic(ch) for ch in test_chapters[:3]]
        test_qids   = []
        for tk in test_topics:
            test_qids += _pick_questions(questions, tk, "medium", n=4)
        test_qids = test_qids[:15]

        steps.append({
            "step":           n,
            "dost_type":      "practiceTest",
            "target_chapter": ", ".join(test_chapters[:3]),
            "parameters": {
                "difficulty":       "medium",
                "duration_minutes": 60,
                "paperPattern":     exam_pattern,
            },
            "question_ids":  test_qids,
            "diagnosis":     "EXAM_SIMULATION",
            "reasoning": (
                "Final exam simulation across the highest-risk chapters to "
                "validate all previous DOST work under timed conditions."
            ),
            "student_message": (
                f"Final exam simulation. Chapters: {', '.join(test_chapters[:3])}. "
                "60 minutes. Treat it like the real thing — no breaks, no notes."
            ),
        })

    return {
        "student_id":    student["student_id"],
        "name":          student["name"],
        "exam_pattern":  exam_pattern,
        "days_to_exam":  days_to_exam(),
        "total_steps":   len(steps),
        "priority_chapters": [r["chapter"] for r in risk_profiles[:3]
                               if r["urgency"] in ("CRITICAL", "HIGH", "MEDIUM")],
        "plan":          steps,
    }
