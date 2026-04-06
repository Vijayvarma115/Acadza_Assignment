"""
risk_scorer.py
─────────────────────────────────────────────────────────────────────
Combines BKT knowledge state + Ebbinghaus forgetting into a single
exam-day risk score per chapter.

This is the brain of the system. The output of this module drives
the entire DOST recommendation order — not averages, not percentages,
not rules — but a probabilistic forecast of what the student will
and won't know on exam day.

Risk Score = 1 − P(student retains this chapter on exam day)

A high risk score means: "We predict with confidence that if exam
were today + N days from now, the student would fail this chapter."
"""

from bkt_engine import run_bkt_for_student
from forgetting import compute_chapter_memory_profile
from data_loader import chapter_avg_pct, chapter_to_topic


def score_student_risks(student: dict) -> list[dict]:
    """
    Returns a list of chapter risk profiles, sorted by exam_risk_score
    descending (most dangerous chapter first).

    Each entry:
    {
        chapter, subject, topic_key,
        p_known_now,         ← BKT output
        stability_days,      ← Ebbinghaus stability
        retrievability_now,  ← R(days_since_review)
        p_known_at_exam,     ← the critical metric
        exam_risk_score,     ← 1 - p_known_at_exam
        urgency,             ← CRITICAL/HIGH/MEDIUM/LOW/SAFE
        days_since_review,
        bkt_history,         ← how knowledge evolved
        avg_pct,             ← raw performance for context
        attempts,
    }
    """
    # Step 1: Run BKT across all attempts
    bkt_states = run_bkt_for_student(student)

    # Step 2: Get chapter-level performance stats
    chapter_stats = chapter_avg_pct(student)

    # Step 3: Compute memory profile for each chapter
    profiles = []
    for chapter, state in bkt_states.items():
        stats = chapter_stats.get(chapter, {})
        avg_pct   = stats.get("avg_pct", 0.0)
        attempts  = stats.get("attempts", 1)
        subject   = stats.get("subject", state.subject)
        last_date = stats.get("last_date", state.last_date or "")

        profile = compute_chapter_memory_profile(
            chapter     = chapter,
            subject     = subject,
            p_known_now = state.p_known,
            attempts    = attempts,
            last_date   = last_date,
            avg_pct     = avg_pct,
        )

        profile["avg_pct"]     = avg_pct
        profile["attempts"]    = attempts
        profile["topic_key"]   = chapter_to_topic(chapter)
        profile["bkt_history"] = state.history

        profiles.append(profile)

    # Sort by exam risk (highest risk first)
    profiles.sort(key=lambda x: x["exam_risk_score"], reverse=True)
    return profiles


def get_student_summary(student: dict, risk_profiles: list[dict]) -> dict:
    """
    High-level student summary combining all signals.
    """
    attempts = student["attempts"]
    pcts = [a["normalized_pct"] for a in attempts]

    completed  = sum(1 for a in attempts if a.get("completed", False))
    abort_rate = round(1 - completed / len(attempts), 2)
    avg_pct    = round(sum(pcts) / len(pcts), 1)

    # Split into risk tiers
    critical = [r for r in risk_profiles if r["urgency"] == "CRITICAL"]
    high     = [r for r in risk_profiles if r["urgency"] == "HIGH"]
    medium   = [r for r in risk_profiles if r["urgency"] == "MEDIUM"]
    low_safe = [r for r in risk_profiles if r["urgency"] in ("LOW", "SAFE")]

    # Trend: last 3 vs first 3
    early = sum(pcts[:3]) / 3 if len(pcts) >= 3 else pcts[0]
    late  = sum(pcts[-3:]) / 3 if len(pcts) >= 3 else pcts[-1]
    delta = late - early
    trend = "improving" if delta > 8 else ("declining" if delta < -8 else "stable")

    # Behavioral signals
    skip_rates = [
        a.get("skipped", 0) / max(a.get("total_questions", 1), 1)
        for a in attempts
    ]
    avg_skip = round(sum(skip_rates) / len(skip_rates), 2)

    # Detect test-anxiety: assignment mode scores >> test mode scores
    test_pcts   = [a["normalized_pct"] for a in attempts if a.get("mode") == "test"]
    assign_pcts = [a["normalized_pct"] for a in attempts if a.get("mode") == "assignment"]
    test_anxiety = (
        len(assign_pcts) >= 2 and len(test_pcts) >= 2 and
        (sum(assign_pcts)/len(assign_pcts)) - (sum(test_pcts)/len(test_pcts)) > 20
    )

    # Subject-wise average
    subj_buckets = {}
    for a in attempts:
        s = a["subject"]
        subj_buckets.setdefault(s, []).append(a["normalized_pct"])
    subject_avg = {
        s: round(sum(v)/len(v), 1) for s, v in subj_buckets.items()
    }

    return {
        "student_id":         student["student_id"],
        "name":               student["name"],
        "stream":             student.get("stream", "JEE"),
        "class":              student.get("class"),
        "total_attempts":     len(attempts),
        "overall_avg_pct":    avg_pct,
        "performance_trend":  trend,
        "trend_delta":        round(delta, 1),
        "completion_rate":    round(completed / len(attempts), 2),
        "abort_rate":         abort_rate,
        "avg_skip_rate":      avg_skip,
        "test_anxiety":       test_anxiety,
        "subject_avg":        subject_avg,
        "risk_profiles":      risk_profiles,
        "critical_chapters":  [r["chapter"] for r in critical],
        "high_risk_chapters": [r["chapter"] for r in high],
        "medium_chapters":    [r["chapter"] for r in medium],
        "safe_chapters":      [r["chapter"] for r in low_safe],
        "days_to_exam":       risk_profiles[0]["days_to_exam"] if risk_profiles else 0,
        "exam_readiness_pct": round(
            (1 - sum(r["exam_risk_score"] for r in risk_profiles) / max(len(risk_profiles), 1)) * 100, 1
        ),
    }
