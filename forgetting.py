"""
forgetting.py
─────────────────────────────────────────────────────────────────────
Ebbinghaus Forgetting Curve — Hermann Ebbinghaus, 1885.
Still the most empirically validated model of memory decay in
existence. Used in SuperMemo, Anki, and Duolingo's spaced-repetition.

The formula:
    R(t) = e^(−t / S)

Where:
    R(t) = retrievability (probability of recall) after t days
    t    = days since last review
    S    = stability (how long this memory lasts before decaying to ~37%)

Stability S is NOT fixed. It grows with each successful review:
    S_new = S_old × difficulty_factor × review_quality_factor

This means:
    - A chapter reviewed once 3 days ago: decays quickly
    - A chapter reviewed 5 times over 3 weeks: decays slowly
    - A chapter never reviewed after initial learning: lowest stability

We combine BKT output with forgetting to get:
    P(known_at_exam) = P(known_now) × R(days_remaining)

This is the metric we optimize — not "what's weak now" but
"what will this student NOT know on exam day."
"""

import math
from datetime import datetime, date


# ── Stability base values (days) by subject ────────────────────────
# JEE concepts are harder to retain than general knowledge.
# Physics formulas decay faster than Chemistry reaction rules.
BASE_STABILITY = {
    "Physics":     3.5,   # mechanics, electrostatics — formula-heavy
    "Chemistry":   4.5,   # reactions, bonding — more rule-based, stickier
    "Mathematics": 4.0,   # procedures — need practice to retain
}
DEFAULT_STABILITY = 4.0

# How much stability grows per additional review (multiplicative)
STABILITY_GROWTH_FACTOR = 2.2  # each review roughly doubles retention time

# Exam date — assume JEE Mains April 2026 (real upcoming date)
EXAM_DATE = date(2026, 4, 13)
REFERENCE_DATE = date(2026, 4, 6)   # today


def days_since(date_str: str) -> float:
    """Days elapsed from a date string (YYYY-MM-DD) to today."""
    if not date_str:
        return 60.0  # assume very stale if unknown
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0.0, (REFERENCE_DATE - d).days)
    except ValueError:
        return 60.0


def days_to_exam() -> int:
    """Days remaining until exam from today."""
    return max(1, (EXAM_DATE - REFERENCE_DATE).days)


def compute_stability(subject: str, num_reviews: int,
                      avg_performance: float) -> float:
    """
    Stability S in days.

    Grows with number of reviews and quality of performance.
    High performance → stronger encoding → higher stability.

    Formula: S = S_base × (growth_factor ^ reviews) × performance_multiplier
    Capped at 90 days (semester-level retention).
    """
    S_base = BASE_STABILITY.get(subject, DEFAULT_STABILITY)

    # Each review compounds stability
    review_compound = STABILITY_GROWTH_FACTOR ** max(0, num_reviews - 1)

    # Performance quality: 0.5 to 1.5 multiplier
    # avg_performance is 0–1 (fraction of max score)
    performance_multiplier = 0.5 + avg_performance

    S = S_base * review_compound * performance_multiplier
    return min(S, 90.0)  # cap at 90 days


def retrievability(days_elapsed: float, stability: float) -> float:
    """
    R(t) = e^(−t / S)
    Probability of recalling the material after `days_elapsed` days.
    """
    if stability <= 0:
        return 0.0
    return math.exp(-days_elapsed / stability)


def p_known_at_exam(p_known_now: float,
                    days_elapsed_since_last_review: float,
                    stability: float) -> float:
    """
    Core metric: probability the student retains this chapter on exam day.

    P(known_at_exam) = P(known_now) × R(days_elapsed + days_to_exam)

    We project forward to exam day: total elapsed time includes both
    time already passed since last review AND days until exam.
    """
    total_decay_days = days_elapsed_since_last_review + days_to_exam()
    R = retrievability(total_decay_days, stability)
    return round(p_known_now * R, 4)


def compute_chapter_memory_profile(chapter: str,
                                   subject: str,
                                   p_known_now: float,
                                   attempts: int,
                                   last_date: str,
                                   avg_pct: float) -> dict:
    """
    Full memory profile for one (student, chapter).
    Returns everything needed for the risk scorer.
    """
    elapsed   = days_since(last_date)
    avg_perf  = avg_pct / 100.0
    stability = compute_stability(subject, attempts, avg_perf)
    R_now     = retrievability(elapsed, stability)
    p_exam    = p_known_at_exam(p_known_now, elapsed, stability)
    risk      = round(1.0 - p_exam, 4)

    # Urgency: how fast is this knowledge decaying?
    # dR/dt = -(1/S) * R — normalized to 0-1
    decay_rate = round((1.0 / max(stability, 1)) * R_now, 4)

    return {
        "chapter":             chapter,
        "subject":             subject,
        "p_known_now":         round(p_known_now, 4),
        "stability_days":      round(stability, 1),
        "days_since_review":   round(elapsed, 0),
        "retrievability_now":  round(R_now, 4),
        "p_known_at_exam":     round(p_exam, 4),
        "exam_risk_score":     risk,    # 0=safe, 1=certain to forget
        "decay_rate_per_day":  decay_rate,
        "days_to_exam":        days_to_exam(),
        "urgency": _urgency_label(risk, elapsed, stability),
    }


def _urgency_label(risk: float, elapsed: float, stability: float) -> str:
    """Human-readable urgency classification."""
    if risk > 0.80:
        return "CRITICAL"     # will almost certainly forget by exam
    if risk > 0.60:
        return "HIGH"         # serious risk of forgetting
    if risk > 0.40:
        return "MEDIUM"       # needs review before exam
    if risk > 0.20:
        return "LOW"          # comfortable but don't neglect
    return "SAFE"             # well-consolidated, exam-ready
