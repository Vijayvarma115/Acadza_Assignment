"""
bkt_engine.py
─────────────────────────────────────────────────────────────────────
Bayesian Knowledge Tracing (BKT) — Corbett & Anderson, 1994.
Used in production by Carnegie Learning, ASSISTments, and dozens of
serious adaptive learning platforms.

The model treats knowledge as a HIDDEN binary state:
  KNOWN  (student has mastered this chapter)
  UNKNOWN (student has not yet mastered it)

We never observe the state directly. We observe PERFORMANCE — and
performance is noisy because:
  - A student who DOESN'T know can still get lucky (guess)
  - A student who DOES know can still make a mistake (slip)

Four parameters per chapter (fitted from EdTech literature defaults,
tunable per subject):
  P_L0  — prior probability of knowing before any attempt
  P_T   — probability of learning (UNKNOWN → KNOWN) per attempt
  P_G   — guess probability  (P(correct | UNKNOWN))
  P_S   — slip probability   (P(wrong   | KNOWN))

After each attempt we update P(known) using Bayes rule.
The result is a credible belief about the student's current knowledge
state — not an average, not a percentage, but a probability.

Why this beats averaging:
  avg_pct of [90, 10, 90] = 63.3%  → "medium, assign medium practice"
  BKT of [90, 10, 90]             → P(known)=0.71, trend recovered
  avg_pct of [10, 10, 15]         → "weak"
  BKT of [10, 10, 15]            → P(known)=0.08, deeply unknown,
                                     needs concept first
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ── BKT defaults tuned for JEE (high-stakes, conceptual domain) ────
# Lower P_T than MOOCs — JEE concepts don't click in one session.
# Higher P_S than standard — students misread questions under time.
DEFAULT_PARAMS = {
    "Physics":     {"P_L0": 0.30, "P_T": 0.08, "P_G": 0.20, "P_S": 0.15},
    "Chemistry":   {"P_L0": 0.30, "P_T": 0.09, "P_G": 0.22, "P_S": 0.12},
    "Mathematics": {"P_L0": 0.35, "P_T": 0.10, "P_G": 0.18, "P_S": 0.10},
}
FALLBACK_PARAMS = {"P_L0": 0.30, "P_T": 0.09, "P_G": 0.20, "P_S": 0.12}


@dataclass
class ChapterKnowledgeState:
    """
    Tracks the evolving knowledge belief for one (student, chapter) pair.
    """
    chapter:    str
    subject:    str
    p_known:    float = 0.30          # current P(known) — starts at prior
    attempts:   int   = 0
    history:    list  = field(default_factory=list)  # [(date, p_correct, p_known_after)]
    last_date:  Optional[str] = None


def _performance_to_p_correct(normalized_pct: float,
                                attempt_rate: float,
                                completed: bool) -> float:
    """
    Convert a raw attempt outcome into P(correct) for BKT update.

    We can't observe per-question correctness — only session-level
    percentage. We model P(correct) as:
      - base: normalized_pct / 100
      - penalized if student didn't complete (left early = unknown territory)
      - penalized by skip rate (skipped ≈ gave up)

    Range: [0, 1]
    """
    base = normalized_pct / 100.0
    completion_penalty = 1.0 if completed else 0.75
    engagement_factor  = max(0.5, attempt_rate)
    return min(1.0, base * completion_penalty * engagement_factor)


def update_bkt(state: ChapterKnowledgeState,
               p_correct_observed: float,
               subject: str) -> ChapterKnowledgeState:
    """
    Perform one BKT update step.

    Given current P(known) and observed performance:
    1. Compute P(known | evidence) via Bayes rule
    2. Apply learning transition: student may have learned during attempt
    """
    params = DEFAULT_PARAMS.get(subject, FALLBACK_PARAMS)
    P_L = state.p_known
    P_T = params["P_T"]
    P_G = params["P_G"]
    P_S = params["P_S"]

    # ── Step 1: Bayes update ───────────────────────────────────────
    # P(correct) = P(correct|known)*(1-P_S) + P(correct|unknown)*P_G
    # We use p_correct_observed as a soft evidence signal.

    # Posterior: P(known | observed_performance)
    if p_correct_observed >= 0.5:
        # Evidence of correctness
        p_correct_given_known   = 1.0 - P_S
        p_correct_given_unknown = P_G
        evidence = p_correct_observed  # weight of "got it right"
    else:
        # Evidence of incorrectness
        p_correct_given_known   = P_S
        p_correct_given_unknown = 1.0 - P_G
        evidence = 1.0 - p_correct_observed

    numerator   = p_correct_given_known   * P_L * evidence
    denominator = (p_correct_given_known  * P_L * evidence +
                   p_correct_given_unknown * (1 - P_L) * evidence)

    if denominator < 1e-10:
        p_known_given_evidence = P_L
    else:
        p_known_given_evidence = numerator / denominator

    # ── Step 2: Learning transition ───────────────────────────────
    # Even if student didn't know at start, they may have learned
    p_known_after = p_known_given_evidence + (1 - p_known_given_evidence) * P_T

    # Clamp to valid probability range
    p_known_after = max(0.01, min(0.99, p_known_after))

    state.p_known  = p_known_after
    state.attempts += 1
    state.history.append({
        "p_correct_observed": round(p_correct_observed, 3),
        "p_known_after":      round(p_known_after, 3),
    })

    return state


def run_bkt_for_student(student: dict) -> dict[str, ChapterKnowledgeState]:
    """
    Process all attempts chronologically for a student.
    Returns {chapter_name: ChapterKnowledgeState} with final P(known).
    """
    # Sort attempts by date to respect temporal order
    attempts = sorted(student["attempts"], key=lambda a: a.get("date", ""))

    states: dict[str, ChapterKnowledgeState] = {}

    for attempt in attempts:
        subject   = attempt.get("subject", "Physics")
        chapters  = attempt.get("chapters", [])
        pct       = attempt.get("normalized_pct", 0.0)
        completed = attempt.get("completed", True)
        total_q   = attempt.get("total_questions", 1)
        attempted = attempt.get("attempted", 0)
        attempt_rate = attempted / max(total_q, 1)
        date      = attempt.get("date", "")

        p_correct = _performance_to_p_correct(pct, attempt_rate, completed)

        # Each chapter in this attempt gets the same evidence signal
        # (limitation: we don't have per-chapter scores within a session)
        for chapter in chapters:
            if chapter not in states:
                params = DEFAULT_PARAMS.get(subject, FALLBACK_PARAMS)
                states[chapter] = ChapterKnowledgeState(
                    chapter  = chapter,
                    subject  = subject,
                    p_known  = params["P_L0"],
                )

            states[chapter].last_date = date
            states[chapter] = update_bkt(states[chapter], p_correct, subject)

    return states
