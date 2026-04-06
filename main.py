"""
main.py — v2
FastAPI application with BKT + Ebbinghaus engine.
Same 4 endpoints as spec, completely different intelligence underneath.
"""

import re
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_students, load_question_bank, load_dost_config, get_student
from risk_scorer import score_student_risks, get_student_summary
from adaptive_planner import build_dost_plan

app = FastAPI(
    title="Acadza Adaptive Recommender — BKT + Ebbinghaus Engine",
    description=(
        "Bayesian Knowledge Tracing + Ebbinghaus Forgetting Curve. "
        "Recommends based on predicted exam-day knowledge state, "
        "not just current performance averages."
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def _data():
    students  = load_students()
    questions, qid_index = load_question_bank()
    dost_cfg  = load_dost_config()
    return students, questions, qid_index, dost_cfg

def _students():  return _data()[0]
def _questions(): return _data()[1]
def _qid_index(): return _data()[2]


def _require_student(student_id: str) -> dict:
    s = get_student(_students(), student_id)
    if not s:
        raise HTTPException(404, f"Student '{student_id}' not found. Valid: STU_001–STU_010")
    return s


def _html_plain(html: str) -> str:
    if not html: return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()[:300]


# ── POST /analyze/{student_id} ─────────────────────────────────────

@app.post("/analyze/{student_id}")
def analyze(student_id: str):
    """
    Full analysis using BKT + forgetting curve.
    Shows P(known), stability, retrievability, and exam-day risk per chapter.
    """
    student       = _require_student(student_id)
    risk_profiles = score_student_risks(student)
    summary       = get_student_summary(student, risk_profiles)
    return summary


# ── POST /recommend/{student_id} ───────────────────────────────────

@app.post("/recommend/{student_id}")
def recommend(student_id: str):
    """
    Step-by-step DOST plan. Each step includes the diagnosis
    (WHY this DOST was chosen) + BKT-driven reasoning.
    """
    student       = _require_student(student_id)
    risk_profiles = score_student_risks(student)
    summary       = get_student_summary(student, risk_profiles)

    plan = build_dost_plan(
        student       = student,
        risk_profiles = risk_profiles,
        questions     = _questions(),
        abort_rate    = summary["abort_rate"],
        skip_rate     = summary["avg_skip_rate"],
    )
    return plan


# ── GET /question/{question_id} ────────────────────────────────────

@app.get("/question/{question_id}")
def get_question(question_id: str):
    qid_index = _qid_index()
    q = qid_index.get(question_id) or next(
        (x for x in _questions() if x.get("_id") == question_id), None
    )
    if not q:
        raise HTTPException(404, f"Question '{question_id}' not found.")

    qt = q.get("questionType", "scq")
    content = q.get(qt) or q.get("scq") or q.get("mcq") or q.get("integerQuestion") or {}

    return {
        "qid":              q.get("qid"),
        "_id":              q.get("_id"),
        "questionType":     qt,
        "subject":          q.get("subject"),
        "topic":            q.get("topic"),
        "subtopic":         q.get("subtopic"),
        "difficulty":       q.get("difficulty"),
        "question_preview": _html_plain(content.get("question", "")),
        "answer":           content.get("answer"),
        "solution_preview": _html_plain(content.get("solution", "")),
    }


# ── GET /leaderboard ───────────────────────────────────────────────

@app.get("/leaderboard")
def leaderboard():
    """
    Ranks students by exam_readiness_pct — a forward-looking score
    based on predicted knowledge retention on exam day.
    Not just current performance — who is PREPARED for the exam.
    """
    entries = []
    for student in _students():
        risk_profiles = score_student_risks(student)
        summary       = get_student_summary(student, risk_profiles)

        # Composite: 50% exam readiness + 30% current performance + 20% completion
        composite = round(
            summary["exam_readiness_pct"]  * 0.50 +
            summary["overall_avg_pct"]     * 0.30 +
            summary["completion_rate"]     * 20,
            2,
        )

        entries.append({
            "student_id":         student["student_id"],
            "name":               student["name"],
            "composite_score":    composite,
            "exam_readiness_pct": summary["exam_readiness_pct"],
            "overall_avg_pct":    summary["overall_avg_pct"],
            "completion_rate":    summary["completion_rate"],
            "trend":              summary["performance_trend"],
            "critical_chapters":  summary["critical_chapters"],
            "safe_chapters":      summary["safe_chapters"],
            "top_risk_chapter":   risk_profiles[0]["chapter"] if risk_profiles else "N/A",
        })

    entries.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return {"leaderboard": entries}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
