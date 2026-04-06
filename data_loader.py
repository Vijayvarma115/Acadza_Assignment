"""
data_loader.py — v2
Loads and normalizes all data. Same marks normalization as v1.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"

CHAPTER_TO_TOPIC = {
    "thermodynamics":       "thermodynamics",
    "electrostatics":       "electrostatics",
    "kinematics":           "kinematics",
    "optics":               "optics",
    "heat transfer":        "heat_transfer",
    "rotational mechanics": "rotational_mechanics",
    "laws of motion":       "laws_of_motion",
    "organic chemistry":    "organic_chemistry",
    "chemical bonding":     "chemical_bonding",
    "physical chemistry":   "physical_chemistry",
    "coordinate geometry":  "coordinate_geometry",
    "algebra":              "algebra",
    "calculus":             "calculus",
    "probability":          "probability",
    "modern physics":       "modern_physics",
}

def chapter_to_topic(ch: str) -> str:
    return CHAPTER_TO_TOPIC.get(ch.lower().strip(), ch.lower().replace(" ", "_"))


def normalize_marks(raw, total_questions: int) -> float:
    max_marks = total_questions * 4
    if raw is None:
        return 0.0
    s = str(raw).strip()
    if re.match(r'^\+\d+\s*-\d+$', s):
        pos = int(re.search(r'\+(\d+)', s).group(1))
        neg = int(re.search(r'-(\d+)', s).group(1))
        return round(max((pos - neg) / max_marks * 100, 0), 2)
    pct = re.search(r'\((\d+\.?\d*)%\)', s)
    if pct:
        return round(float(pct.group(1)), 2)
    frac = re.match(r'^(\d+\.?\d*)/(\d+\.?\d*)$', s)
    if frac:
        n, d = float(frac.group(1)), float(frac.group(2))
        return round(n / d * 100, 2) if d else 0.0
    try:
        return round(float(s) / max_marks * 100, 2)
    except ValueError:
        return 0.0


def load_students() -> list[dict]:
    with open(DATA_DIR / "student_performance.json", encoding="utf-8-sig") as f:
        students = json.load(f)
    for student in students:
        for att in student["attempts"]:
            tq = att.get("total_questions", 25)
            att["normalized_pct"] = normalize_marks(att.get("marks"), tq)
            # Precompute attempt_rate for convenience
            att["attempt_rate"] = att.get("attempted", 0) / max(tq, 1)
    return students


def load_question_bank() -> tuple[list[dict], dict[str, dict]]:
    with open(DATA_DIR / "question_bank.json", encoding="utf-8-sig") as f:
        raw = json.load(f)
    seen, clean, qid_index = set(), [], {}
    for q in raw:
        raw_id = q.get("_id", "")
        oid = raw_id.get("$oid", "") if isinstance(raw_id, dict) else str(raw_id)
        q["_id"] = oid
        if oid in seen:
            continue
        seen.add(oid)
        qt = q.get("questionType", "scq")
        content = q.get(qt) or q.get("scq") or q.get("mcq") or q.get("integerQuestion") or {}
        if not content.get("answer") or q.get("difficulty") is None:
            continue
        clean.append(q)
        if q.get("qid"):
            qid_index[q["qid"]] = q
    return clean, qid_index


def load_dost_config() -> dict:
    with open(DATA_DIR / "dost_config.json", encoding="utf-8-sig") as f:
        return json.load(f)


def get_student(students: list[dict], student_id: str) -> dict | None:
    return next((s for s in students if s["student_id"] == student_id), None)


def chapter_avg_pct(student: dict) -> dict[str, dict]:
    """
    Returns {chapter: {avg_pct, attempts, subject, last_date}} 
    needed by the forgetting model.
    """
    buckets = defaultdict(list)
    for att in sorted(student["attempts"], key=lambda a: a.get("date", "")):
        subj = att["subject"]
        date = att.get("date", "")
        pct  = att["normalized_pct"]
        for ch in att.get("chapters", []):
            buckets[ch].append({"pct": pct, "subject": subj, "date": date})

    result = {}
    for ch, recs in buckets.items():
        result[ch] = {
            "avg_pct":  round(sum(r["pct"] for r in recs) / len(recs), 2),
            "attempts": len(recs),
            "subject":  recs[-1]["subject"],
            "last_date": recs[-1]["date"],
        }
    return result
