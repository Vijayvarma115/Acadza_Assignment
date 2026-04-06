"""
recommender_fixed.py
────────────────────────────────────────────────────────────────────
THE BUG — Location: recommend(), the student_profile normalization block.

ORIGINAL (BUGGY):
    cohort_baseline = student_matrix.mean(axis=0)
    student_profile = student_matrix[student_idx] - cohort_baseline   # ← correct gap

    profile_norm    = np.linalg.norm(cohort_baseline)                  # ← BUG: wrong vector
    student_profile = cohort_baseline / (profile_norm + 1e-10)         # ← BUG: discards the gap

WHAT GOES WRONG:
    Line 1 correctly computes the student's personalized gap vector.
    Line 2 then computes the norm of cohort_baseline (wrong vector).
    Line 3 overwrites student_profile with the NORMALIZED COHORT AVERAGE —
    making every student's query vector identical. The gap is thrown away.

    Result: 10/10 overlap for students with completely different weakness profiles.
    No crash, no error, plausible-looking scores — the silent bug.

ROOT CAUSE IN ONE SENTENCE:
    The variable being normalized (cohort_baseline) is not the variable
    that should be normalized (student_profile).

FIX — two variable name corrections:
    profile_norm    = np.linalg.norm(student_profile)          # was cohort_baseline
    student_profile = student_profile / (profile_norm + 1e-10) # was cohort_baseline

WHY AI MISSES IT:
    LLMs pattern-match "normalize the profile" and accept any 3-line
    normalization block as correct. The key test is semantic — run the
    code on students with opposite profiles and check for overlap.
    Buggy: 10/10. Fixed: 0/10.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

TOPICS = [
    "mechanics", "thermodynamics", "electrostatics", "optics",
    "modern_physics", "organic_chemistry", "inorganic_chemistry",
    "physical_chemistry", "algebra", "calculus", "coordinate_geometry",
    "trigonometry"
]
TOPIC_TO_IDX      = {t: i for i, t in enumerate(TOPICS)}
DIFFICULTY_WEIGHT = {"easy": 0.5, "medium": 1.0, "hard": 1.5}


def build_feature_matrix(records, record_type="student"):
    matrix = np.zeros((len(records), len(TOPICS)))
    if record_type == "student":
        for i, rec in enumerate(records):
            for topic, score in rec.get("weakness_scores", {}).items():
                if topic in TOPIC_TO_IDX:
                    matrix[i, TOPIC_TO_IDX[topic]] = score
    else:
        for i, rec in enumerate(records):
            topic  = rec.get("topic", "")
            weight = DIFFICULTY_WEIGHT.get(rec.get("difficulty", "medium"), 1.0)
            if topic in TOPIC_TO_IDX:
                matrix[i, TOPIC_TO_IDX[topic]] = weight
    return normalize(matrix, axis=1, norm="l2")


def recommend(student_matrix, question_matrix, questions, student_idx, top_n=10):
    cohort_baseline = student_matrix.mean(axis=0)
    student_profile = student_matrix[student_idx] - cohort_baseline

    # ── FIX ──────────────────────────────────────────────────────────
    profile_norm    = np.linalg.norm(student_profile)            # FIXED
    student_profile = student_profile / (profile_norm + 1e-10)   # FIXED
    # ─────────────────────────────────────────────────────────────────

    similarities = cosine_similarity(
        student_profile.reshape(1, -1), question_matrix
    ).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [{
        "question_id": questions[idx]["id"],
        "topic":       questions[idx]["topic"],
        "difficulty":  questions[idx]["difficulty"],
        "score":       round(float(similarities[idx]), 4)
    } for idx in top_indices]


def main():
    students = [
        {"name": "Arjun", "weakness_scores": {
            "mechanics": 0.9, "thermodynamics": 0.85, "electrostatics": 0.8,
            "optics": 0.75, "modern_physics": 0.7,
            "organic_chemistry": 0.15, "inorganic_chemistry": 0.1,
            "physical_chemistry": 0.2, "algebra": 0.1, "calculus": 0.15,
            "coordinate_geometry": 0.1, "trigonometry": 0.05}},
        {"name": "Priya", "weakness_scores": {
            "mechanics": 0.1, "thermodynamics": 0.15, "electrostatics": 0.1,
            "optics": 0.05, "modern_physics": 0.2,
            "organic_chemistry": 0.92, "inorganic_chemistry": 0.85,
            "physical_chemistry": 0.88, "algebra": 0.15, "calculus": 0.1,
            "coordinate_geometry": 0.12, "trigonometry": 0.08}},
        {"name": "Rahul", "weakness_scores": {
            "mechanics": 0.15, "thermodynamics": 0.1, "electrostatics": 0.12,
            "optics": 0.08, "modern_physics": 0.1,
            "organic_chemistry": 0.1, "inorganic_chemistry": 0.15,
            "physical_chemistry": 0.12, "algebra": 0.92, "calculus": 0.88,
            "coordinate_geometry": 0.85, "trigonometry": 0.8}},
    ]
    questions = []
    qid = 1
    for topic in TOPICS:
        for diff in ["easy", "medium", "hard"]:
            for _ in range(3):
                questions.append({"id": f"Q{qid:04d}", "topic": topic, "difficulty": diff})
                qid += 1

    sm = build_feature_matrix(students, "student")
    qm = build_feature_matrix(questions, "question")

    for i, student in enumerate(students):
        recs     = recommend(sm, qm, questions, i, top_n=10)
        top_weak = sorted(student["weakness_scores"],
                          key=student["weakness_scores"].get, reverse=True)[:3]
        print(f"\n{'='*60}\nRecommendations for {student['name']}:")
        print(f"  Weakest topics: {top_weak}")
        for r in recs:
            print(f"    {r['question_id']}  topic={r['topic']:<22s}  "
                  f"diff={r['difficulty']:<8s}  score={r['score']}")

    all_recs = {
        students[i]["name"]: {r["question_id"] for r in recommend(sm, qm, questions, i)}
        for i in range(len(students))
    }
    names = list(all_recs.keys())
    print(f"\n{'='*60}\nOverlap (should be 0/10 after fix):")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n = len(all_recs[names[i]] & all_recs[names[j]])
            print(f"  {names[i]} vs {names[j]}: {n}/10 in common")


if __name__ == "__main__":
    main()
