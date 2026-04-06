"""
generate_outputs.py
Generates sample_outputs/ for all 10 students + leaderboard.
Run: python3 generate_outputs.py
"""
import json
from pathlib import Path
from data_loader import load_students, load_question_bank
from risk_scorer import score_student_risks, get_student_summary
from adaptive_planner import build_dost_plan

OUT = Path("sample_outputs")
OUT.mkdir(exist_ok=True)


def leaderboard_entry(student, summary, risk_profiles):
    composite = round(
        summary["exam_readiness_pct"] * 0.50 +
        summary["overall_avg_pct"]    * 0.30 +
        summary["completion_rate"]    * 20, 2
    )
    return {
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
    }


def main():
    print("Loading data...")
    students  = load_students()
    questions, qid_index = load_question_bank()
    lb_entries = []

    for s in students:
        sid  = s["student_id"]
        name = s["name"]
        print(f"  {sid}  {name}...")

        risks   = score_student_risks(s)
        summary = get_student_summary(s, risks)
        plan    = build_dost_plan(s, risks, questions,
                                  summary["abort_rate"], summary["avg_skip_rate"])

        with open(OUT / f"{sid}_analysis.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(OUT / f"{sid}_recommendation.json", "w") as f:
            json.dump(plan, f, indent=2)

        lb_entries.append(leaderboard_entry(s, summary, risks))

    lb_entries.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, e in enumerate(lb_entries, 1):
        e["rank"] = i

    with open(OUT / "leaderboard.json", "w") as f:
        json.dump({"leaderboard": lb_entries}, f, indent=2)

    print("\n✓  Done — sample_outputs/")
    print(f"\n{'Rank':<5} {'Name':<22} {'ReadyPct':>9} {'AvgPct':>7} {'Composite':>10}  Top Risk Chapter")
    print("-" * 80)
    for e in lb_entries:
        print(f"{e['rank']:<5} {e['name']:<22} {e['exam_readiness_pct']:>9.1f}% "
              f"{e['overall_avg_pct']:>7.1f}% {e['composite_score']:>10.1f}  "
              f"{e['top_risk_chapter']}")


if __name__ == "__main__":
    main()
