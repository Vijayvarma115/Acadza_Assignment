# Acadza Student Recommender — BKT + Ebbinghaus Engine

## Setup & Running

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open: http://127.0.0.1:8000/docs
```

Or generate all outputs without the server:
```bash
python3 generate_outputs.py
```

---

## The Approach — Why It's Different

Most systems ask: *"What did the student score last week?"*
This system asks: *"What will the student actually know on exam day?"*

That's a fundamentally different question, and it requires two models working together.

---

### Layer 1 — Bayesian Knowledge Tracing (BKT)

**Reference:** Corbett & Anderson, 1994. Used in production at Carnegie Learning, ASSISTments, Khan Academy.

Raw scores are noisy. A student who scores 40% might actually know 60% of the material — they ran out of time, skipped questions they knew, or got unlucky on a few. A student who scores 70% might have guessed correctly on 30% of those.

BKT models knowledge as a **hidden binary state** — KNOWN or UNKNOWN — that we can never observe directly, only infer from performance. The update formula uses four parameters:

| Parameter | Meaning | Physics | Chemistry | Maths |
|---|---|---|---|---|
| P(L0) | Prior probability of knowing before any attempt | 0.30 | 0.30 | 0.35 |
| P(T) | Probability of learning per attempt | 0.08 | 0.09 | 0.10 |
| P(G) | Guess rate — gets it right without knowing | 0.20 | 0.22 | 0.18 |
| P(S) | Slip rate — knows it but still gets it wrong | 0.15 | 0.12 | 0.10 |

These are tuned for JEE specifically: P(T) is low because JEE concepts don't click in one session. P(S) is higher than standard because timed pressure causes careless errors.

After each attempt: P(known | evidence) is computed via Bayes rule, then adjusted for the learning transition. The result is a credible belief about the student's current knowledge state — not an average.

**Why this beats averaging:**
- avg_pct of [90, 10, 90] = 63.3% → "medium, assign medium practice"
- BKT of [90, 10, 90]: P(known) steadily climbs, accounts for the dip, correctly identifies partial mastery
- avg_pct of [10, 10, 15] = 11.7% → just "weak"
- BKT of [10, 10, 15]: P(known)=0.08 and decreasing → diagnosis: DEEPLY_UNKNOWN, needs concept first

---

### Layer 2 — Ebbinghaus Forgetting Curve

**Reference:** Hermann Ebbinghaus, 1885. The empirical basis for Anki, SuperMemo, Duolingo.

Even if P(known) is high, that knowledge **decays over time**. The formula:

```
R(t) = e^(−t / S)
```

Where R(t) is the probability of recall after t days, and S is **memory stability** — how long this particular memory holds before decaying to 37% (1/e ≈ 0.37).

Stability grows with each successful review:
```
S = S_base × (2.2 ^ reviews) × performance_multiplier
```

Physics formulas decay faster (S_base=3.5 days) than Chemistry rules (S_base=4.5 days). Each review roughly doubles how long the memory holds.

---

### The Core Metric — P(known at exam)

```
P(known_at_exam) = P(known_now)  ×  R(days_since_review + days_to_exam)
```

This projects each chapter forward to exam day. A chapter with P(known)=0.80 that was last studied 25 days ago and the exam is 7 days away:

```
total_decay = 25 + 7 = 32 days
R(32) = e^(−32/S)  
```

If S=8 days (studied twice): R ≈ 0.018 → P(exam) ≈ 0.014 → **CRITICAL risk**

This is why **Neha Joshi (78% average, class topper) gets a 3.5% exam readiness score** — she's been away from her chapters for weeks. Her knowledge exists but it's decaying. The system correctly prescribes revision + spaced practice, not concept or formula basics.

---

### Differential Diagnosis

Same symptom (failing a chapter) can have different causes. The planner identifies WHY before prescribing treatment:

| Diagnosis | P(known) | Stability | Prescription |
|---|---|---|---|
| NEVER_LEARNED | < 0.25, ≤1 attempt | any | concept → formula → easy assignment |
| DEEPLY_UNKNOWN | < 0.35, multiple attempts | any | formula → [pickingPower if skip>25%] → easy assignment |
| LEARNED_UNSTABLE | 0.35–0.55 | < 5 days | medium assignment → clickingPower |
| STABLE_DECAYING | ≥ 0.55 | any | revision → medium assignment |
| EXAM_PRESSURE | ≥ 0.55, high R | good | speedRace → practiceTest |

This is the critical distinction. Two students both failing Thermodynamics: one has NEVER_LEARNED (needs concept), the other has STABLE_DECAYING (needs revision, not concept — they knew it, they just forgot). Giving both the same treatment is what every naive system does.

---

## Handling the Marks Field

Six formats in the raw data, handled in cascade:

| Format | Example | Rule |
|---|---|---|
| `X/Y (Z%)` | `"49/120 (40.8%)"` | Extract Z — most precise |
| `X/Y` | `"39/100"` | X÷Y × 100 |
| `+X -Y` | `"+48 -8"` | (X−Y) ÷ (total_q×4) × 100 |
| Plain number | `"22"`, `49` | value ÷ (total_q×4) × 100 |

**Assumption:** Max marks = total_questions × 4 (JEE Mains: +4 correct, −1 wrong). Verified: 25-question test → max 100, 30-question test → max 120, matches all fraction-format entries.

---

## Leaderboard Formula

```
composite = exam_readiness_pct × 0.50
           + overall_avg_pct   × 0.30
           + completion_rate   × 20
```

Exam readiness (forward-looking) gets the most weight. A student who scored 70% last month but hasn't reviewed since is less prepared than one who scored 50% and reviewed everything last week.

---

## Debug Task — The Bug in recommender_buggy.py

**Location:** `recommend()`, the 3-line normalization block.

```python
# BUGGY — computes norm of wrong vector, then discards the student gap entirely
cohort_baseline = student_matrix.mean(axis=0)
student_profile = student_matrix[student_idx] - cohort_baseline  # gap computed correctly

profile_norm    = np.linalg.norm(cohort_baseline)                 # ← BUG: wrong vector
student_profile = cohort_baseline / (profile_norm + 1e-10)        # ← BUG: student_profile overwritten with cohort average
```

The gap vector computed on line 2 is discarded on line 3. Every student ends up with the same query vector — the normalized cohort average. Result: 10/10 overlap regardless of how different the weakness profiles are.

**Fix:** Two variable name changes.
```python
profile_norm    = np.linalg.norm(student_profile)           # student_profile, not cohort_baseline
student_profile = student_profile / (profile_norm + 1e-10)  # same
```

After fix: 0/10 overlap. Arjun gets Physics questions, Priya gets Chemistry, Rahul gets Maths.

**Why AI misses it:** LLMs pattern-match on code structure. The 3-line block looks like standard cosine-similarity preprocessing — `compute_something`, `compute_norm`, `normalize`. The actual values being passed aren't semantically verified, only syntactically scanned. The only reliable way to catch this: run the code, print the overlap, see 10/10, then trace manually.

---

## What I'd Improve Given More Time

**1. Per-question BKT instead of per-session:** The current model uses session-level P(correct) as evidence. In a real system you'd have per-question correctness — richer signal, more accurate BKT.

**2. SM-2 scheduling:** The forgetting model currently computes risk passively. A full SM-2 implementation would also schedule the *next optimal review date* for each chapter — turning the plan from a list of DOSTs into a calendar.

**3. Subject prerequisite graph:** Thermodynamics depends on Laws of Motion which depends on Kinematics. BKT for Thermodynamics should partially inherit uncertainty from upstream nodes. A Bayesian network over chapter prerequisites would give much more accurate P(known) estimates.

**4. Multi-concept questions:** A single JEE question often tests 2–3 chapters simultaneously. Time spent on the slowest question is actually evidence about specific subtopics — the outlier question data in student_performance.json is underused here.

**5. Calibrate BKT parameters per student:** Right now P(T) and P(S) are fixed defaults. With enough data you'd estimate these per student using EM — some students learn fast (high P(T)), others make more careless errors (high P(S)).
* **View UI:** `http://127.0.0.1:8000/docs`
* **Batch Output:** `python3 generate_outputs.py`
* **Debug Fix:** Compare buggy vs fixed recommenders


