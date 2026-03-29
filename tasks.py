"""
Three tasks for the Form Filling Assistant environment.
Each task has:
- A form definition (fields + expected types)
- A user profile (messy raw text)
- Ground truth answers
- A grader function that scores 0.0–1.0
"""

from typing import Dict, Any


# ─────────────────────────────────────────────
# TASK 1 — Easy: Simple Contact Form
# ─────────────────────────────────────────────
TASK_1 = {
    "task_id": 1,
    "difficulty": "easy",
    "description": "Fill a simple contact form from a user profile.",
    "form_fields": {
        "full_name": "str",
        "email": "str",
        "phone": "str",
        "city": "str"
    },
    "user_profile": (
        "Hi, my name is Priya Sharma. You can reach me at priya.sharma@gmail.com "
        "or call me on +91-9876543210. I currently live in Mumbai."
    ),
    "ground_truth": {
        "full_name": "priya sharma",
        "email": "priya.sharma@gmail.com",
        "phone": "9876543210",
        "city": "mumbai"
    },
    "instructions": (
        "Extract the user's full name, email address, phone number, and city "
        "from the profile text and fill each form field. "
        "Fill one field at a time using the action: {field_name, field_value}."
    )
}

# ─────────────────────────────────────────────
# TASK 2 — Medium: Job Application Form
# ─────────────────────────────────────────────
TASK_2 = {
    "task_id": 2,
    "difficulty": "medium",
    "description": "Fill a job application form from a messy resume-style profile.",
    "form_fields": {
        "full_name": "str",
        "date_of_birth": "str",       # expected: YYYY-MM-DD
        "highest_education": "str",
        "years_of_experience": "int",
        "current_job_title": "str",
        "skills": "str",              # comma-separated
        "city": "str"
    },
    "user_profile": (
        "My name is Arjun Mehta, born on 15th March 1995. "
        "I completed my B.Tech in Computer Science from IIT Delhi in 2017. "
        "I've been working as a Software Engineer for about 6 years now, "
        "currently at a startup in Bangalore. "
        "I know Python, JavaScript, SQL, and a bit of Docker. "
        "Looking for new opportunities!"
    ),
    "ground_truth": {
        "full_name": "arjun mehta",
        "date_of_birth": "1995-03-15",
        "highest_education": "b.tech",
        "years_of_experience": 6,
        "current_job_title": "software engineer",
        "skills": "python, javascript, sql, docker",
        "city": "bangalore"
    },
    "instructions": (
        "Extract the required job application details from the profile. "
        "Format date_of_birth as YYYY-MM-DD. "
        "List skills as comma-separated lowercase values. "
        "Fill one field at a time."
    )
}

# ─────────────────────────────────────────────
# TASK 3 — Hard: Multi-section KYC Form
# ─────────────────────────────────────────────
TASK_3 = {
    "task_id": 3,
    "difficulty": "hard",
    "description": "Fill a multi-section KYC (Know Your Customer) form with validation rules.",
    "form_fields": {
        # Personal
        "full_name": "str",
        "date_of_birth": "str",       # YYYY-MM-DD
        "gender": "str",              # male/female/other
        "nationality": "str",
        # Contact
        "email": "str",
        "phone": "str",
        "address": "str",
        # Financial
        "annual_income": "int",       # in INR, numeric only
        "employment_type": "str",     # salaried/self-employed/unemployed
        "pan_number": "str",          # format: ABCDE1234F (10 chars)
    },
    "user_profile": (
        "Full Name: Kavya Nair | DOB: 7 July 1990 | Female | Indian citizen. "
        "Email: kavya.nair@outlook.com, Phone: 9845012345. "
        "Address: 42, MG Road, Kochi, Kerala - 682001. "
        "She works as a freelance graphic designer (self-employed). "
        "Annual earnings roughly around 8 lakhs per year. "
        "PAN: BXYPK7823G"
    ),
    "ground_truth": {
        "full_name": "kavya nair",
        "date_of_birth": "1990-07-07",
        "gender": "female",
        "nationality": "indian",
        "email": "kavya.nair@outlook.com",
        "phone": "9845012345",
        "address": "42, mg road, kochi, kerala - 682001",
        "annual_income": 800000,
        "employment_type": "self-employed",
        "pan_number": "BXYPK7823G"
    },
    "instructions": (
        "Fill all 10 fields of this KYC form carefully. Rules: "
        "1) date_of_birth must be YYYY-MM-DD format. "
        "2) gender must be one of: male/female/other. "
        "3) annual_income must be an integer in INR (8 lakhs = 800000). "
        "4) employment_type must be: salaried/self-employed/unemployed. "
        "5) pan_number must be exactly 10 characters in format ABCDE1234F. "
        "Fill one field at a time."
    )
}

ALL_TASKS = {1: TASK_1, 2: TASK_2, 3: TASK_3}


# ─────────────────────────────────────────────
# Grader — scores a filled form 0.0 to 1.0
# ─────────────────────────────────────────────

def normalize(value: Any) -> str:
    """Normalize a value for comparison: lowercase string, strip spaces."""
    if value is None:
        return ""
    return str(value).lower().strip().replace("  ", " ")


def grade_field(field_name: str, agent_value: Any, truth_value: Any, task_id: int) -> float:
    """
    Score a single field. Returns 1.0 (correct), 0.5 (partial), or 0.0 (wrong).
    """
    agent_norm = normalize(agent_value)
    truth_norm = normalize(truth_value)

    # Exact match
    if agent_norm == truth_norm:
        return 1.0

    # Numeric fields — allow ±5% tolerance
    if field_name in ("years_of_experience", "annual_income"):
        try:
            a = float(agent_norm.replace(",", ""))
            t = float(truth_norm.replace(",", ""))
            if t != 0 and abs(a - t) / t <= 0.05:
                return 1.0
        except ValueError:
            pass

    # Skills — partial credit for each correct skill
    if field_name == "skills":
        agent_skills = set(s.strip() for s in agent_norm.split(","))
        truth_skills = set(s.strip() for s in truth_norm.split(","))
        if not truth_skills:
            return 0.0
        overlap = len(agent_skills & truth_skills)
        return round(overlap / len(truth_skills), 2)

    # Phone — strip non-digits
    if field_name == "phone":
        agent_digits = "".join(filter(str.isdigit, agent_norm))
        truth_digits = "".join(filter(str.isdigit, truth_norm))
        return 1.0 if agent_digits == truth_digits else 0.0

    # Date — accept common alternate formats
    if field_name == "date_of_birth":
        # Already checked exact match above
        # Try partial: at least year and month correct
        parts_a = agent_norm.replace("/", "-").replace(".", "-").split("-")
        parts_t = truth_norm.split("-")
        if len(parts_a) == 3 and len(parts_t) == 3:
            year_ok = parts_a[0] == parts_t[0]
            month_ok = parts_a[1].zfill(2) == parts_t[1].zfill(2)
            day_ok = parts_a[2].zfill(2) == parts_t[2].zfill(2)
            if year_ok and month_ok and day_ok:
                return 1.0
            elif year_ok and month_ok:
                return 0.5

    # Partial string match for address (long fields)
    if field_name == "address":
        words_agent = set(agent_norm.split())
        words_truth = set(truth_norm.split())
        if not words_truth:
            return 0.0
        overlap = len(words_agent & words_truth)
        score = overlap / len(words_truth)
        return min(round(score, 2), 1.0)

    return 0.0


def grade_submission(task_id: int, filled_fields: Dict[str, Any]) -> float:
    """
    Grade a fully (or partially) filled form.
    Returns a score between 0.0 and 1.0.
    """
    task = ALL_TASKS[task_id]
    ground_truth = task["ground_truth"]
    total_fields = len(ground_truth)

    if total_fields == 0:
        return 0.0

    total_score = 0.0
    for field_name, truth_value in ground_truth.items():
        agent_value = filled_fields.get(field_name, None)
        field_score = grade_field(field_name, agent_value, truth_value, task_id)
        total_score += field_score

    return round(total_score / total_fields, 4)
