"""
Three tasks for the Form Filling Assistant environment.
"""

from typing import Dict, Any


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

TASK_2 = {
    "task_id": 2,
    "difficulty": "medium",
    "description": "Fill a job application form from a messy resume-style profile.",
    "form_fields": {
        "full_name": "str",
        "date_of_birth": "str",
        "highest_education": "str",
        "years_of_experience": "int",
        "current_job_title": "str",
        "skills": "str",
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

TASK_3 = {
    "task_id": 3,
    "difficulty": "hard",
    "description": "Fill a multi-section KYC form with contradictory info and implicit values.",
    "form_fields": {
        "full_name": "str",
        "date_of_birth": "str",
        "gender": "str",
        "nationality": "str",
        "email": "str",
        "phone": "str",
        "address": "str",
        "annual_income": "int",
        "employment_type": "str",
        "pan_number": "str",
    },
    "user_profile": (
        "Hi, I'm Kavya Nair — though my friends call me Kay. "
        "I was born in the summer of 1990, specifically the 7th of July. "
        "You can reach me at kavya.nair@outlook.com — NOT my old id kavya1990@yahoo.com which I no longer use. "
        "My number is 9845012345 but my sister's number 9880001234 is sometimes listed under my name — please use mine. "
        "I live at 42, MG Road, Kochi, Kerala - 682001. "
        "I'm a freelance graphic designer — so self-employed. "
        "Last year I made somewhere between 7.5 and 8.5 lakhs, let's say roughly 8 lakhs. "
        "I'm Indian, female. "
        "My PAN is BXYPK7823G but my husband's PAN is ZZZZZ9999Z — make sure you use mine. "
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
        "Fill all 10 fields carefully. Watch out for: "
        "1) Multiple email addresses — use the current one. "
        "2) Multiple phone numbers — use the applicant's own. "
        "3) Multiple PAN numbers — use the applicant's own. "
        "4) Income given as a range — use the midpoint as an integer in INR. "
        "5) Date of birth given descriptively — convert to YYYY-MM-DD. "
        "6) employment_type must be: salaried/self-employed/unemployed. "
        "Fill one field at a time."
    )
}

ALL_TASKS = {1: TASK_1, 2: TASK_2, 3: TASK_3}

EPS = 1e-6


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower().strip().replace("  ", " ")


def grade_field(field_name: str, agent_value: Any, truth_value: Any, task_id: int) -> float:
    agent_norm = normalize(agent_value)
    truth_norm = normalize(truth_value)

    if agent_norm == truth_norm:
        return 1.0

    if field_name in ("years_of_experience", "annual_income"):
        try:
            a = float(agent_norm.replace(",", ""))
            t = float(truth_norm.replace(",", ""))
            if t != 0 and abs(a - t) / t <= 0.05:
                return 1.0
        except ValueError:
            pass

    if field_name == "skills":
        agent_skills = set(s.strip() for s in agent_norm.split(","))
        truth_skills = set(s.strip() for s in truth_norm.split(","))
        if not truth_skills:
            return 0.0
        overlap = len(agent_skills & truth_skills)
        return round(overlap / len(truth_skills), 2)

    if field_name == "phone":
        agent_digits = "".join(filter(str.isdigit, agent_norm))
        truth_digits = "".join(filter(str.isdigit, truth_norm))
        return 1.0 if agent_digits == truth_digits else 0.0

    if field_name == "date_of_birth":
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
    Grade a fully or partially filled form.
    Returns a score strictly between 0.0 and 1.0 (exclusive).
    """
    task = ALL_TASKS[task_id]
    ground_truth = task["ground_truth"]
    total_fields = len(ground_truth)

    if total_fields == 0:
        return EPS

    total_score = 0.0
    for field_name, truth_value in ground_truth.items():
        agent_value = filled_fields.get(field_name, None)
        field_score = grade_field(field_name, agent_value, truth_value, task_id)
        total_score += field_score

    raw = total_score / total_fields

    # Force strictly inside (0, 1) — validator requires this
    if raw <= 0.0:
        return EPS
    if raw >= 1.0:
        return 1.0 - EPS

    return raw