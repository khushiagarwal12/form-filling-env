"""
Three tasks for the Form Filling Assistant environment.
"""

from typing import Dict, Any

EPS = 1e-6


def safe(score: float) -> float:
    try:
        score = float(score)
    except:
        return EPS
    if score <= 0.0:
        return EPS
    if score >= 1.0:
        return 1.0 - EPS
    return score


TASK_1 = {
    "task_id": 1,
    "difficulty": "easy",
    "description": "Fill a simple contact form from a user profile.",
    "form_fields": {
        "full_name": "str",
        "email": "str",
        "phone": "str",
        "city": "str",
        "country_code": "str"
    },
    "user_profile": (
        "Hi, my name is Priya Sharma. You can reach me at priya.sharma@gmail.com "
        "or call me on +91-9876543210. I currently live in Mumbai, India."
    ),
    "ground_truth": {
        "full_name": "priya sharma",
        "email": "priya.sharma@gmail.com",
        "phone": "9876543210",
        "city": "mumbai",
        "country_code": "356"
    },
    "instructions": (
        "Extract the user's full name, email address, phone number, city "
        "and 2-letter country code (e.g. 'in' for India) from the profile. "
        "Fill one field at a time."
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
        "I know Python, JavaScript, SQL, and Docker."
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
        "Extract structured information. Format DOB as YYYY-MM-DD. "
        "Skills must be comma-separated lowercase. "
        "Fill one field at a time."
    )
}

TASK_3 = {
    "task_id": 3,
    "difficulty": "hard",
    "description": "Fill a complex KYC form with ambiguous and contradictory info.",
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
        "marital_status": "str"
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
        "We got married in 2018."
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
        "pan_number": "BXYPK7823G",
        "marital_status": "married since 2018"
    },
    "instructions": (
        "Fill all 11 fields carefully. Watch out for: "
        "1) Multiple email addresses — use the current one. "
        "2) Multiple phone numbers — use the applicant's own. "
        "3) Multiple PAN numbers — use the applicant's own. "
        "4) Income given as a range — use the midpoint as integer in INR. "
        "5) Date of birth given descriptively — convert to YYYY-MM-DD. "
        "6) employment_type must be: salaried/self-employed/unemployed. "
        "7) marital_status must include year of marriage. "
        "Fill one field at a time."
    )
}

ALL_TASKS = {1: TASK_1, 2: TASK_2, 3: TASK_3}


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower().strip().replace("  ", " ")


def grade_field(field_name: str, agent_value: Any, truth_value: Any, task_id: int) -> float:
    a = normalize(agent_value)
    t = normalize(truth_value)

    if a == t:
        return 1.0 - EPS

    if field_name in ("years_of_experience", "annual_income"):
        try:
            av = float(a.replace(",", ""))
            tv = float(t.replace(",", ""))
            if tv != 0 and abs(av - tv) / tv <= 0.05:
                return 1.0 - EPS
        except:
            pass

    if field_name == "skills":
        a_set = set(x.strip() for x in a.split(",") if x.strip())
        t_set = set(x.strip() for x in t.split(",") if x.strip())
        if not t_set:
            return EPS
        overlap = len(a_set & t_set)
        return safe(overlap / len(t_set))

    if field_name == "phone":
        a_digits = "".join(filter(str.isdigit, a))
        t_digits = "".join(filter(str.isdigit, t))
        return 1.0 - EPS if a_digits == t_digits else EPS

    if field_name == "date_of_birth":
        a_parts = a.replace("/", "-").replace(".", "-").split("-")
        t_parts = t.split("-")
        if len(a_parts) == 3 and len(t_parts) == 3:
            year_ok = a_parts[0] == t_parts[0]
            month_ok = a_parts[1].zfill(2) == t_parts[1].zfill(2)
            day_ok = a_parts[2].zfill(2) == t_parts[2].zfill(2)
            if year_ok and month_ok and day_ok:
                return 1.0 - EPS
            elif year_ok and month_ok:
                return 0.5
        return EPS

    if field_name == "address":
        a_words = set(a.split())
        t_words = set(t.split())
        if not t_words:
            return EPS
        return safe(len(a_words & t_words) / len(t_words))

    return EPS


def grade_submission(task_id: int, filled_fields: Dict[str, Any]) -> float:
    """Grade a filled form. Returns score strictly between 0 and 1."""
    task = ALL_TASKS[task_id]
    gt = task["ground_truth"]
    n = len(gt)

    if n == 0:
        return EPS

    total = 0.0
    for k, v in gt.items():
        total += grade_field(k, filled_fields.get(k), v, task_id)

    return safe(total / n)