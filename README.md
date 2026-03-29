# 📋 Form Filling Assistant — OpenEnv Environment

An OpenEnv-compliant environment where an AI agent learns to fill real-world forms by extracting information from messy natural language user profiles.

---

## 🌍 Motivation

People fill forms constantly — job applications, KYC documents, contact forms, government registrations. This is a task where agents that can read natural language and extract structured information have immediate, real-world value. This environment trains and evaluates exactly that skill.

---

## 🧠 How It Works

The agent receives:
- A **form** (a set of named fields with expected types)
- A **user profile** (raw, messy natural language text)

The agent must fill in the correct values, **one field per step**, by extracting the right information from the profile.

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `form_fields` | dict | Field names → expected types |
| `filled_fields` | dict | Fields filled so far |
| `user_profile` | str | Raw natural language user info |
| `task_id` | int | 1, 2, or 3 |
| `instructions` | str | Task-specific instructions |
| `remaining_fields` | int | How many fields still to fill |

## ⚡ Action Space

| Field | Type | Description |
|---|---|---|
| `field_name` | str | Name of the field to fill |
| `field_value` | any | Value to fill in |

## 🏆 Reward Function

- **+score/n** per step where score is 0.0–1.0 based on field correctness
- **Partial credit** for approximate matches (e.g. skill lists, addresses)
- **−0.1** penalty for filling an invalid/unknown field name
- Full episode max = **1.0**

---

## 📋 Tasks

### Task 1 — Easy: Contact Form
- **Fields (4):** full_name, email, phone, city
- **Profile:** Clean, direct sentence
- **Expected score for strong model:** ~0.95+

### Task 2 — Medium: Job Application
- **Fields (7):** full_name, date_of_birth, highest_education, years_of_experience, current_job_title, skills, city
- **Profile:** Resume-style paragraph with implicit info
- **Challenge:** Date formatting, numeric extraction, skill normalization
- **Expected score for strong model:** ~0.80–0.90

### Task 3 — Hard: KYC Form
- **Fields (10):** personal + contact + financial info
- **Profile:** Dense structured-but-messy text with abbreviations
- **Challenge:** PAN validation, income conversion (lakhs→rupees), strict format rules
- **Expected score for strong model:** ~0.65–0.80

---

## 🚀 Setup & Usage

### Local

```bash
git clone <repo>
cd form-filling-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t form-filling-env .
docker run -p 7860:7860 form-filling-env
```

### API

```bash
# Reset task 1
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": 1}'

# Fill a field
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"task_id": 1, "field_name": "full_name", "field_value": "Priya Sharma"}'

# Check state
curl http://localhost:7860/state?task_id=1
```

---

## 🤖 Baseline Inference

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN= -groq-api-key
python inference.py
```

### Baseline Scores (llama-3.3-70b-versatile via Groq)

| Task | Difficulty | Score |
|---|---|---|
| Task 1 | Easy | 1.0000 |
| Task 2 | Medium | 0.8571 |
| Task 3 | Hard | 1.0000 |
| **Average** | | **0.9524** |
---

## 📁 Project Structure

```
form-filling-env/
├── app.py           # FastAPI server (reset/step/state endpoints)
├── environment.py   # Core environment logic
├── models.py        # Pydantic typed models
├── tasks.py         # 3 task definitions + graders
├── inference.py     # Baseline inference script
├── openenv.yaml     # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```
