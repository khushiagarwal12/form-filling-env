"""
inference.py — Baseline inference script for Form Filling Assistant OpenEnv.

Uses the OpenAI API client to run a model against all 3 tasks.
Reads credentials from environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="gorq-api-key"
    python inference.py
"""

import os
import json
import time
from openai import OpenAI
from environment import FormFillingEnv
from models import Action

# ─────────────────────────────────────────────
# Load credentials from environment variables
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

# Initialize the client
client = None
try:
    import httpx
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        http_client=httpx.Client(transport=httpx.HTTPTransport())
    )
except Exception:
    pass

if client is None:
    try:
        client = OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL
        )
    except Exception as e:
        raise RuntimeError(f"Could not initialize OpenAI client: {e}")

# ─────────────────────────────────────────────
# System prompt for the agent
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a form-filling assistant. You will be given:
1. A form with fields that need to be filled
2. A user profile (raw text) containing the information

Your job is to extract the correct information and fill the form one field at a time.

At each step, respond with a JSON object containing exactly two keys:
{
  "field_name": "the field to fill",
  "field_value": "the extracted value"
}

Rules:
- Only output valid JSON, nothing else
- Fill one field per response
- For dates, use YYYY-MM-DD format
- For phone numbers, use digits only (no dashes or spaces)
- For annual_income, use integer rupees (e.g. 8 lakhs = 800000)
- For employment_type, use: salaried / self-employed / unemployed
- For gender, use: male / female / other
- For skills, use comma-separated lowercase values
"""


def build_user_prompt(obs: dict, filled_so_far: dict) -> str:
    """Build the user message for the LLM from current observation."""
    form_fields = obs["form_fields"]
    filled = obs["filled_fields"]
    remaining = {k: v for k, v in form_fields.items() if k not in filled}

    prompt = f"""User Profile:
{obs['user_profile']}

Form Fields to Fill:
{json.dumps(form_fields, indent=2)}

Already Filled:
{json.dumps(filled, indent=2)}

Remaining Fields: {list(remaining.keys())}

Instructions: {obs['instructions']}

Pick ONE remaining field and fill it. Output only JSON."""
    return prompt


def run_task(task_id: int) -> float:
    """
    Run the baseline agent on a single task.
    Returns the final score (strictly between 0 and 1).
    """
    task_name = f"task_{task_id}"

    print(f"[START] task={task_name}", flush=True)

    env = FormFillingEnv(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    total_fields = len(obs_dict["form_fields"])
    filled_so_far = {}
    done = False
    step = 0
    max_steps = total_fields + 3  # buffer for retries

    while not done and step < max_steps:
        step += 1
        prompt = build_user_prompt(obs_dict, filled_so_far)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON action
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action_data = json.loads(raw.strip())
            field_name = action_data["field_name"]
            field_value = action_data["field_value"]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Step {step}: Failed to parse LLM response - {e}", flush=True)
            print(f"  Raw response: {raw[:200]}", flush=True)
            continue
        except Exception as e:
            print(f"  Step {step}: LLM call failed - {e}", flush=True)
            break

        # Take step in environment
        action = Action(field_name=field_name, field_value=field_value)
        result = env.step(action)

        obs_dict = result.observation.model_dump()
        filled_so_far = obs_dict["filled_fields"]
        done = result.done

        reward = result.reward.score
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

        time.sleep(0.3)  # Avoid rate limiting

    final = env.final_score()
    # Clamp to strictly (0, 1)
    final = max(1e-6, min(final, 1 - 1e-6))
    print(f"[END] task={task_name} score={final:.6f} steps={step}", flush=True)
    return final


# ─────────────────────────────────────────────
# Main - run all 3 tasks and report
# ─────────────────────────────────────────────
if __name__ == "__main__":
    scores = {}
    for task_id in [1, 2, 3]:
        scores[task_id] = run_task(task_id)
        # Clamp again before printing summary
        scores[task_id] = max(1e-6, min(scores[task_id], 1 - 1e-6))

    avg = sum(scores.values()) / len(scores)
    avg = max(1e-6, min(avg, 1 - 1e-6))

    print(f"\n{'='*50}", flush=True)
    print("FINAL BASELINE SCORES", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Task 1 (Easy):   {scores[1]:.6f}", flush=True)
    print(f"  Task 2 (Medium): {scores[2]:.6f}", flush=True)
    print(f"  Task 3 (Hard):   {scores[3]:.6f}", flush=True)
    print(f"  Average:         {avg:.6f}", flush=True)
    print(f"{'='*50}", flush=True)
