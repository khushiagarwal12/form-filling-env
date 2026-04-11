"""
sanity_test.py - Complete local sanity test for Form Filling Assistant
Run this before every submission to catch issues early.

Usage:
    # Terminal 1: python app.py
    # Terminal 2: python sanity_test.py
"""

import requests
import json
import subprocess
import sys

BASE_URL = "http://localhost:7860"
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, condition, actual=""):
    status = PASS if condition else FAIL
    results.append((status, name, actual))
    print(f"{status} {name}" + (f" → {actual}" if actual else ""))
    return condition


def test_endpoint(method, path, body=None, params=None):
    try:
        url = f"{BASE_URL}{path}"
        if method == "GET":
            r = requests.get(url, params=params, timeout=10)
        else:
            r = requests.post(url, json=body, params=params, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}


print("\n" + "="*55)
print("SANITY TEST — Form Filling Assistant")
print("="*55 + "\n")

# ─────────────────────────────────────────────
# 1. Homepage
# ─────────────────────────────────────────────
print("── 1. Homepage ──")
status, data = test_endpoint("GET", "/")
check("Homepage returns 200", status == 200)
check("Status is ok", data.get("status") == "ok", data.get("status"))
check("Tasks list present", data.get("tasks") == [1, 2, 3], str(data.get("tasks")))

# ─────────────────────────────────────────────
# 2. Reset — all 3 formats
# ─────────────────────────────────────────────
print("\n── 2. Reset endpoint ──")
status, data = test_endpoint("POST", "/reset")
check("Empty reset returns 200", status == 200)
check("Empty reset has form_fields", "form_fields" in data)
check("Empty reset has user_profile", "user_profile" in data)
check("Empty reset filled_fields is empty", data.get("filled_fields") == {})

status, data = test_endpoint("POST", "/reset", params={"task_id": 1})
check("Reset with query param works", status == 200)

status, data = test_endpoint("POST", "/reset", body={"task_id": 2})
check("Reset with JSON body works", status == 200)
check("Reset task_id=2 correct", data.get("task_id") == 2, str(data.get("task_id")))

# ─────────────────────────────────────────────
# 3. Task endpoint
# ─────────────────────────────────────────────
print("\n── 3. Task endpoint ──")
for tid in [1, 2, 3]:
    status, data = test_endpoint("GET", "/task", params={"task_id": tid})
    check(f"Task {tid} returns 200", status == 200)
    check(f"Task {tid} has form_fields", "form_fields" in data)
    check(f"Task {tid} has difficulty", "difficulty" in data)

# ─────────────────────────────────────────────
# 4. Step endpoint
# ─────────────────────────────────────────────
print("\n── 4. Step endpoint ──")
test_endpoint("POST", "/reset", body={"task_id": 1})
status, data = test_endpoint("POST", "/step", body={
    "task_id": 1,
    "field_name": "full_name",
    "field_value": "Priya Sharma"
})
check("Step returns 200", status == 200)
check("Step has observation", "observation" in data)
check("Step has reward", "reward" in data)
check("Step has done", "done" in data)
check("Step reward between 0 and 1",
    0.0 < data.get("reward", {}).get("score", -1) < 1.0,
    str(data.get("reward", {}).get("score")))

# ─────────────────────────────────────────────
# 5. Score endpoint — empty form
# ─────────────────────────────────────────────
print("\n── 5. Score endpoint (empty form) ──")
for tid in [1, 2, 3]:
    test_endpoint("POST", "/reset", body={"task_id": tid})
    status, data = test_endpoint("GET", "/score", params={"task_id": tid})
    score = data.get("final_score", -1)
    check(f"Score task {tid} returns 200", status == 200)
    check(f"Score task {tid} strictly between 0 and 1",
        0.0 < score < 1.0, str(score))

# ─────────────────────────────────────────────
# 6. State endpoint
# ─────────────────────────────────────────────
print("\n── 6. State endpoint ──")
status, data = test_endpoint("GET", "/state", params={"task_id": 1})
check("State returns 200", status == 200)
check("State has task_id", "task_id" in data)
check("State has filled_fields", "filled_fields" in data)

# ─────────────────────────────────────────────
# 7. Score after filling all fields
# ─────────────────────────────────────────────
print("\n── 7. Score after full episode ──")
from environment import FormFillingEnv
from tasks import ALL_TASKS
from models import Action

for tid in [1, 2, 3]:
    env = FormFillingEnv(task_id=tid)
    env.reset()
    gt = ALL_TASKS[tid]["ground_truth"]
    for field, value in gt.items():
        try:
            env.step(Action(field_name=field, field_value=value))
        except:
            pass
    score = env.final_score()
    check(f"Task {tid} perfect score strictly < 1.0", score < 1.0, str(score))
    check(f"Task {tid} perfect score strictly > 0.0", score > 0.0, str(score))

# ─────────────────────────────────────────────
# 8. inference.py scores
# ─────────────────────────────────────────────
print("\n── 8. Inference score range ──")
from environment import FormFillingEnv
from tasks import ALL_TASKS
from models import Action

for tid in [1, 2, 3]:
    env = FormFillingEnv(task_id=tid)
    env.reset()
    score = env.final_score()
    check(f"Task {tid} empty score > 0", score > 0.0, str(score))
    check(f"Task {tid} empty score < 1", score < 1.0, str(score))

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "="*55)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED — ready to push! ✓")
else:
    print("SOME TESTS FAILED — fix before pushing!")
    print("\nFailed tests:")
    for r in results:
        if r[0] == FAIL:
            print(f"  {r[1]} → {r[2]}")
print("="*55)
