"""
hf_test.py - Tests your live HuggingFace Space exactly like the validator does
Run this before submitting!
"""

import requests

BASE_URL = "https://khushiagarwal12-form-filling-env.hf.space"

print("\n" + "="*55)
print("TESTING LIVE HUGGINGFACE SPACE")
print("="*55 + "\n")

passed = 0
failed = 0

def check(name, condition, actual=""):
    global passed, failed
    status = "[PASS]" if condition else "[FAIL]"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"{status} {name}" + (f" → {actual}" if actual else ""))

# 1. Homepage
r = requests.get(f"{BASE_URL}/")
check("Homepage 200", r.status_code == 200)
check("Status ok", r.json().get("status") == "ok")

# 2. Reset all formats
r = requests.post(f"{BASE_URL}/reset")
check("Empty reset 200", r.status_code == 200, str(r.status_code))

r = requests.post(f"{BASE_URL}/reset?task_id=1")
check("Reset query param 200", r.status_code == 200, str(r.status_code))

r = requests.post(f"{BASE_URL}/reset", json={"task_id": 1})
check("Reset JSON body 200", r.status_code == 200, str(r.status_code))

# 3. Score on empty form — critical!
for tid in [1, 2, 3]:
    requests.post(f"{BASE_URL}/reset", json={"task_id": tid})
    r = requests.get(f"{BASE_URL}/score?task_id={tid}")
    score = r.json().get("final_score", -1)
    check(f"Score task {tid} strictly between 0 and 1",
          0.0 < score < 1.0, str(score))

# 4. Step
requests.post(f"{BASE_URL}/reset", json={"task_id": 1})
r = requests.post(f"{BASE_URL}/step", json={
    "task_id": 1,
    "field_name": "full_name",
    "field_value": "Priya Sharma"
})
check("Step 200", r.status_code == 200, str(r.status_code))
score = r.json().get("reward", {}).get("score", -1)
check("Step reward between 0 and 1", 0.0 < score < 1.0, str(score))

# 5. Score after step
r = requests.get(f"{BASE_URL}/score?task_id=1")
score = r.json().get("final_score", -1)
check("Score after step between 0 and 1", 0.0 < score < 1.0, str(score))

# 6. Task endpoint
r = requests.get(f"{BASE_URL}/task?task_id=1")
check("Task endpoint 200", r.status_code == 200, str(r.status_code))

# 7. State endpoint
r = requests.get(f"{BASE_URL}/state?task_id=1")
check("State endpoint 200", r.status_code == 200, str(r.status_code))

# Summary
print("\n" + "="*55)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL PASSED — safe to submit!")
else:
    print("FAILURES FOUND — fix before submitting!")
print("="*55)
