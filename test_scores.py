from environment import FormFillingEnv
from tasks import ALL_TASKS
from models import Action

print("Testing score clamping...")

# Test empty form (score should be > 0.0)
for task_id in [1, 2, 3]:
    env = FormFillingEnv(task_id=task_id)
    env.reset()
    score = env.final_score()
    status = "PASS" if 0.0 < score < 1.0 else "FAIL"
    print(f"[{status}] Task {task_id} empty score: {score}")

# Test perfect form (score should be < 1.0)
for task_id in [1, 2, 3]:
    env = FormFillingEnv(task_id=task_id)
    env.reset()
    ground_truth = ALL_TASKS[task_id]["ground_truth"]
    for field, value in ground_truth.items():
        try:
            env.step(Action(field_name=field, field_value=value))
        except:
            pass
    score = env.final_score()
    status = "PASS" if 0.0 < score < 1.0 else "FAIL"
    print(f"[{status}] Task {task_id} perfect score: {score}")

print("Done!")