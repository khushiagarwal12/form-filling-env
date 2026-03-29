"""
FastAPI server for the Form Filling Assistant OpenEnv environment.
Exposes /reset, /step, /state endpoints as required by the OpenEnv spec.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

from environment import FormFillingEnv
from models import Action

app = FastAPI(
    title="Form Filling Assistant — OpenEnv",
    description="An AI environment where agents learn to fill real-world forms from messy user profiles.",
    version="1.0.0"
)

# Global env instances — one per task
_envs: Dict[int, FormFillingEnv] = {
    1: FormFillingEnv(task_id=1),
    2: FormFillingEnv(task_id=2),
    3: FormFillingEnv(task_id=3),
}


class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    task_id: int = 1
    field_name: str
    field_value: Any


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "environment": "Form Filling Assistant", "tasks": [1, 2, 3]}


# ─────────────────────────────────────────────
# reset()
# ─────────────────────────────────────────────
@app.post("/reset")
def reset(task_id: int = 1, request: Optional[ResetRequest] = None):
    """Reset the environment for a given task and return initial observation."""
    if request is not None:
        task_id = request.task_id
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")
    obs = _envs[task_id].reset()
    return obs.model_dump()


# ─────────────────────────────────────────────
# step()
# ─────────────────────────────────────────────
@app.post("/step")
def step(request: StepRequest):
    """Agent fills one field. Returns observation, reward, done, info."""
    task_id = request.task_id
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

    action = Action(field_name=request.field_name, field_value=request.field_value)
    try:
        result = _envs[task_id].step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump()


# ─────────────────────────────────────────────
# state()
# ─────────────────────────────────────────────
@app.get("/state")
def state(task_id: int = 1):
    """Return the full current state of the environment."""
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")
    return _envs[task_id].state()

# ─────────────────────────────────────────────
# task()
# ─────────────────────────────────────────────
@app.get("/task")
def get_task(task_id: int = 1):
    """Return task definition for a given task_id."""
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")
    from tasks import ALL_TASKS
    task = ALL_TASKS[task_id]
    return {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "description": task["description"],
        "form_fields": task["form_fields"],
        "instructions": task["instructions"]
    }


# ─────────────────────────────────────────────
# final_score()
# ─────────────────────────────────────────────
@app.get("/score")
def score(task_id: int = 1):
    """Return the final graded score for the current episode."""
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")
    return {"task_id": task_id, "final_score": _envs[task_id].final_score()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
