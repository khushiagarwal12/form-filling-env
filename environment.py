"""
Form Filling Assistant — OpenEnv Environment

The agent is given a form (set of fields) and a user profile (raw text).
It must fill in the correct values one field at a time.

API:
    env = FormFillingEnv(task_id=1)
    obs = env.reset()
    result = env.step(Action(field_name="full_name", field_value="John Doe"))
    state = env.state()
"""

from typing import Any, Dict, Optional
from models import Observation, Action, Reward, StepResult
from tasks import ALL_TASKS, grade_field, grade_submission


class FormFillingEnv:
    """
    OpenEnv-compliant environment for the Form Filling Assistant task.

    The agent fills one field per step. Episode ends when all fields are
    filled or the agent fills an invalid field.

    Rewards:
        +1.0/n per correct field (n = total fields)
        +0.5/n for partial credit (e.g. partial skill match)
        -0.1  for filling an invalid/unknown field name
        0.0   for an incorrect field value
    """

    def __init__(self, task_id: int = 1):
        if task_id not in ALL_TASKS:
            raise ValueError(f"task_id must be 1, 2, or 3. Got: {task_id}")
        self.task_id = task_id
        self._task = ALL_TASKS[task_id]
        self._filled: Dict[str, Any] = {}
        self._step_count: int = 0
        self._cumulative_score: float = 0.0
        self._done: bool = False

    # ─────────────────────────────────────────
    # reset() — start a fresh episode
    # ─────────────────────────────────────────
    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._filled = {}
        self._step_count = 0
        self._cumulative_score = 0.0
        self._done = False
        return self._build_observation()

    # ─────────────────────────────────────────
    # step() — agent takes an action
    # ─────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        """
        Agent fills one field.
        Returns: StepResult(observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        field_name = action.field_name.strip().lower()
        field_value = action.field_value
        valid_fields = self._task["form_fields"]
        ground_truth = self._task["ground_truth"]
        total_fields = len(ground_truth)

        # ── Penalize invalid field names ──
        if field_name not in valid_fields:
            reward = Reward(
                score=0.0,
                correct=False,
                message=f"'{field_name}' is not a valid field. Valid fields: {list(valid_fields.keys())}",
                cumulative_score=self._cumulative_score
            )
            obs = self._build_observation()
            done = self._check_done()
            return StepResult(observation=obs, reward=reward, done=done, info={"step": self._step_count})

        # ── Score this field ──
        truth_value = ground_truth[field_name]
        field_score = grade_field(field_name, field_value, truth_value, self.task_id)

        # Per-step reward = field_score / total_fields (so full episode = 1.0 max)
        step_reward = round(field_score / total_fields, 4)
        self._cumulative_score = round(
            min(self._cumulative_score + step_reward, 1.0), 4
        )

        # Store the filled value
        self._filled[field_name] = field_value

        # Build message
        if field_score == 1.0:
            message = f"✓ '{field_name}' filled correctly!"
        elif field_score > 0:
            message = f"~ '{field_name}' partially correct (score: {field_score}). Expected: {truth_value}"
        else:
            message = f"✗ '{field_name}' incorrect. Got: '{field_value}'"

        reward = Reward(
            score=step_reward,
            correct=(field_score == 1.0),
            message=message,
            cumulative_score=self._cumulative_score
        )

        done = self._check_done()
        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self._step_count,
                "field_score": field_score,
                "filled_count": len(self._filled),
                "total_fields": total_fields
            }
        )

    # ─────────────────────────────────────────
    # state() — return full current state
    # ─────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        """Return the full internal state of the environment."""
        return {
            "task_id": self.task_id,
            "difficulty": self._task["difficulty"],
            "form_fields": self._task["form_fields"],
            "filled_fields": dict(self._filled),
            "ground_truth": self._task["ground_truth"],
            "user_profile": self._task["user_profile"],
            "step_count": self._step_count,
            "cumulative_score": self._cumulative_score,
            "done": self._done,
            "remaining_fields": len(self._task["form_fields"]) - len(self._filled)
        }

    # ─────────────────────────────────────────
    # Final grade for the full episode
    # ─────────────────────────────────────────
    def final_score(self) -> float:
        """Returns the final graded score (0.0–1.0) for the whole episode."""
        return grade_submission(self.task_id, self._filled)

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────
    def _check_done(self) -> bool:
        """Episode ends when all fields have been filled."""
        all_filled = set(self._task["form_fields"].keys()) == set(self._filled.keys())
        if all_filled:
            self._done = True
        return self._done

    def _build_observation(self) -> Observation:
        """Build an Observation from current state."""
        filled_set = set(self._filled.keys())
        all_fields = set(self._task["form_fields"].keys())
        remaining = len(all_fields - filled_set)

        return Observation(
            form_fields=self._task["form_fields"],
            filled_fields=dict(self._filled),
            user_profile=self._task["user_profile"],
            task_id=self.task_id,
            instructions=self._task["instructions"],
            remaining_fields=remaining
        )
