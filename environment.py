"""
Form Filling Assistant — OpenEnv Environment
"""

from typing import Any, Dict
from models import Observation, Action, Reward, StepResult
from tasks import ALL_TASKS, grade_field, grade_submission

MIN_SCORE = 0.1
MAX_SCORE = 0.9


class FormFillingEnv:
    def __init__(self, task_id: int = 1):
        if task_id not in ALL_TASKS:
            raise ValueError(f"task_id must be 1, 2, or 3. Got: {task_id}")
        self.task_id = task_id
        self._task = ALL_TASKS[task_id]
        self._filled: Dict[str, Any] = {}
        self._step_count: int = 0
        self._cumulative_score: float = 0.0
        self._done: bool = False

    def reset(self) -> Observation:
        self._filled = {}
        self._step_count = 0
        self._cumulative_score = 0.0
        self._done = False
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        field_name = action.field_name.strip().lower()
        field_value = action.field_value

        valid_fields = self._task["form_fields"]
        ground_truth = self._task["ground_truth"]
        total_fields = len(ground_truth)

        if field_name not in valid_fields:
            reward = Reward(
                score=MIN_SCORE,
                correct=False,
                message=f"'{field_name}' is not a valid field",
                cumulative_score=self._cumulative_score
            )
            return StepResult(
                observation=self._build_observation(),
                reward=reward,
                done=self._check_done(),
                info={"step": self._step_count}
            )

        truth_value = ground_truth[field_name]
        field_score = grade_field(field_name, field_value, truth_value, self.task_id)

        step_reward = field_score / total_fields
        self._cumulative_score = min(self._cumulative_score + step_reward, MAX_SCORE)
        self._filled[field_name] = field_value

        reward = Reward(
            score=step_reward,
            correct=(field_score >= MAX_SCORE),
            message=f"{field_name} scored {field_score:.4f}",
            cumulative_score=self._cumulative_score
        )

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._check_done(),
            info={"step": self._step_count}
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self._task["difficulty"],
            "form_fields": self._task["form_fields"],
            "filled_fields": dict(self._filled),
            "user_profile": self._task["user_profile"],
            "step_count": self._step_count,
            "cumulative_score": self._cumulative_score,
            "done": self._done,
            "remaining_fields": len(self._task["form_fields"]) - len(self._filled)
        }

    def final_score(self) -> float:
        raw = grade_submission(self.task_id, self._filled)
        return max(MIN_SCORE, min(float(raw), MAX_SCORE))

    def _check_done(self) -> bool:
        if set(self._task["form_fields"].keys()) == set(self._filled.keys()):
            self._done = True
        return self._done

    def _build_observation(self) -> Observation:
        remaining = len(self._task["form_fields"]) - len(self._filled)
        return Observation(
            form_fields=self._task["form_fields"],
            filled_fields=dict(self._filled),
            user_profile=self._task["user_profile"],
            task_id=self.task_id,
            instructions=self._task["instructions"],
            remaining_fields=remaining
        )