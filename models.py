"""
Typed Pydantic models for the Form Filling Assistant OpenEnv environment.
Defines Observation, Action, and Reward models per the OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class Observation(BaseModel):
    """
    What the agent sees at each step.
    - form_fields: dict of field_name -> expected type (e.g. {"name": "str", "age": "int"})
    - filled_fields: dict of field_name -> value filled so far
    - user_profile: raw messy text describing the user
    - task_id: which task is being run (1, 2, or 3)
    - instructions: natural language description of what the agent must do
    """
    form_fields: Dict[str, str] = Field(..., description="Fields to fill: name -> expected type")
    filled_fields: Dict[str, Any] = Field(default_factory=dict, description="Fields filled so far")
    user_profile: str = Field(..., description="Raw user info text to extract data from")
    task_id: int = Field(..., description="Task number: 1 (easy), 2 (medium), 3 (hard)")
    instructions: str = Field(..., description="What the agent must do")
    remaining_fields: int = Field(..., description="How many fields still need to be filled")


class Action(BaseModel):
    """
    What the agent does at each step.
    - field_name: the form field to fill
    - field_value: the value to put in that field
    The agent fills one field per step.
    """
    field_name: str = Field(..., description="The name of the form field to fill")
    field_value: Any = Field(..., description="The value to put in the field")


class Reward(BaseModel):
    """
    Reward signal returned after each step.
    - score: float between 0.0 and 1.0
    - correct: whether this specific field was filled correctly
    - message: human-readable feedback
    - cumulative_score: total score so far in this episode
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Reward for this step (0.0 to 1.0)")
    correct: bool = Field(..., description="Was this field filled correctly?")
    message: str = Field(..., description="Feedback message")
    cumulative_score: float = Field(..., ge=0.0, le=1.0, description="Total score so far")


class StepResult(BaseModel):
    """Full result returned by step()"""
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Is the episode complete?")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")
