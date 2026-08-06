from pydantic import BaseModel, Field
from typing import Literal, List

class CognitiveState(BaseModel):
    student_id: str
    current_topic: str
    skill_level: Literal["novice", "developing", "proficient"] = "novice"
    misconceptions: List[str] = Field(default_factory=list)
    hint_stage: int = Field(0, description="0=no hint given, increments per escalating hint")
    mastery_score: float = Field(0.0, ge=0.0, le=1.0)