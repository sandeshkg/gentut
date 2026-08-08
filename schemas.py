from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class CognitiveState(BaseModel):
    student_id: str
    current_topic: str
    skill_level: Literal["novice", "developing", "proficient"] = "novice"
    misconceptions: List[str] = Field(default_factory=list)
    hint_stage: int = Field(0, description="0=no hint given, increments per escalating hint")
    mastery_score: float = Field(0.0, ge=0.0, le=1.0)

class TutorAction(BaseModel):
    next_action: Literal["give_hint", "ask_clarifying_question", "present_new_content", "mark_mastered"]
    reasoning: str = Field(description="Brief explanation of why this action was chosen")
    target_misconception: Optional[str] = Field(default=None, description="Which misconception this action addresses, if any")

class TutorContent(BaseModel):
    content_type: Literal["hint", "clarifying_question", "new_content", "mastery_message"]
    message: str = Field(description="The actual text shown to the student")
    difficulty_note: Optional[str] = Field(default=None, description="Internal note on pacing/difficulty, not shown to student")
