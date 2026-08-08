import re
from schemas import CognitiveState

def extract_json(raw: str) -> str:
    """Pull the first {...} block out of raw model output, stripping prose/markdown fences."""
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return match.group(0) if match else raw

def build_skill_identifier_prompt(student_message: str) -> str:
    example = CognitiveState(
        student_id="s001",
        current_topic="recursion",
        skill_level="novice",
        misconceptions=["confuses base case with recursive case"],
        hint_stage=1,
        mastery_score=0.2
    ).model_dump_json(indent=2)

    return f"""You are a Skill Identifier agent. Extract the student's cognitive state from their message.

Output ONLY a raw JSON object. No prose, no markdown code fences, no explanation before or after.

Field rules:
- student_id: any string identifier
- current_topic: short string, the topic the student is asking about
- skill_level: MUST be exactly one of these three strings: "novice", "developing", "proficient"
- misconceptions: a list of short strings describing likely misunderstandings
- hint_stage: MUST be an integer (0, 1, 2, 3...), NOT a string label
- mastery_score: MUST be a float between 0.0 and 1.0, NOT a percentage

Example of a correctly formatted output:
{example}

Student message: "{student_message}"

JSON output:"""

def skill_identifier_agent(student_message: str, model, tokenizer) -> CognitiveState:
    """Text -> CognitiveState. Raises pydantic.ValidationError on failure (let caller log it)."""
    prompt = build_skill_identifier_prompt(student_message)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, max_length=None)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_json(raw)
    return CognitiveState.model_validate_json(cleaned)
