import re
from schemas import CognitiveState, TutorAction

def extract_json(raw: str) -> str:
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
- current_topic: short string, the topic the student is asking about. If the message is unclear, gibberish, or has no identifiable topic, use exactly "unclear" as the value — never output null.
- skill_level: MUST be exactly one of these three strings: "novice", "developing", "proficient"
- misconceptions: a list of short strings describing likely misunderstandings
- hint_stage: MUST be an integer (0, 1, 2, 3...), NOT a string label
- mastery_score: MUST be a float between 0.0 and 1.0, NOT a percentage

Example of a correctly formatted output:
{example}

Student message: "{student_message}"

JSON output:"""

def skill_identifier_agent(student_message: str, model, tokenizer):
    """Returns (state, raw_output). state is None if validation failed."""
    prompt = build_skill_identifier_prompt(student_message)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, max_length=None)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_json(raw)
    try:
        state = CognitiveState.model_validate_json(cleaned)
        return state, raw
    except Exception:
        return None, raw

def build_profiler_prompt(state: CognitiveState) -> str:
    example = TutorAction(
        next_action="give_hint",
        reasoning="Student has attempted twice and still shows the same misconception; a targeted hint is more useful than new content.",
        target_misconception="confuses base case with recursive case"
    ).model_dump_json(indent=2)

    return f"""You are a Profiler agent in a tutoring system. Given the student's current cognitive state, decide what the tutor should do next.

Output ONLY a raw JSON object. No prose, no markdown code fences, no explanation before or after.

Field rules:
- next_action: MUST be exactly one of: "give_hint", "ask_clarifying_question", "present_new_content", "mark_mastered"
- reasoning: a short string explaining the choice
- target_misconception: a string from the student's misconceptions list, or null if not applicable

Example of a correctly formatted output:
{example}

Current student state:
{state.model_dump_json(indent=2)}

JSON output:"""

def profiler_agent(state: CognitiveState, model, tokenizer):
    """Returns (action, raw_output). action is None if validation failed."""
    prompt = build_profiler_prompt(state)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, max_length=None)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_json(raw)
    try:
        action = TutorAction.model_validate_json(cleaned)
        return action, raw
    except Exception:
        return None, raw
