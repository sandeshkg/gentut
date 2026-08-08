import re
from schemas import CognitiveState, TutorAction, TutorContent, GraphState
from langgraph.graph import StateGraph, END

# ---------- shared helpers ----------

def extract_json(raw: str) -> str:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return match.group(0) if match else raw

PLACEHOLDER_TOPICS = {"same_topic_as_previous", "your_topic_name", "same as previous", "unchanged", "same"}

# ---------- Skill Identifier ----------

def build_skill_identifier_prompt(student_message: str, prior_state=None, recent_history=None) -> str:
    example = CognitiveState(
        student_id="EXAMPLE_ID", current_topic="EXAMPLE_TOPIC_DO_NOT_COPY", skill_level="proficient",
        misconceptions=["EXAMPLE_MISCONCEPTION_DO_NOT_COPY"], hint_stage=2, mastery_score=0.9
    ).model_dump_json(indent=2)

    context_block = ""
    if prior_state is not None:
        context_block = f"""
Previous cognitive state (this reflects the ACTUAL ongoing session — update it based on the new message; don't reset it unless the topic clearly changed):
{prior_state.model_dump_json(indent=2)}
"""
    if recent_history:
        context_block += f"\nRecent conversation:\n" + "\n".join(recent_history[-4:])

    return f"""You are a Skill Identifier agent. Extract/update the student's cognitive state given the conversation so far.

Output ONLY a raw JSON object. No prose, no markdown code fences.

CRITICAL: The example below shows FORMAT ONLY. Its field values are placeholders — do NOT copy them into your answer.

Field rules:
- student_id: keep the same as previous state if given, else any string
- current_topic: always output the actual topic name as a real string (e.g. "for loops", "recursion"). NEVER output placeholder phrases like "same as previous" or "your_topic_name" — if the topic hasn't changed, repeat the exact same real topic string as before.
- skill_level: one of "novice", "developing", "proficient". If the student expresses understanding/confidence (e.g. "got it", "that makes sense"), increase skill_level.
- misconceptions: carry forward unresolved ones from prior state, remove resolved ones, add new ones actually implied by this message.
- hint_stage: increment by 1 from the previous state's hint_stage if another hint was needed on the same misconception; reset to 0 only on a genuinely new topic.
- mastery_score: float 0.0-1.0, generally increasing across turns on the same topic if the student shows understanding.

Example showing FORMAT ONLY (do not reuse these values):
{example}
{context_block}
Latest student message: "{student_message}"

JSON output:"""

def skill_identifier_agent(student_message: str, model, tokenizer, prior_state=None, recent_history=None):
    """Returns (state, raw_output). state is None if validation failed."""
    prompt = build_skill_identifier_prompt(student_message, prior_state, recent_history)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, max_length=None)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_json(raw)
    try:
        state = CognitiveState.model_validate_json(cleaned)
        # programmatic guard: fix placeholder topic labels rather than relying purely on prompt wording
        if prior_state is not None and state.current_topic.strip().lower() in PLACEHOLDER_TOPICS:
            state.current_topic = prior_state.current_topic
        return state, raw
    except Exception:
        return None, raw

# ---------- Profiler ----------

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

# ---------- Content Creator ----------

def build_content_creator_prompt(state: CognitiveState, action: TutorAction) -> str:
    example = TutorContent(
        content_type="hint",
        message="Take a look at your loop's condition — does it ever change inside the loop body? If not, that's why it never ends.",
        difficulty_note="Keep hint conceptual, avoid giving the full answer directly."
    ).model_dump_json(indent=2)

    return f"""You are a Content Creator agent in a tutoring system. Generate the actual tutoring content to show the student, based on their cognitive state and the chosen tutor action.

Output ONLY a raw JSON object. No prose, no markdown code fences, no explanation before or after.

Field rules:
- content_type: MUST match the tutor action's intent, one of: "hint", "clarifying_question", "new_content", "mastery_message"
- message: the actual text shown to the student. Be encouraging, concise, and pedagogically sound.
- difficulty_note: a short internal note on pacing, or null

Example of a correctly formatted output:
{example}

Student's current state:
{state.model_dump_json(indent=2)}

Chosen tutor action:
{action.model_dump_json(indent=2)}

JSON output:"""

def content_creator_agent(state: CognitiveState, action: TutorAction, gemini_model):
    """Returns (content, raw_output). content is None if validation failed."""
    prompt = build_content_creator_prompt(state, action)
    response = gemini_model.generate_content(prompt)
    raw = response.text
    cleaned = extract_json(raw)
    try:
        content = TutorContent.model_validate_json(cleaned)
        return content, raw
    except Exception:
        return None, raw

# ---------- LangGraph wiring (single source of truth) ----------

def make_tutoring_graph(model, tokenizer, gemini_model):
    def node_skill_identifier(state: GraphState) -> dict:
        cog_state, raw = skill_identifier_agent(
            state.student_message, model, tokenizer,
            prior_state=state.cognitive_state,
            recent_history=state.conversation_history
        )
        if cog_state is None:
            cog_state = state.cognitive_state or CognitiveState(student_id="s001", current_topic="unclear")
        return {"cognitive_state": cog_state}

    def node_profiler(state: GraphState) -> dict:
        action, raw = profiler_agent(state.cognitive_state, model, tokenizer)
        if action is None:
            action = TutorAction(next_action="ask_clarifying_question", reasoning="fallback: profiler validation failed")
        return {"tutor_action": action}

    def node_content_creator(state: GraphState) -> dict:
        content, raw = content_creator_agent(state.cognitive_state, state.tutor_action, gemini_model)
        if content is None:
            content = TutorContent(content_type="clarifying_question", message="Could you tell me more about what you're finding tricky?")
        history = state.conversation_history + [state.student_message, content.message]
        return {"tutor_content": content, "conversation_history": history, "turn_count": state.turn_count + 1}

    graph = StateGraph(GraphState)
    graph.add_node("skill_identifier", node_skill_identifier)
    graph.add_node("profiler", node_profiler)
    graph.add_node("content_creator", node_content_creator)

    graph.set_entry_point("skill_identifier")
    graph.add_edge("skill_identifier", "profiler")
    graph.add_edge("profiler", "content_creator")
    graph.add_edge("content_creator", END)  # each invoke() = one turn; multi-turn loop is driven externally

    return graph.compile()
