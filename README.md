# GenTut — Multi-Agent Intelligent Tutoring System

A stateful, multi-agent tutoring system built with LangGraph, Llama-3-8B-Instruct, and Gemini. Instead of fine-tuning, GenTut uses few-shot prompting and Pydantic-validated structured output to keep three cooperating agents in sync on a shared "Cognitive State" as a student works through a topic.

Built for the CCE IISc "LLMs — A Hands-on Approach" course (May–July 2026).

## Architecture

```
Student message
      │
      ▼
┌─────────────────┐     ┌───────────┐     ┌──────────────────┐
│ Skill Identifier │ ──▶ │ Profiler  │ ──▶ │ Content Creator   │ ──▶ Tutor response
│  (Llama-3-8B)    │     │(Llama-3-8B)│     │  (Gemini 2.5 Flash)│
└─────────────────┘     └───────────┘     └──────────────────┘
      ▲                                              │
      └──────────── conversation history ─────────────┘
```

- **Skill Identifier** — extracts/updates `CognitiveState` (topic, skill level, misconceptions, hint stage, mastery score) from the student's message and prior state.
- **Profiler** — decides the next `TutorAction` (give a hint, ask a clarifying question, present new content, or mark mastered).
- **Content Creator** — generates the actual `TutorContent` message shown to the student.

All three agents communicate exclusively through Pydantic-validated JSON, orchestrated as a LangGraph state machine (`GraphState`). Each `invoke()` call is one conversational turn; multi-turn sessions are driven externally (by the UI or eval harness) by feeding the returned state back in as input to the next turn.

## Repo structure

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic models: `CognitiveState`, `TutorAction`, `TutorContent`, `GraphState` |
| `agents.py` | Prompt builders, agent functions, and `make_tutoring_graph()` — the LangGraph wiring |
| `eval_utils.py` | `ResultsLogger` — reusable harness for logging trial results across all agents/metrics |
| `app.py` | Streamlit chat UI with a live Cognitive State sidebar |
| `step6_eval_results.csv` | Full trial-level eval log (schema fidelity, judge scores, learning gain) |
| `eval_summary.csv` | Final metrics summary table |
| `requirements.txt` | Frozen dependency list |
| `GenTut_Final_Report.docx` | Full project write-up |

## Results

| Metric | Target | Result |
|---|---|---|
| Schema Fidelity (all 3 agents) | >80% | **100%** |
| Pedagogical Quality (LLM-judge, 1–5) | >4 | **4.80** |
| Learning Gain Delta (avg) | positive | **+0.22** |
| End-to-End Latency (avg/turn) | <3s | ~14.7s *(see report, Limitations)* |

See `GenTut_Final_Report.docx` for full methodology, findings, and discussion.

## Setup

```bash
git clone <this-repo>
cd gentut
pip install -q -r requirements.txt
```

Requires a Hugging Face token (with access to `meta-llama/Meta-Llama-3-8B-Instruct` — accept the license on the model page first) and a Gemini API key. Set these as environment variables or in a `.env` file (gitignored):

```
HF_TOKEN=...
GEMINI_API_KEY=...
```

### Run in a notebook (Colab/Kaggle)
```python
from dotenv import load_dotenv
import os
load_dotenv()

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

# load model, tokenizer (see agents.py for the expected 4-bit BitsAndBytesConfig setup)
# load gemini via google.generativeai

from schemas import GraphState
from agents import make_tutoring_graph

tutoring_graph = make_tutoring_graph(model, tokenizer, gemini)
result = tutoring_graph.invoke(GraphState(student_message="I don't understand why my for loop never terminates."))
print(result["tutor_content"])
```

### Run the UI
```bash
streamlit run app.py --server.port 8501 --server.headless true
```
On a hosted notebook environment (Colab/Kaggle) without a public URL, tunnel with `pyngrok`:
```python
from pyngrok import ngrok
ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
print(ngrok.connect(8501))
```

## Known limitations

- **Latency** (~14.7s/turn) exceeds the <3s target — three sequential LLM calls per turn on free-tier T4 hardware. See report Section 7 for discussion.
- **Confidence-signal under-weighting** — the system sometimes continues offering hints after a student has explicitly signaled understanding. Partially mitigated; documented in report Section 6.4.

## Compute notes

Developed on free-tier T4 GPUs, initially via Google Colab, migrated to Kaggle mid-project after hitting Colab usage limits. Both platforms are ephemeral (no persistent local disk), so all code is versioned in this repo rather than relying on notebook state, and secrets are loaded via each platform's secret manager into an untracked `.env` file at session start.
