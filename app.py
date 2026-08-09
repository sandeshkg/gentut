import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import google.generativeai as genai

from schemas import GraphState
from agents import make_tutoring_graph

st.set_page_config(page_title="GenTut", layout="wide")

@st.cache_resource
def load_models():
    login(token=os.getenv("HF_TOKEN"))
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    gemini = genai.GenerativeModel("gemini-2.5-flash")

    graph = make_tutoring_graph(model, tokenizer, gemini)
    return graph

st.title("🎓 GenTut — AI Tutor")

with st.spinner("Loading models... this takes a minute on first load"):
    tutoring_graph = load_models()

if "graph_state" not in st.session_state:
    st.session_state.graph_state = GraphState()
if "display_history" not in st.session_state:
    st.session_state.display_history = []

with st.sidebar:
    st.subheader("📊 Cognitive State")
    cog = st.session_state.graph_state.cognitive_state
    if cog:
        st.json(cog.model_dump())
    else:
        st.write("No data yet — ask a question to begin.")

for role, msg in st.session_state.display_history:
    with st.chat_message(role):
        st.write(msg)

user_input = st.chat_input("Ask a question...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.display_history.append(("user", user_input))

    st.session_state.graph_state.student_message = user_input
    with st.spinner("Thinking..."):
        result = tutoring_graph.invoke(st.session_state.graph_state)
    st.session_state.graph_state = GraphState(**result)

    tutor_msg = st.session_state.graph_state.tutor_content.message
    with st.chat_message("assistant"):
        st.write(tutor_msg)
    st.session_state.display_history.append(("assistant", tutor_msg))

    st.rerun()
