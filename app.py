import os
import requests
import streamlit as st
import logging
from typing import List, Optional

from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports
from schemas import output_parser, DocumentAnalysis
from utils.document_loader import load_document
from utils.analysis import analyze_documents
from constants import (
    DEFAULT_PROMPTS,
    TONE_OPTIONS,
    INSTRUCTION_OPTIONS,
    LENGTH_OPTIONS,
    SUPPORTED_FILE_TYPES,
)

# ----------------------------------------------------------------
# Configuration and Logging
# ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="DocMind AI: Local LLM for AI-Powered Document Analysis 🧠",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput > label, .stTextArea > label, .stSelectbox > label, .stFileUploader > label {
            color: #333;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_ollama_models(base_url: str) -> List[str]:
    """
    Fetch available Ollama models from the API.
    Cached to avoid repeatedly hitting the endpoint on each Streamlit rerun.
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models_info = response.json().get("models", [])
            return [model["name"] for model in models_info]
        else:
            logging.error(f"Failed to fetch models: {response.status_code}")
            return ["llama2"]  # Fallback
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        return ["llama2"]  # Fallback


def display_parsed_output(parsed_output: DocumentAnalysis, file_label: str):
    """Utility to display the parsed output from the LLM analysis."""
    st.subheader(f"Analysis of {file_label}")
    st.markdown(f"**Summary:** {parsed_output.summary}")
    if parsed_output.key_insights:
        st.markdown("**Key Insights:**")
        for insight in parsed_output.key_insights:
            st.markdown(f"- {insight}")
    if parsed_output.action_items:
        st.markdown("**Action Items:**")
        for action in parsed_output.action_items:
            st.markdown(f"- {action}")
    if parsed_output.open_questions:
        st.markdown("**Open Questions:**")
        for question in parsed_output.open_questions:
            st.markdown(f"- {question}")


def process_and_analyze_documents(
    documents,
    llm_model,
    prompt_str: Optional[str],
    tone: str,
    instruction: str,
    length: str,
    custom_instructions: Optional[str],
    file_label: str = "Uploaded Document",
):
    """
    Given loaded documents, run them through the text splitter and then the analysis pipeline.
    Finally parse and display results in Streamlit.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    output = analyze_documents(
        split_docs,
        prompt_str,
        llm_model,
        tone,
        instruction,
        length,
        custom_instructions,
    )

    # Attempt to parse structured output
    try:
        parsed_output = output_parser.parse(output)
        display_parsed_output(parsed_output, file_label)
    except Exception as e:
        st.error(f"Error parsing LLM output for {file_label}: {e}")
        st.text("Raw LLM Output:")
        st.text(output)


# ----------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------
st.title("DocMind AI: Local LLM for AI-Powered Document Analysis 🧠")
st.markdown("**Unlock the power of local AI to analyze your documents!**")

with st.expander("Instructions", expanded=True):
    st.markdown(
        """
        ## Getting Started
        1. **Install Ollama:** Follow instructions on the [Ollama website](https://ollama.com/).
        2. **Run Ollama:** Ensure Ollama is running.
        3. **Download Models:** Download LLM models (e.g., `ollama pull llama2`).
        """
    )

# Ollama configuration
ollama_base_url = st.text_input(
    "Ollama Base URL", value="http://localhost:11434", help="Ollama server address."
)
available_models = get_ollama_models(ollama_base_url)
model_name = st.selectbox(
    "Ollama Model",
    options=available_models,
    help="Select an installed Ollama model. If no models are shown, ensure Ollama is running and models are installed.",
)

uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, XLSX, MD, JSON, XML, RTF, CSV, MSG, PPTX, ODT, EPUB, ...)",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
    help="Upload files for analysis.",
)

# Initialize the LLM
llm = ChatOllama(base_url=ollama_base_url, model=model_name)

# Add "Custom Prompt" to default prompts
prompts = {**DEFAULT_PROMPTS, "Custom Prompt": None}
selected_prompt_name = st.selectbox(
    "Select Analysis Prompt",
    list(prompts.keys()),
    help="Choose a default prompt or use a custom one.",
)

custom_prompt = None
if selected_prompt_name == "Custom Prompt":
    custom_prompt = st.text_area(
        "Enter your custom prompt",
        help="Enter specific instructions for analyzing the document. Ensure it aligns with the output schema if you expect structured output.",
    )

# Tone, Instructions, Length
selected_tone = st.selectbox("Select Tone", list(TONE_OPTIONS.keys()))
instruction_options = {**INSTRUCTION_OPTIONS, "Custom Instructions": None}
selected_instruction = st.selectbox(
    "Select Instructions", list(instruction_options.keys())
)
custom_instructions = None
if selected_instruction == "Custom Instructions":
    custom_instructions = st.text_area(
        "Enter your custom instructions",
        help="Enter any additional instructions to fine-tune the analysis.",
    )
selected_length = st.selectbox(
    "Select Desired Length/Detail", list(LENGTH_OPTIONS.keys())
)

analysis_mode = st.radio(
    "Analysis Mode",
    ["Analyze each document separately", "Combine analysis for all documents"],
    help="Choose whether to analyze documents individually or together.",
)


# ----------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------
if uploaded_files:
    # Determine the prompt to use
    prompt_to_use = (
        custom_prompt
        if selected_prompt_name == "Custom Prompt"
        else prompts[selected_prompt_name]
    )

    if analysis_mode == "Analyze each document separately":
        for file in uploaded_files:
            with st.spinner(f"Analyzing {file.name}..."):
                loader = load_document(file)
                if loader:
                    docs = loader.load()
                    if docs:
                        process_and_analyze_documents(
                            documents=docs,
                            llm_model=llm,
                            prompt_str=prompt_to_use,
                            tone=selected_tone,
                            instruction=selected_instruction,
                            length=selected_length,
                            custom_instructions=custom_instructions,
                            file_label=file.name,
                        )
                    else:
                        st.warning(f"No content found in {file.name}")
                else:
                    st.error(
                        f"Unable to load {file.name}. See logs or console for details."
                    )

    else:  # Combine analysis for all documents
        all_docs = []
        for file in uploaded_files:
            with st.spinner(f"Loading {file.name}..."):
                loader = load_document(file)
                if loader:
                    all_docs.extend(loader.load())
                else:
                    st.error(
                        f"Unable to load {file.name}. See logs or console for details."
                    )

        if all_docs:
            with st.spinner("Analyzing all documents together..."):
                process_and_analyze_documents(
                    documents=all_docs,
                    llm_model=llm,
                    prompt_str=prompt_to_use,
                    tone=selected_tone,
                    instruction=selected_instruction,
                    length=selected_length,
                    custom_instructions=custom_instructions,
                    file_label="All Documents",
                )
        else:
            st.warning("No documents loaded for combined analysis.")
