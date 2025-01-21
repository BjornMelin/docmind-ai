import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schemas import output_parser
from constants import (
    DEFAULT_PROMPTS,
    TONE_OPTIONS,
    INSTRUCTION_OPTIONS,
    LENGTH_OPTIONS,
    SUPPORTED_FILE_TYPES,
)
from utils.document_loader import load_document
from utils.analysis import analyze_documents
import logging

# Configure logging at the start of the application
logging.basicConfig(level=logging.INFO)

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="DocMind AI Local LLM",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput > label {
            color: #333;
        }
        .stTextArea > label {
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
        .stSelectbox > label {
            color: #333;
        }
        .stFileUploader > label {
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DocMind AI Local LLM 🧠")
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

ollama_base_url = st.text_input(
    "Ollama Base URL", value="http://localhost:11434", help="Ollama server address."
)
model_name = st.text_input(
    "Ollama Model Name", value="llama3.3", help="Name of the Ollama model."
)

uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, XLSX, MD, JSON, XML, RTF, CSV, MSG, PPTX, ODT, EPUB, ...)",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
    help="Upload files for analysis.",
)

llm = ChatOllama(model=model_name, base_url=ollama_base_url)

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
        help="Enter your specific instructions for analyzing the document. Ensure it aligns with the output schema if you expect structured output.",
    )

# --- Tone Selection ---
selected_tone = st.selectbox(
    "Select Tone",
    list(TONE_OPTIONS.keys()),
    help="Choose the desired tone for the analysis.",
)

# --- Persona/Instruction Selection ---
instruction_options = {**INSTRUCTION_OPTIONS, "Custom Instructions": None}
selected_instruction = st.selectbox(
    "Select Instructions",
    list(instruction_options.keys()),
    help="Choose specific instructions for the analysis.",
)

custom_instructions = None
if selected_instruction == "Custom Instructions":
    custom_instructions = st.text_area(
        "Enter your custom instructions",
        help="Enter any additional instructions to fine-tune the analysis.",
    )

# --- Length/Detail Selection ---
selected_length = st.selectbox(
    "Select Desired Length/Detail",
    list(LENGTH_OPTIONS.keys()),
    help="Choose the desired length or level of detail for the analysis.",
)

# --- Analysis Mode ---
analysis_mode = st.radio(
    "Analysis Mode",
    ["Analyze each document separately", "Combine analysis for all documents"],
    help="Choose whether to analyze documents individually or together.",
)

# --- Text Splitting Configuration ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- Main Application ---
if uploaded_files:
    if analysis_mode == "Analyze each document separately":
        for file in uploaded_files:
            with st.spinner(f"Analyzing {file.name}"):
                # Save uploaded file temporarily
                with open(file.name, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    loader = load_document(file)
                    documents = loader.load()
                    if documents:
                        prompt_to_use = (
                            custom_prompt
                            if selected_prompt_name == "Custom Prompt"
                            else prompts[selected_prompt_name]
                        )
                        split_docs = text_splitter.split_documents(documents)
                        output = analyze_documents(
                            split_docs,
                            prompt_to_use,
                            llm,
                            selected_tone,
                            selected_instruction,
                            selected_length,
                            custom_instructions,
                        )
                        try:
                            parsed_output = output_parser.parse(output)
                            st.subheader(f"Analysis of {file.name}")
                            st.markdown(f"**Summary:** {parsed_output.summary}")
                            if parsed_output.key_insights:
                                st.markdown("**Key Insights:**")
                                for item in parsed_output.key_insights:
                                    st.markdown(f"- {item}")
                            if parsed_output.action_items:
                                st.markdown("**Action Items:**")
                                for item in parsed_output.action_items:
                                    st.markdown(f"- {item}")
                            if parsed_output.open_questions:
                                st.markdown("**Open Questions:**")
                                for item in parsed_output.open_questions:
                                    st.markdown(f"- {item}")
                        except Exception as e:
                            st.error(f"Error parsing LLM output: {e}")
                            st.text("Raw LLM Output:")
                            st.text(output)
                    else:
                        st.warning(f"No content found in {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                # Clean up temporary file
                os.remove(file.name)
    else:  # Combine analysis for all documents
        all_docs = []
        for file in uploaded_files:
            with st.spinner(f"Loading {file.name}"):
                # Save uploaded file temporarily
                with open(file.name, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    loader = load_document(file)
                    all_docs.extend(loader.load())
                except Exception as e:
                    logging.error(f"Error loading {file.name}: {e}")
                    st.error(f"Error loading {file.name}: {e}")
                # Clean up temporary file
                os.remove(file.name)
        if all_docs:
            with st.spinner("Analyzing all documents together..."):
                prompt_to_use = (
                    custom_prompt
                    if selected_prompt_name == "Custom Prompt"
                    else prompts[selected_prompt_name]
                )
                split_docs = text_splitter.split_documents(all_docs)
                output = analyze_documents(
                    split_docs,
                    prompt_to_use,
                    llm,
                    selected_tone,
                    selected_instruction,
                    selected_length,
                    custom_instructions,
                )
                try:
                    parsed_output = output_parser.parse(output)
                    st.subheader("Combined Analysis of All Documents")
                    st.markdown(f"**Summary:** {parsed_output.summary}")
                    if parsed_output.key_insights:
                        st.markdown("**Key Insights:**")
                        for item in parsed_output.key_insights:
                            st.markdown(f"- {item}")
                    if parsed_output.action_items:
                        st.markdown("**Action Items:**")
                        for item in parsed_output.action_items:
                            st.markdown(f"- {item}")
                    if parsed_output.open_questions:
                        st.markdown("**Open Questions:**")
                        for item in parsed_output.open_questions:
                            st.markdown(f"- {item}")
                except Exception as e:
                    st.error(f"Error parsing LLM output: {e}")
                    st.text("Raw LLM Output:")
                    st.text(output)
        else:
            st.warning("No documents loaded for combined analysis.")
