import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2Loader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    JSONLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    CSVLoader,
    UnstructuredEmailLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
    UnstructuredEPubLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import os


# --- Pydantic Model for Structured Output ---
class DocumentAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the document.")
    key_insights: list[str] = Field(description="Key insights or takeaways.")
    action_items: list[str] = Field(description="A list of actionable items.")
    open_questions: list[str] = Field(
        description="A list of open questions or areas for further investigation."
    )


output_parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)

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
        /* ... (same CSS as before) ... */
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
    "Ollama Model Name", value="llama2", help="Name of the Ollama model."
)

uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, XLSX, MD, JSON, XML, RTF, CSV, MSG, PPTX, ODT, EPUB, ...)",
    type=[
        "pdf",
        "docx",
        "txt",
        "xlsx",
        "md",
        "json",
        "xml",
        "rtf",
        "csv",
        "msg",
        "pptx",
        "odt",
        "epub",
        "py",
        "js",
        "java",
        "ts",
        "tsx",
        "c",
        "cpp",
        "h",
    ],
    accept_multiple_files=True,
    help="Upload files for analysis.",
)

llm = ChatOllama(base_url=ollama_base_url, model=model_name)

# --- Default Prompts ---
default_prompts = {
    "Comprehensive Document Analysis": """Analyze this document to provide a concise summary, identify key insights, list actionable items, and highlight any open questions. {tone_instructions} {custom_instructions} {length_instructions} Format your response as a JSON object with "summary", "key_insights", "action_items", and "open_questions" keys as instructed by the following schema:
    {format_instructions}""",
    "Extract Key Insights and Action Items": """Identify the key insights and actionable items from this document. {tone_instructions} {custom_instructions} {length_instructions} Format your response as a JSON object with "key_insights" and "action_items" keys as instructed by the following schema:
    {format_instructions}""",
    "Summarize and Identify Open Questions": """Provide a summary of the document and list any open questions or areas requiring further investigation. {tone_instructions} {custom_instructions} {length_instructions} Format your response as a JSON object with "summary" and "open_questions" keys as instructed by the following schema:
    {format_instructions}""",
    "Custom Prompt": None,
}

selected_prompt_name = st.selectbox(
    "Select Analysis Prompt",
    list(default_prompts.keys()),
    help="Choose a default prompt or use a custom one.",
)

custom_prompt = None
if selected_prompt_name == "Custom Prompt":
    custom_prompt = st.text_area(
        "Enter your custom prompt",
        help="Enter your specific instructions for analyzing the document. Ensure it aligns with the output schema if you expect structured output.",
    )

# --- Tone Selection ---
tone_options = {
    "Professional": "Maintain a professional and objective tone.",
    "Academic": "Use an academic and formal tone, appropriate for scholarly research.",
    "Informal": "Adopt an informal and conversational tone.",
    "Creative": "Be creative and imaginative in your response.",
    "Neutral": "Maintain a neutral tone, avoiding any strong opinions or biases.",
    "Direct": "Be direct and to-the-point, avoiding unnecessary elaboration.",
    "Empathetic": "Respond with empathy and understanding, suitable for sensitive topics.",
    "Humorous": "Incorporate humor and wit where appropriate.",
    "Authoritative": "Sound confident and authoritative in your response.",
    "Inquisitive": "Adopt an inquisitive tone, focusing on exploration and questioning.",
}

selected_tone = st.selectbox(
    "Select Tone",
    list(tone_options.keys()),
    help="Choose the desired tone for the analysis.",
)

# --- Persona/Instruction Selection ---
instruction_options = {
    "General Assistant": "Act as a general-purpose assistant.",
    "Researcher": "Focus on in-depth research and analysis, providing detailed explanations and citations where appropriate.",
    "Software Engineer": "Tailor your responses to a software engineer, focusing on technical details, code quality, and system design.",
    "Product Manager": "Act as a product manager, considering market needs, product strategy, and user experience.",
    "Data Scientist": "Respond as a data scientist, emphasizing data analysis, statistical significance, and model accuracy.",
    "Business Analyst": "Provide analysis from a business perspective, considering profitability, market trends, and strategic implications.",
    "Technical Writer": "Focus on clear and concise documentation, suitable for technical manuals and user guides.",
    "Marketing Specialist": "Tailor your responses to marketing concerns, such as branding, customer engagement, and market positioning.",
    "HR Manager": "Respond with a focus on human resources considerations, such as employee well-being, recruitment, and training.",
    "Legal Advisor": "Provide information with a legal perspective, focusing on compliance, regulations, and potential legal issues.",
    "Custom Instructions": None,
}

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
length_options = {
    "Concise": "Provide a brief and to-the-point response.",
    "Detailed": "Provide a thorough and comprehensive response, with detailed explanations and examples.",
    "Comprehensive": "Provide a comprehensive response including any necessary details.",
    "Bullet Points": "Provide your response in bullet point format.",
}

selected_length = st.selectbox(
    "Select Desired Length/Detail",
    list(length_options.keys()),
    help="Choose the desired length or level of detail for the analysis.",
)

# --- Analysis Mode ---
analysis_mode = st.radio(
    "Analysis Mode",
    ["Analyze each document separately", "Combine analysis for all documents"],
    help="Choose whether to analyze documents individually or together.",
)


# --- Functions ---
def load_document(file):
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return PyPDFLoader(file_path=file.name)
    elif file_extension == "docx":
        return Docx2Loader(file_path=file.name)
    elif file_extension == "txt":
        return TextLoader(file_path=file.name)
    elif file_extension == "xlsx":
        return UnstructuredExcelLoader(file_path=file.name)
    elif file_extension == "md":
        return UnstructuredMarkdownLoader(file_path=file.name)
    elif file_extension == "json":
        return JSONLoader(file_path=file.name, jq_schema=".")
    elif file_extension == "xml":
        return UnstructuredXMLLoader(file_path=file.name)
    elif file_extension == "rtf":
        return UnstructuredRTFLoader(file_path=file.name)
    elif file_extension == "csv":
        return CSVLoader(file_path=file.name)
    elif file_extension == "msg":
        return UnstructuredEmailLoader(file_path=file.name)
    elif file_extension == "pptx":
        return UnstructuredPowerPointLoader(file_path=file.name)
    elif file_extension == "odt":
        return UnstructuredODTLoader(file_path=file.name)
    elif file_extension == "epub":
        return UnstructuredEPubLoader(file_path=file.name)
    elif file_extension in ["py", "js", "java", "ts", "tsx", "c", "cpp", "h"]:
        return TextLoader(file_path=file.name)
    raise ValueError(f"Unsupported file type: {file_extension}")


def analyze_documents(docs, prompt_template):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Construct the final prompt using the selected tone and instructions
    tone_instructions = tone_options[selected_tone]
    length_instructions = length_options[selected_length]
    instruction = instruction_options[selected_instruction]

    prompt = prompt_template.format(
        format_instructions=output_parser.get_format_instructions(),
        tone_instructions=f" {tone_instructions}",
        custom_instructions=f" {instruction} {custom_instructions if custom_instructions else ''}",
        length_instructions=f" {length_instructions}",
    )

    final_prompt = PromptTemplate(
        template=prompt,
        input_variables=["text"],
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=final_prompt,
        combine_prompt=final_prompt,
    )
    return chain.run(split_docs)


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
                            else default_prompts[selected_prompt_name]
                        )
                        output = analyze_documents(documents, prompt_to_use)
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
                    st.error(f"Error loading {file.name}: {e}")
                # Clean up temporary file
                os.remove(file.name)
        if all_docs:
            with st.spinner("Analyzing all documents together..."):
                prompt_to_use = (
                    custom_prompt
                    if selected_prompt_name == "Custom Prompt"
                    else default_prompts[selected_prompt_name]
                )
                output = analyze_documents(all_docs, prompt_to_use)
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
