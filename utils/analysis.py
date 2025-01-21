"""
Document analysis chain configuration and execution.

This module handles the setup and execution of the LangChain analysis pipeline.
"""

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from typing import Optional
import streamlit as st
from langchain_core.language_models.base import BaseLanguageModel

from schemas import output_parser
from constants import DEFAULT_PROMPTS, TONE_OPTIONS, LENGTH_OPTIONS, INSTRUCTION_OPTIONS


def analyze_documents(
    docs: list,
    prompt_template: str,
    llm: BaseLanguageModel,
    selected_tone: str,
    selected_instruction: str,
    selected_length: str,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Execute document analysis pipeline using configured prompts and LLM.

    Args:
        docs: List of documents to analyze
        prompt_template: Base prompt template string
        llm: Configured language model instance
        selected_tone: Key for tone configuration
        selected_instruction: Key for instruction configuration
        selected_length: Key for length configuration
        custom_instructions: Optional custom instructions from user

    Returns:
        Analysis result string from LLM
    """
    try:
        # Construct final prompt with configured options
        # Fetch tone, length, and instruction options
        tone_instructions = TONE_OPTIONS.get(selected_tone, "")
        length_instructions = LENGTH_OPTIONS.get(selected_length, "")
        instruction = INSTRUCTION_OPTIONS.get(selected_instruction, "")

        final_prompt = PromptTemplate(
            template=prompt_template.format(
                format_instructions=output_parser.get_format_instructions(),
                tone_instructions=tone_instructions,
                custom_instructions=f"{instruction} {custom_instructions or ''}",
                length_instructions=length_instructions,
            ),
            input_variables=["text"],
        )

        # Configure and execute analysis chain
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=final_prompt,
            combine_prompt=final_prompt,
        )
        return chain.run(docs)

    except Exception as e:
        st.error(f"Analysis pipeline error: {str(e)}")
        raise
