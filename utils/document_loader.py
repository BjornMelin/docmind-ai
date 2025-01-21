"""
Document loading utilities for handling various file formats.

This module provides functionality to load documents of different formats
using appropriate LangChain document loaders.
"""

from langchain_community.document_loaders import TextLoader
from typing import Union
import streamlit as st
from constants import FILE_LOADER_MAP, SUPPORTED_FILE_TYPES, SUPPORTED_CODE_FILE_TYPES
from streamlit.runtime.uploaded_file_manager import UploadedFile
import logging

# Configure logging at the start of your application
logging.basicConfig(level=logging.ERROR)

def load_document(file: UploadedFile) -> Union[TextLoader, None]:
    """
    Load a document using the appropriate loader based on file extension.

    Args:
        file: Uploaded file object from Streamlit

    Returns:
        Loader instance for the document

    Raises:
        ValueError: If unsupported file type is provided
    """
    try:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Handle special case for code files
        if file_extension in SUPPORTED_CODE_FILE_TYPES:
            return TextLoader(file_path=file.name)

        # Get the appropriate loader class from the mapping
        loader_class = FILE_LOADER_MAP.get(file_extension)

        if not loader_class:
            raise ValueError(f"No loader configured for file type: {file_extension}")

        # JSONLoader requires special handling
        if file_extension == "json":
            return loader_class(file_path=file.name, jq_schema=".")

        # General case for other loaders
        return loader_class(file_path=file.name)

    except Exception as e:
        logging.error(f"Error initializing loader for {file.name}: {str(e)}")
        st.error(f"Error initializing loader for {file.name}: {str(e)}")
        return None
