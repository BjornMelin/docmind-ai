from typing import Union, Optional
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    CSVLoader,
    UnstructuredEmailLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
    UnstructuredEPubLoader,
    PyPDFLoader,
)
from unstructured.file_utils.filetype import detect_filetype, FileType
import unstructured as ul
import streamlit as st
from constants import FILE_LOADER_MAP, SUPPORTED_FILE_TYPES, SUPPORTED_CODE_FILE_TYPES
from streamlit.runtime.uploaded_file_manager import UploadedFile
import logging
import tempfile
import os

# Configure logging at the start of your application
logging.basicConfig(level=logging.ERROR)


def get_unstructured_loader(
    file_path: str, loader_class: Optional[str] = None
) -> BaseLoader:
    """
    Instantiate and return an unstructured.io loader for the given file type.

    Args:
        file_path: Path to the document file
        loader_class: Optional loader class name for specific unstructured.io loaders

    Returns:
        Loader instance for the document

    Raises:
        ValueError: If an appropriate loader class is not found or if detection fails
    """
    # If a loader class name is specified, use it
    if loader_class == "PyPDFLoader":
        return PyPDFLoader(file_path)
    elif loader_class == "UnstructuredWordDocumentLoader":
        return ul.UnstructuredDocxLoader(file_path)
    elif loader_class == "TextLoader":
        return TextLoader(file_path)
    elif loader_class == "UnstructuredExcelLoader":
        return UnstructuredExcelLoader(file_path)
    elif loader_class == "UnstructuredMarkdownLoader":
        return UnstructuredMarkdownLoader(file_path)
    elif loader_class == "JSONLoader":
        return ul.UnstructuredJsonLoader(file_path)
    elif loader_class == "UnstructuredXMLLoader":
        return UnstructuredXMLLoader(file_path)
    elif loader_class == "UnstructuredRTFLoader":
        return UnstructuredRTFLoader(file_path)
    elif loader_class == "CSVLoader":
        return CSVLoader(file_path)
    elif loader_class == "UnstructuredEmailLoader":
        return UnstructuredEmailLoader(file_path)
    elif loader_class == "UnstructuredPowerPointLoader":
        return UnstructuredPowerPointLoader(file_path)
    elif loader_class == "UnstructuredODTLoader":
        return UnstructuredODTLoader(file_path)
    elif loader_class == "UnstructuredEPubLoader":
        return UnstructuredEPubLoader(file_path)
    else:
        raise ValueError(f"No matching loader found for: {loader_class}")


def load_document(file: UploadedFile) -> Union[BaseLoader, None]:
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

        # Create a temporary file to store the uploaded content
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.write(file.read())
        temp_file.close()

        # Handle special case for code files
        if file_extension in SUPPORTED_CODE_FILE_TYPES:
            return get_unstructured_loader(temp_file_path, "TextLoader")

        # Get the appropriate loader class from the mapping
        loader_class = FILE_LOADER_MAP.get(file_extension)

        if not loader_class:
            raise ValueError(f"No loader configured for file type: {file_extension}")

        # Return the loader instance
        return get_unstructured_loader(temp_file_path, loader_class)

    except Exception as e:
        logging.error(f"Error initializing loader for {file.name}: {str(e)}")
        st.error(f"Error initializing loader for {file.name}: {str(e)}")
        return None

    finally:
        # Clean up the temporary file after loading (if it was created)
        if "temp_file_path" in locals():
            os.unlink(temp_file_path)
