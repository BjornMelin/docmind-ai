"""
Application constants and configuration values.

This module contains all constant values, default configurations, 
and lookup dictionaries used throughout the application.
"""

from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
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

# --- Prompt Configuration ---
DEFAULT_PROMPTS = {
    "Comprehensive Document Analysis": (
        "Analyze this document to provide a concise summary, identify key insights, "
        "list actionable items, and highlight any open questions. {tone_instructions} "
        "{custom_instructions} {length_instructions} Format your response as a JSON object "
        "with 'summary', 'key_insights', 'action_items', and 'open_questions' keys as "
        "instructed by the following schema:\n{format_instructions}"
    ),
    "Extract Key Insights and Action Items": (
        "Identify the key insights and actionable items from this document. "
        "{tone_instructions} {custom_instructions} {length_instructions} "
        "Format your response as a JSON object with 'key_insights' and 'action_items' keys "
        "as instructed by the following schema:\n{format_instructions}"
    ),
    "Summarize and Identify Open Questions": (
        "Provide a summary of the document and list any open questions or areas requiring "
        "further investigation. {tone_instructions} {custom_instructions} {length_instructions} "
        "Format your response as a JSON object with 'summary' and 'open_questions' keys as "
        "instructed by the following schema:\n{format_instructions}"
    )
}

# --- Tone Options ---
TONE_OPTIONS = {
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

# --- Instruction Options ---
INSTRUCTION_OPTIONS = {
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
}

# --- Length/Detail Options ---
LENGTH_OPTIONS = {
    "Concise": "Provide a brief and to-the-point response.",
    "Detailed": "Provide a thorough and comprehensive response, with detailed explanations and examples.",
    "Comprehensive": "Provide a comprehensive response including any necessary details.",
    "Bullet Points": "Provide your response in bullet point format.",
}

# --- File Handling Configuration ---
FILE_LOADER_MAP = {
    "pdf": PyPDFLoader,
    "docx": UnstructuredWordDocumentLoader,
    "txt": TextLoader,
    "xlsx": UnstructuredExcelLoader,
    "md": UnstructuredMarkdownLoader,
    "json": JSONLoader,
    "xml": UnstructuredXMLLoader,
    "rtf": UnstructuredRTFLoader,
    "csv": CSVLoader,
    "msg": UnstructuredEmailLoader,
    "pptx": UnstructuredPowerPointLoader,
    "odt": UnstructuredODTLoader,
    "epub": UnstructuredEPubLoader,
}

SUPPORTED_FILE_TYPES = [
    "pdf", "docx", "txt", "xlsx", "md", "json", "xml", "rtf", "csv", "msg",
    "pptx", "odt", "epub", "py", "js", "java", "ts", "tsx", "c", "cpp", "h"
]

SUPPORTED_CODE_FILE_TYPES = [
    "py", "js", "java", "ts", "tsx", "c", "cpp", "h"
]
