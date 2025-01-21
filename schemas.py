"""
Defines Pydantic models for structured output parsing and schema definitions.
"""

from pydantic import BaseModel, Field
from langchain.output_parsers.pydantic import PydanticOutputParser


class DocumentAnalysis(BaseModel):
    """Structured analysis output model for document processing."""

    summary: str = Field(description="A concise summary of the document.")
    key_insights: list[str] = Field(description="Key insights or takeaways.")
    action_items: list[str] = Field(description="A list of actionable items.")
    open_questions: list[str] = Field(
        description="A list of open questions or areas for further investigation."
    )


# Initialize output parser for consistent use across the application
output_parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)
