"""
Data models and type definitions for the Adaptive RAG system.
"""

from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """State of the adaptive RAG graph."""
    question: str
    generation: str
    documents: List[str]
    retry_count: int  # Track retries to prevent infinite loops


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeDocuments(BaseModel):
    """Grade documents for relevance to a question."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
