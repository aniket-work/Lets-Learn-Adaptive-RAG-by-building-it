"""
Configuration settings for the Adaptive RAG system.
"""

import os
from typing import Optional


class Config:
    """Configuration class for Adaptive RAG."""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")  # For embeddings only
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Model configurations
    DEFAULT_MODEL: str = "llama3-8b-8192"  # Groq model
    ROUTER_MODEL: str = "llama3-70b-8192"  # Groq model for routing
    EMBEDDING_MODEL: str = "text-embedding-ada-002"  # OpenAI for embeddings
    
    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 0
    
    # Retrieval settings
    RETRIEVAL_K: int = 4  # Number of documents to retrieve
    WEB_SEARCH_K: int = 3  # Number of web search results
    
    # Temperature settings
    TEMPERATURE: float = 0.0
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "./data/vectorstore"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are set."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not cls.OPENAI_API_KEY:
            print("⚠️  OPENAI_API_KEY not set. Embeddings may not work properly.")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        return True
