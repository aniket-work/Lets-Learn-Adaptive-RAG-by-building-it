"""
Adaptive RAG package initialization.
"""

from .adaptive_rag import AdaptiveRAG
from .document_processor import DocumentProcessor
from .evaluator import SimpleEvaluator
from .config import Config

__version__ = "1.0.0"
__all__ = ["AdaptiveRAG", "DocumentProcessor", "SimpleEvaluator", "Config"]
