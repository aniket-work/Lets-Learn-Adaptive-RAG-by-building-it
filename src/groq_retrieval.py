"""
Alternative embedding providers for when OpenAI is not available.
"""

from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import Config


class AlternativeEmbeddings:
    """Alternative embedding providers."""
    
    @staticmethod
    def get_embeddings():
        """Get the best available embedding model."""
        try:
            # Try OpenAI first if available
            if Config.OPENAI_API_KEY:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        except ImportError:
            pass
        
        # Fallback to HuggingFace embeddings (free, local)
        print("🔄 Using HuggingFace embeddings (local, free)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


class GroqDocumentRetriever:
    """Enhanced document retriever that works with Groq and alternative embeddings."""
    
    def __init__(self, vectorstore_path: str = Config.VECTOR_STORE_PATH):
        self.vectorstore_path = vectorstore_path
        self.embeddings = AlternativeEmbeddings.get_embeddings()
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create a vectorstore from documents."""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
    
    def load_vectorstore(self) -> None:
        """Load an existing vectorstore."""
        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        except Exception as e:
            raise FileNotFoundError(f"Could not load vectorstore from {self.vectorstore_path}: {e}")
    
    def save_vectorstore(self) -> None:
        """Save the vectorstore to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(self.vectorstore_path)
    
    def retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant documents for a question."""
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() or load_vectorstore() first.")
        
        return self.retriever.invoke(question)
