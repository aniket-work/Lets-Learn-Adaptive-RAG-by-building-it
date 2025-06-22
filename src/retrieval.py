"""
Document retrieval and web search functionality.
"""

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch

from .config import Config


class DocumentRetriever:
    """Handles document retrieval from vectorstore."""
    
    def __init__(self, vectorstore_path: str = Config.VECTOR_STORE_PATH):
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
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


class WebSearcher:
    """Handles web search functionality."""    
    def __init__(self, k: int = Config.WEB_SEARCH_K):
        self.web_search_tool = TavilySearch(k=k)
    
    def search(self, question: str) -> Document:
        """Perform web search and return results as a Document."""
        try:
            docs = self.web_search_tool.invoke({"query": question})
            
            # Handle different response formats
            if isinstance(docs, list):
                # If docs is a list of dicts
                if docs and isinstance(docs[0], dict):
                    web_results = "\n".join([
                        d.get("content", d.get("text", str(d))) 
                        for d in docs
                    ])
                else:
                    # If docs is a list of strings or other types
                    web_results = "\n".join([str(d) for d in docs])
            elif isinstance(docs, str):
                # If docs is already a string
                web_results = docs
            else:
                # Fallback - convert to string
                web_results = str(docs)
                
            return Document(page_content=web_results)
        except Exception as e:
            print(f"Web search error: {e}")
            return Document(page_content=f"Web search failed: {e}")
