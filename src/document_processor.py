"""
Document processing utilities for loading and splitting documents.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config


class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_csv(self, file_path: str) -> List[Document]:
        """Load documents from a CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        loader = CSVLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load documents from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
        """Load documents from a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        loader = DirectoryLoader(directory_path, glob=glob_pattern)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def create_sample_data(self, output_path: str = "./data/sample_context.csv") -> str:
        """Create sample data for testing."""
        sample_data = """content
"Interlibrary loan (abbreviated ILL) is a service that enables patrons of one library to borrow physical materials and receive electronic documents that are held by another library. The service expands library patrons' access to resources beyond their local library's holdings."
"After receiving a request from their patron, the borrowing library identifies potential lending libraries with the desired item. The lending library then delivers the item physically or electronically, and the borrowing library receives the item, delivers it to their patron, and if necessary, arranges for its return."
"Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
"Retrieval-Augmented Generation (RAG) is a technique that combines retrieval systems with generative language models to provide more accurate and contextually relevant responses by incorporating external knowledge sources."
"Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development."
"The solar system consists of the Sun and the objects that orbit it, including eight planets, their moons, and smaller bodies like asteroids and comets. Earth is the third planet from the Sun and the only known planet to harbor life."
"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        return output_path
