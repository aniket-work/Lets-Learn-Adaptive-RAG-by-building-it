"""
RAG chain for generating responses based on retrieved documents.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import Config


class RAGChain:
    """Handles RAG generation using retrieved documents."""
    
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        self.llm = ChatGroq(model=model_name, temperature=Config.TEMPERATURE)
        
        template = """
You are a helpful assistant that answers questions based on the following context.
Use the provided context to answer the question.

Context: {context}
Question: {question}
Answer:
"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self, question: str, documents: List[Document]) -> str:
        """Generate an answer based on the question and documents."""
        context = self.format_docs(documents)
        return self.chain.invoke({"context": context, "question": question})
