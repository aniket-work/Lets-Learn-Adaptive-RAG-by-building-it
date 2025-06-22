"""
Basic example of using the Adaptive RAG system.

Requirements:
- GROQ_API_KEY: Get from https://console.groq.com/
- TAVILY_API_KEY: Get from https://tavily.com/
- Optional: OPENAI_API_KEY for embeddings (HuggingFace used as fallback)
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path to find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from src import AdaptiveRAG, DocumentProcessor


def main():
    """Main example function."""
    print("🚀 Adaptive RAG Example")
    print("=" * 50)
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # Create sample data if it doesn't exist
    sample_file = doc_processor.create_sample_data()
    print(f"✅ Created sample data: {sample_file}")
    
    # Load and process documents
    documents = doc_processor.load_csv(sample_file)
    print(f"✅ Loaded {len(documents)} document chunks")
    
    # Initialize Adaptive RAG
    rag_system = AdaptiveRAG()
    
    # Set up vectorstore with documents
    rag_system.setup_vectorstore(documents)
    print("✅ Vectorstore created and saved")
    
    # Example queries
    questions = [
        "How does interlibrary loan work?",  # Should route to vectorstore
        "What is RAG?",  # Might route to web search if not in data
        "Tell me about machine learning",  # Should route to vectorstore
        "What's the weather like today?",  # Should route to web search
    ]
    
    for question in questions:
        print(f"\n🤔 Question: {question}")
        print("-" * 40)
        
        try:
            answer = rag_system.query_simple(question)
            print(f"💡 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


if __name__ == "__main__":
    main()
