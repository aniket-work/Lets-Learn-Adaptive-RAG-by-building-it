"""
Simple evaluation example using built-in evaluation system.

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

from src import AdaptiveRAG, DocumentProcessor, SimpleEvaluator


def main():
    """Main evaluation example."""
    print("🔍 Adaptive RAG Simple Evaluation Example")
    print("=" * 50)
    
    # Initialize components
    doc_processor = DocumentProcessor()
    
    # Create sample data if it doesn't exist
    sample_file = doc_processor.create_sample_data()
    documents = doc_processor.load_csv(sample_file)
    print(f"✅ Loaded {len(documents)} document chunks")
    
    # Initialize RAG system
    rag_system = AdaptiveRAG()
    rag_system.setup_vectorstore(documents)
    print("✅ RAG system initialized")
    
    # Initialize evaluator
    evaluator = SimpleEvaluator(rag_system)
    
    # Test questions - mix of vectorstore and web search queries
    test_questions = [
        "How does interlibrary loan work?",  # Should route to vectorstore
        "What is machine learning?",  # Should route to vectorstore
        "Tell me about Python programming language",  # Should route to vectorstore
        "What is the solar system?",  # Should route to vectorstore
        "What's the current weather in New York?",  # Should route to web search
        "Who won the latest Nobel Prize in Physics?",  # Should route to web search
        "What is RAG in AI?",  # Might route to web search
        "How do retrieval systems work?",  # Context-dependent routing
    ]
    
    print(f"\n📝 Evaluating {len(test_questions)} questions...")
    print("-" * 50)
    
    # Run evaluation
    results = evaluator.evaluate_batch(test_questions)
    
    if results:
        print(f"✅ Completed evaluation of {len(results)} questions")
        
        # Print detailed report
        evaluator.print_detailed_report()
        
        # Print individual results
        print("\n🔍 Show individual results? (y/n): ", end="")
        show_individual = input().lower().strip() == 'y'
        
        if show_individual:
            evaluator.print_individual_results()
        
        # Save results
        try:
            df = evaluator.save_results_to_csv()
            print(f"\n📊 Results summary:")
            print(f"   Questions answered: {len(df)}")
            print(f"   Average response time: {df['response_time'].mean():.2f}s")
            print(f"   Average answer length: {df['answer_length'].mean():.1f} words")
            print(f"   Vectorstore queries: {(df['route_taken'] == 'vectorstore').sum()}")
            print(f"   Web search queries: {(df['route_taken'] == 'web_search').sum()}")
            
        except Exception as e:
            print(f"⚠️  Could not save to CSV: {e}")
    
    else:
        print("❌ No evaluation results generated")
    
    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
