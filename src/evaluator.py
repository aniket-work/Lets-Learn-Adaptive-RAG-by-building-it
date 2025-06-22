"""
Simple built-in evaluation system for Adaptive RAG.
No external dependencies required.
"""

import time
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document

from .adaptive_rag import AdaptiveRAG


@dataclass
class EvaluationMetrics:
    """Simple evaluation metrics."""
    question: str
    answer: str
    context_used: str
    route_taken: str
    response_time: float
    document_count: int
    answer_length: int
    contains_citations: bool


class SimpleEvaluator:
    """Simple evaluation system for RAG responses."""
    
    def __init__(self, rag_system: AdaptiveRAG):
        self.rag_system = rag_system
        self.evaluation_results: List[EvaluationMetrics] = []
    
    def evaluate_response(self, question: str) -> EvaluationMetrics:
        """Evaluate a single response."""
        print(f"📝 Evaluating: {question}")
        
        start_time = time.time()
        
        # Track the routing decision
        route_taken = self.rag_system.question_router.route(question)
        
        # Get the full response
        inputs = {"question": question}
        result = self.rag_system.app.invoke(inputs)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Extract information
        answer = result.get("generation", "")
        documents = result.get("documents", [])
        
        # Calculate metrics
        context_used = ""
        document_count = 0
        
        if documents:
            if isinstance(documents, list):
                context_used = "\n".join([
                    doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                    for doc in documents
                ])
                document_count = len(documents)
            else:
                context_used = documents.page_content if hasattr(documents, 'page_content') else str(documents)
                document_count = 1
        
        # Simple heuristics
        answer_length = len(answer.split()) if answer else 0
        contains_citations = any(phrase in answer.lower() for phrase in [
            "according to", "based on", "the document", "the context", "as mentioned"
        ])
        
        metrics = EvaluationMetrics(
            question=question,
            answer=answer,
            context_used=context_used,
            route_taken=route_taken,
            response_time=response_time,
            document_count=document_count,
            answer_length=answer_length,
            contains_citations=contains_citations
        )
        
        self.evaluation_results.append(metrics)
        return metrics
    
    def evaluate_batch(self, questions: List[str]) -> List[EvaluationMetrics]:
        """Evaluate multiple questions."""
        results = []
        for question in questions:
            try:
                result = self.evaluate_response(question)
                results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating '{question}': {e}")
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a simple evaluation report."""
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # Calculate basic statistics
        total_questions = len(self.evaluation_results)
        avg_response_time = sum(r.response_time for r in self.evaluation_results) / total_questions
        avg_answer_length = sum(r.answer_length for r in self.evaluation_results) / total_questions
        avg_document_count = sum(r.document_count for r in self.evaluation_results) / total_questions
        
        # Route distribution
        vectorstore_count = sum(1 for r in self.evaluation_results if r.route_taken == "vectorstore")
        web_search_count = sum(1 for r in self.evaluation_results if r.route_taken == "web_search")
        
        # Citation analysis
        citations_count = sum(1 for r in self.evaluation_results if r.contains_citations)
        
        report = {
            "summary": {
                "total_questions": total_questions,
                "avg_response_time_seconds": round(avg_response_time, 2),
                "avg_answer_length_words": round(avg_answer_length, 1),
                "avg_documents_used": round(avg_document_count, 1),
            },
            "routing": {
                "vectorstore_queries": vectorstore_count,
                "web_search_queries": web_search_count,
                "vectorstore_percentage": round((vectorstore_count / total_questions) * 100, 1),
                "web_search_percentage": round((web_search_count / total_questions) * 100, 1),
            },
            "quality_indicators": {
                "responses_with_citations": citations_count,
                "citation_percentage": round((citations_count / total_questions) * 100, 1),
                "responses_with_context": sum(1 for r in self.evaluation_results if r.document_count > 0),
            },
            "performance": {
                "fastest_response": round(min(r.response_time for r in self.evaluation_results), 2),
                "slowest_response": round(max(r.response_time for r in self.evaluation_results), 2),
                "longest_answer": max(r.answer_length for r in self.evaluation_results),
                "shortest_answer": min(r.answer_length for r in self.evaluation_results),
            }
        }
        
        return report
    
    def print_detailed_report(self):
        """Print a detailed evaluation report."""
        report = self.generate_report()
        
        if "error" in report:
            print(f"❌ {report['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 ADAPTIVE RAG EVALUATION REPORT")
        print("="*60)
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"   Total Questions: {report['summary']['total_questions']}")
        print(f"   Avg Response Time: {report['summary']['avg_response_time_seconds']}s")
        print(f"   Avg Answer Length: {report['summary']['avg_answer_length_words']} words")
        print(f"   Avg Documents Used: {report['summary']['avg_documents_used']}")
        
        # Routing
        print(f"\n🔀 ROUTING ANALYSIS:")
        print(f"   Vectorstore Queries: {report['routing']['vectorstore_queries']} ({report['routing']['vectorstore_percentage']}%)")
        print(f"   Web Search Queries: {report['routing']['web_search_queries']} ({report['routing']['web_search_percentage']}%)")
        
        # Quality
        print(f"\n✅ QUALITY INDICATORS:")
        print(f"   Responses with Citations: {report['quality_indicators']['responses_with_citations']} ({report['quality_indicators']['citation_percentage']}%)")
        print(f"   Responses with Context: {report['quality_indicators']['responses_with_context']}")
        
        # Performance
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Fastest Response: {report['performance']['fastest_response']}s")
        print(f"   Slowest Response: {report['performance']['slowest_response']}s")
        print(f"   Longest Answer: {report['performance']['longest_answer']} words")
        print(f"   Shortest Answer: {report['performance']['shortest_answer']} words")
        
        print("\n" + "="*60)
    
    def print_individual_results(self):
        """Print individual evaluation results."""
        print("\n" + "="*60)
        print("📋 INDIVIDUAL RESULTS")
        print("="*60)
        
        for i, result in enumerate(self.evaluation_results, 1):
            print(f"\n🔍 Question {i}: {result.question}")
            print(f"   Route: {result.route_taken}")
            print(f"   Response Time: {result.response_time:.2f}s")
            print(f"   Documents Used: {result.document_count}")
            print(f"   Answer Length: {result.answer_length} words")
            print(f"   Has Citations: {'Yes' if result.contains_citations else 'No'}")
            print(f"   Answer: {result.answer[:100]}{'...' if len(result.answer) > 100 else ''}")
    
    def save_results_to_csv(self, filename: str = "./data/evaluation_results.csv"):
        """Save evaluation results to CSV."""
        import pandas as pd
        
        data = []
        for result in self.evaluation_results:
            data.append({
                "question": result.question,
                "answer": result.answer,
                "route_taken": result.route_taken,
                "response_time": result.response_time,
                "document_count": result.document_count,
                "answer_length": result.answer_length,
                "contains_citations": result.contains_citations,
                "context_preview": result.context_used[:200] + "..." if len(result.context_used) > 200 else result.context_used
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
        
        return df
