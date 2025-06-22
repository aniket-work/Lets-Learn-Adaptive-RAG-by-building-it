"""
Main Adaptive RAG implementation using LangGraph and Groq.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START

from .models import GraphState
from .router import QuestionRouter
from .groq_retrieval import GroqDocumentRetriever
from .retrieval import WebSearcher
from .rag_chain import RAGChain
from .graders import DocumentGrader, HallucinationGrader, AnswerGrader
from .config import Config


class AdaptiveRAG:
    """Main Adaptive RAG system that combines routing, retrieval, and generation using Groq."""
    
    def __init__(self):
        # Initialize components
        self.question_router = QuestionRouter()
        self.retriever = GroqDocumentRetriever()  # Using Groq-compatible retriever
        self.web_searcher = WebSearcher()
        self.rag_chain = RAGChain()
        self.document_grader = DocumentGrader()
        self.hallucination_grader = HallucinationGrader()
        self.answer_grader = AnswerGrader()
        
        # Build the graph
        self.app = self._build_graph()
    
    def setup_vectorstore(self, documents: List[Document]) -> None:
        """Set up the vectorstore with documents."""
        self.retriever.create_vectorstore(documents)
        self.retriever.save_vectorstore()
    
    def load_vectorstore(self) -> None:
        """Load existing vectorstore."""
        self.retriever.load_vectorstore()
    
    def _build_graph(self) -> StateGraph:
        """Build the adaptive RAG graph."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("transform_query", self._transform_query)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            START,
            self._route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        
        workflow.add_edge("transform_query", "generate")
        
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },        )
        
        return workflow.compile()
    
    def _route_question(self, state: GraphState) -> str:
        """Route question to appropriate data source."""
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.route(question)
        
        if source == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    
    def _retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve documents from vectorstore."""
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.retrieve(question)
        return {"documents": documents, "question": question}
    
    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Grade documents for relevance."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for doc in documents:
            score = self.document_grader.grade(question, doc.page_content)
            if score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        
        return {"documents": filtered_docs, "question": question}
    
    def _generate(self, state: GraphState) -> Dict[str, Any]:
        """Generate answer using RAG."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        generation = self.rag_chain.generate(question, documents)
        return {"documents": documents, "question": question, "generation": generation}    
    def _web_search(self, state: GraphState) -> Dict[str, Any]:
        """Perform web search."""
        print("---WEB SEARCH---")
        question = state["question"]
        
        doc = self.web_searcher.search(question)
        return {"documents": [doc], "question": question}
    
    def _transform_query(self, state: GraphState) -> Dict[str, Any]:
        """Transform query for better results."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        
        # Increment retry counter
        retry_count = state.get("retry_count", 0) + 1
        print(f"---RETRY COUNT: {retry_count}---")
        
        # Simple query transformation - you can make this more sophisticated
        transformed_question = f"Please provide more details about: {question}"
        
        # For now, just return the same question
        # In a real implementation, you might use another LLM to transform the query
        return {
            "question": question, 
            "documents": state.get("documents", []),
            "retry_count": retry_count
        }
    
    def _decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate answer or transform query."""
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        
        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    def _grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """Grade the generation against documents and question."""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
          # Check hallucination
        docs_text = "\n".join([doc.page_content for doc in documents])
        score = self.hallucination_grader.grade(docs_text, generation)
        
        # Check retry count to prevent infinite loops
        retry_count = state.get("retry_count", 0)
        
        if score == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.grade(question, generation)
            if score == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                # Limit retries to prevent infinite loops
                if retry_count >= 3:
                    print("---MAX RETRIES REACHED, ACCEPTING ANSWER---")
                    return "useful"
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def query(self, question: str) -> str:
        """Query the adaptive RAG system."""
        Config.validate()  # Validate configuration
        
        inputs = {"question": question, "retry_count": 0}
        
        # Stream through the graph
        for output in self.app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
                print("\n---\n")
        
        # Return the final generation
        return value.get("generation", "No answer generated")
    
    def query_simple(self, question: str) -> str:
        """Simple query without streaming output."""
        Config.validate()
        
        inputs = {"question": question}
        result = self.app.invoke(inputs)
        return result.get("generation", "No answer generated")
