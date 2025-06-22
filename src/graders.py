"""
Grading components for evaluating documents, answers, and hallucinations.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .models import GradeDocuments, GradeHallucinations, GradeAnswer
from .config import Config


class DocumentGrader:
    """Grades documents for relevance to a question."""
    
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        self.llm = ChatGroq(model=model_name, temperature=Config.TEMPERATURE)
        self.structured_llm = self.llm.with_structured_output(GradeDocuments)
        
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
        
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.grader = self.grade_prompt | self.structured_llm
    
    def grade(self, question: str, document: str) -> str:
        """Grade a document for relevance to a question."""
        result = self.grader.invoke({"question": question, "document": document})
        return result.binary_score


class HallucinationGrader:
    """Grades whether an answer is grounded in facts."""
    
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        self.llm = ChatGroq(model=model_name, temperature=Config.TEMPERATURE)
        self.structured_llm = self.llm.with_structured_output(GradeHallucinations)
        
        system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
        self.grader = self.grade_prompt | self.structured_llm
    
    def grade(self, documents: str, generation: str) -> str:
        """Grade whether a generation is grounded in documents."""
        result = self.grader.invoke({"documents": documents, "generation": generation})
        return result.binary_score


class AnswerGrader:
    """Grades whether an answer addresses the question."""
    
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        self.llm = ChatGroq(model=model_name, temperature=Config.TEMPERATURE)
        self.structured_llm = self.llm.with_structured_output(GradeAnswer)
        
        system_prompt = """You are a grader assessing whether an answer addresses / resolves a question
        
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        
        self.grader = self.grade_prompt | self.structured_llm
    
    def grade(self, question: str, generation: str) -> str:
        """Grade whether a generation addresses the question."""
        result = self.grader.invoke({"question": question, "generation": generation})
        return result.binary_score
