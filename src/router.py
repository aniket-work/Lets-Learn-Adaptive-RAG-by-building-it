"""
Question router to determine the best data source for answering a query.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .models import RouteQuery
from .config import Config


class QuestionRouter:
    """Routes questions to the most appropriate data source."""
    
    def __init__(self, model_name: str = Config.ROUTER_MODEL):
        self.llm = ChatGroq(model=model_name, temperature=Config.TEMPERATURE)
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        
        # System prompt - customize based on your vectorstore content
        system_prompt = """You are an expert at routing a user question to either a vectorstore or web search.

The vectorstore contains information on the following topics:
- Finance and real estate
- Library and research topics
- Biology and microbiology
- Literature and writing
- Movies and entertainment
- Animals and nature
- History and geography
- Astronomy

If the question is related to these topics, route it to the vectorstore. Otherwise, use web search."""
        
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        self.router = self.route_prompt | self.structured_llm
    
    def route(self, question: str) -> str:
        """Route a question to the appropriate data source."""
        result = self.router.invoke({"question": question})
        return result.datasource
    
    def update_topics(self, topics: list[str]) -> None:
        """Update the topics in the vectorstore for better routing."""
        topics_text = "\n".join([f"- {topic}" for topic in topics])
        
        system_prompt = f"""You are an expert at routing a user question to either a vectorstore or web search.

The vectorstore contains information on the following topics:
{topics_text}

If the question is related to these topics, route it to the vectorstore. Otherwise, use web search."""
        
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        self.router = self.route_prompt | self.structured_llm
