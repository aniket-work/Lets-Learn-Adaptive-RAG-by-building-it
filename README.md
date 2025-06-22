# 🧠 Adaptive RAG: Learn by Building

A comprehensive implementation of **Adaptive Retrieval-Augmented Generation (RAG)** that intelligently routes queries between vectorstore retrieval and web search based on query characteristics.

## 🎯 What is Adaptive RAG?

Adaptive RAG is a smart approach that adjusts how information is retrieved and used based on the complexity and nature of a query. Instead of treating all queries the same way, it chooses the optimal strategy:

- **📚 Vectorstore Retrieval**: For questions answerable using indexed documents
- **🌐 Web Search**: For questions requiring real-time or external information
- **🔄 Self-Correction**: Validates and improves responses through multiple evaluation layers

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │  Router   │ ◄─── Determines best data source
    └─────┬─────┘      (vectorstore vs web search)
          │
     ┌────▼────┐
     │ Branch  │
     └─┬─────┬─┘
       │     │
   ┌───▼─┐ ┌─▼──────┐
   │ RAG │ │ Web    │
   │     │ │ Search │
   └─┬───┘ └─┬──────┘
     │       │
     └───┬───┘
         │
    ┌────▼────┐
    │ Graders │ ◄─── Validates relevance,
    │         │      hallucinations, and
    └─────────┘      answer quality
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/adaptive-rag
cd adaptive-rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
# Optional: For embeddings (if not using HuggingFace fallback)
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Basic Usage

```python
from src import AdaptiveRAG, DocumentProcessor

# Initialize components
doc_processor = DocumentProcessor()
rag_system = AdaptiveRAG()

# Load your documents
documents = doc_processor.load_csv("path/to/your/data.csv")

# Set up the system
rag_system.setup_vectorstore(documents)

# Ask questions!
answer = rag_system.query("How does machine learning work?")
print(answer)
```

### 4. Run Examples

```bash
# Basic example
python examples/basic_example.py

# Evaluation example with built-in metrics
python examples/evaluation_example.py
```

## 📁 Project Structure

```
adaptive-rag/
├── src/                          # Core implementation
│   ├── __init__.py
│   ├── adaptive_rag.py          # Main RAG system
│   ├── config.py                # Configuration settings
│   ├── models.py                # Data models
│   ├── router.py                # Question routing logic
│   ├── retrieval.py             # Document retrieval & web search
│   ├── rag_chain.py             # RAG generation chain
│   ├── graders.py               # Response validation
│   ├── evaluator.py             # Built-in evaluation system
│   └── document_processor.py    # Document loading & processing
├── examples/                     # Usage examples
│   ├── basic_example.py
│   └── evaluation_example.py
├── data/                        # Data directory
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔧 Core Components

### 1. Question Router
Routes queries to the appropriate data source based on content analysis:
- Analyzes query topics and context
- Routes to vectorstore for known topics
- Routes to web search for external/real-time information

### 2. Document Grader
Evaluates retrieved documents for relevance:
- Filters out irrelevant documents
- Ensures only contextually appropriate information is used

### 3. Hallucination Grader
Validates that generated responses are grounded in facts:
- Checks if answers are supported by retrieved documents
- Prevents fabricated or unsupported claims

### 4. Answer Grader
Ensures responses address the original question:
- Validates that answers are relevant and complete
- Triggers retry mechanisms if responses are inadequate

## 🚀 Key Features

- **🦙 Groq-Powered**: Uses Groq's lightning-fast LLM inference with Llama3 models
- **🎯 Smart Routing**: Automatically chooses between vectorstore and web search
- **📊 Built-in Evaluation**: Comprehensive metrics without external API dependencies
- **🔄 Self-Correction**: Multi-layer validation and response improvement
- **🛠️ Modular Design**: Easy to customize and extend
- **💡 Flexible Embeddings**: Supports OpenAI, HuggingFace, or custom embedding providers

## 📊 Evaluation

The system includes a **built-in evaluation framework** that provides comprehensive metrics without requiring external APIs:

```python
from src import SimpleEvaluator

# Initialize evaluator
evaluator = SimpleEvaluator(rag_system)

# Evaluate questions
questions = ["Your test questions here"]
results = evaluator.evaluate_batch(questions)

# Get detailed report
evaluator.print_detailed_report()

# Save results to CSV
evaluator.save_results_to_csv()
```

### Evaluation Metrics:
- ⏱️ **Response Time**: How fast the system responds
- 🔀 **Routing Analysis**: Vectorstore vs web search distribution
- 📝 **Answer Quality**: Length, citations, context usage
- 📊 **Performance Stats**: Min/max response times and answer lengths

## ⚙️ Configuration

Customize the system behavior in `src/config.py`:

```python
class Config:
    # Model settings (Groq)
    DEFAULT_MODEL = "llama3-8b-8192"
    ROUTER_MODEL = "llama3-70b-8192"
    
    # Retrieval settings
    RETRIEVAL_K = 4  # Documents to retrieve
    WEB_SEARCH_K = 3  # Web search results
    
    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 0
```

## 🎨 Customization

### Adding New Data Sources

1. **Custom Document Loaders**:
```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
# Add support for new file types
documents = processor.load_directory("./your_data/", "**/*.pdf")
```

2. **Custom Topics for Routing**:
```python
from src.router import QuestionRouter

router = QuestionRouter()
router.update_topics([
    "Your custom topic 1",
    "Your custom topic 2",
    # ... more topics
])
```

### Extending Evaluation

```python
# Use the built-in evaluator
from src import SimpleEvaluator

evaluator = SimpleEvaluator(rag_system)

# Add custom metrics or extend the evaluation
results = evaluator.evaluate_batch(questions)
report = evaluator.generate_report()

# Access detailed metrics
for result in evaluator.evaluation_results:
    print(f"Question: {result.question}")
    print(f"Route: {result.route_taken}")
    print(f"Response time: {result.response_time}s")
```

## 🔍 How It Works

1. **Query Analysis**: The router analyzes the incoming question
2. **Source Selection**: Decides between vectorstore or web search
3. **Information Retrieval**: Fetches relevant information from chosen source
4. **Document Grading**: Filters retrieved documents for relevance
5. **Response Generation**: Creates answer using RAG chain
6. **Quality Validation**: Checks for hallucinations and answer quality
7. **Self-Correction**: Retries with query transformation if needed

## 📈 Performance Tips

- **Vectorstore Optimization**: Use appropriate chunk sizes for your domain
- **Model Selection**: Choose models based on speed vs. accuracy trade-offs
- **Caching**: Implement caching for frequently accessed documents
- **Batch Processing**: Process multiple queries together for efficiency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the research paper: [Adaptive RAG](https://arxiv.org/pdf/2403.14403)
- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Simple evaluation system with comprehensive metrics

## 📞 Support

- 📧 Create an issue for bug reports or feature requests
- 💬 Join discussions in the Issues section
- 📖 Check the examples for common use cases

---

**Built with ❤️ for the AI community**
