# AMS GEMEENTE SMART AI AGENT

---

This repository contains the code for a sophisticated AI Agent application capable of answering user queries about amsterdam rules and policies by intelligently leveraging both a private, custom knowledge base (using Retrieval-Augmented Generation - RAG) and real-time web search. Users have granular control over the web search feature, enhancing flexibility and transparency.
---

## ✨ Key Features

- **Hybrid AI & Intelligent Routing**: Combines internal RAG knowledge with real-time web search, dynamically selecting the best information source for each query.

- **User-Controlled Web Access**: Provides a UI toggle to enable or disable web search, allowing users to choose between internal-only knowledge or broader internet access.

- **Transparent AI Workflow (Agent Trace)**: Offers a detailed, step-by-step trace of the agent's internal thought process, including routing decisions, RAG sufficiency verdicts, and information retrieval summaries.

- **Contextual RAG Sufficiency Judgment**: Employs an LLM to critically assess if retrieved RAG content is sufficient to answer a query, preventing incomplete responses and prompting further search if needed.

- **Dynamic Knowledge Ingestion (PDF Upload)**: Users can upload PDF documents directly, which are automatically processed, embedded, and added to the agent's Pinecone knowledge base.

- **Modular & Extensible Design**: Clean, layered architecture (FastAPI, LangGraph, Streamlit) makes it easy to understand, debug, and expand.

- **Persistent Conversation Memory**: LangGraph's checkpointing maintains conversation context across multiple turns.


---

## 🚀 High-Level Architecture

### 🧩 Layers Overview:

- **User Interface (UI)**: Streamlit app for interaction.
- **API Layer**: FastAPI backend that receives and handles requests.
- **Agent Core**: LangGraph-powered AI logic with routing and tools.
- **Knowledge Base**: Pinecone vector DB + HuggingFace embeddings.
- **External Tools**: Anthropic LLM, Tavily Search API.


---

## ⚙️ Technology Stack

- **Language**: Python 3.12+
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Agent Orchestration**: LangGraph
- **LLMs & Tools**: LangChain, Anthropic (Llama 3)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: Pinecone
- **PDF Processing**: PyPDFLoader
- **Search Engine**: Tavily API

---
## 🛠️ Setup and Installation

### Prerequisites

- Python 3.9+
- API Keys:

  - `GROQ_API_KEY`
  - `PINECONE_API_KEY`
  - `TAVILY_API_KEY`

- Pinecone index: `rag-index` with 384 dimensions and cosine metric

### Installation

```bash
git clone https://github.com/vkobayashi/ams-agent-chatbot.git
cd ams-agent-chatbot
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Create a `.env` file at project root:

```dotenv
ANTHROPIC_API_KEY="your_groq_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_ENVIRONMENT="your_pinecone_environment"
TAVILY_API_KEY="your_tavily_api_key"
FASTAPI_BASE_URL="http://localhost:8000"
```

---

## 🏃 Running the Application

Access the application here

[ams agent chatbot](https://ams-agent-chatbot-prototype.streamlit.app/)

---

## 🚀 Future Improvements

- Integrate tools: calculator, calendar, code interpreter
- Stream LLM output token-by-token
- Advanced RAG techniques: reranking, multi-query
- Long-term memory database for chat history
- User authentication & profiles
- Enhanced UI: dark mode, animations, custom themes

---

## 📬 Feedback & Contributions

Feel free to open issues or PRs for suggestions, bugs, or enhancements.

> Built with ❤️ using LangGraph, LangChain, Anthropic, and Streamlit


