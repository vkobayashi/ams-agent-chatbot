# Import dependencies
import os
from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
#from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig


from config import TAVILY_API_KEY
from config import ANTHROPIC_API_KEY
# from config import GROQ_API_KAY
from vectorstore import get_retriever

# Tools

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"

@tool
def rag_search_tool(query: str) -> str:
    """Top-k chunks from KB (empty dtrong if none)"""

    try:
        retriever_instance= get_retriever()
        docs = retriever_instance.invoke(query, k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"

# Pydantic schemas for structured output
class RouteDecision(BaseModel):
    """Schema for the agent's route decision"""
    route: Literal["rag", "web", "answer", "end"] 
    reply: str | None =Field(None, description="Filled only when route ==  'end' ")

class RagJudge(BaseModel):
    sufficient:bool = Field(..., description="True if retrieved information is sufficient to answer the user's " \
    "question, False otherwise.")   

# LLM instances with structured schemas

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# replace with gpt-5.4-nano
router_llm =ChatLiteLLM(model_name="anthropic/claude-sonnet-4-5-20250929",temperature=0.1).with_structured_output(RouteDecision)
judge_llm =ChatLiteLLM(model_name="anthropic/claude-sonnet-4-5-20250929",temperature=0.1).with_structured_output(RagJudge)
answer_llm =ChatLiteLLM(model_name="anthropic/claude-sonnet-4-5-20250929",temperature=0.7)


# State : Shared Data Structure
class AgentState(TypedDict, total=False):
    """Shared data structure for the agent's state"""
    messages: List[BaseMessage]
    route: Literal["rag","web", "answer","end"]
    rag:str
    web:str
    web_search_enabled: bool

#node : for individual functions
# node 1 : router(decision node)
def router_node(state:AgentState,config : RunnableConfig)->AgentState:
    print("Entering router node")
    #extract query
    query=next((m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")
    web_search_enabled = config.get("configurable",{}).get("web_search_enabled",True)
    print(f"Router received web search info: {web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
        "Always answer in the context of the city /gemeente of Amsterdam"

    )

    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general "
            "knowledge "
            "that is unlikely to be in a specific , static knowledge base (e.g., website of amsterdam gemeente). "
            "Always respond in the context of the city /gemeente of Amsterdam."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, policy details, procedures, or any"
            "information that would typically be found in a curated document collection (e.g., 'What is X?','How does Y work?','Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent information, or broad general knowledge that require "
            "up-to-date internet access (e.g., 'Who won in the last election?','What is the latest news on topic X?','Latest policy on sustainability'). "
            "use this website as the main source https://www.amsterdam.nl/en/policy/"
        )

    else:
        system_prompt += (
            "**Web search** is currently DISABLE.** You **MUST NOT** choose the 'web' route. "
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly for your general knowledge."
            "that is unlikely to be in a specific , static knowledge base."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, policy details, procedures, or any"
            "information that would typically be found in a curated document collection, AND for queries that would"
            "normally go to a web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi','How are you?')."
        "\n\nExample routing decisions:"
        "\n- User: 'What is the policy for sustainability?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the budget for 2027?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Which parties form the coalition in amsterdam?' -> Route: 'web' (Current event, requires information from amsterdam gemeente website)."
        "\n- User: 'How do I submit an anti-discrimination complaint?' -> Route: 'rag' (Internal procedure or refer to the amsterdam gemeente website)."
        "\n- User: 'Tell me about Reading initiative.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! I am the amsterdam gemeente AI assistant. How can I assist you today?'"
    )  

    messages = [
        ("system", system_prompt),
        ("user", f"User query: '{query}'\n\nPlease decide the best route for this query. Respond in JSON format with 'route' and optional 'reply' if route is 'end'.")
    ]

    result : RouteDecision=router_llm.invoke(messages)
    initial_router_decision = result.route
    router_overrider_reason = None

    # overide the router decision to go for web search

    if not web_search_enabled and  result.route=="web":
        result.route=="rag"
        router_overrider_reason="Web search disabled by user; redirected to RAG"
        print(f"Router decision overriden :  changed from 'web' to 'rag' .")

    print(f"Router final decision:{result.route}, reply (if 'end'):{result.reply}")

    out={
        "messages": state['messages'],
        "route":result.route,
        "web_search_enabled": web_search_enabled

    }

    if router_overrider_reason:
        out["initial_router_decision"]=initial_router_decision
        out["router_overrider_reason"]=router_overrider_reason

    if result.route == "end":
        out["messages"]=state["messages"]+[AIMessage(content=result.reply or "Hello!")]

    print("Existing router_node")
    return out

# Node 2: RAG Lookup

def rag_node(state:AgentState, config:RunnableConfig)-> AgentState:
    print("Entering rag_node")
    query=next((m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")
    web_search_enabled=config.get("configurable",{}).get("web_search_enabled",True)
    print(f"Router received web search info: {web_search_enabled}")
    print(f"RAG Query: {query}")

    chunks = rag_search_tool.invoke(query)

    #logic to handle chunk

    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error :{chunks}, checking web search enabled status")
        # if rag fails, and web search is enabled
        next_route="web" if web_search_enabled else "answer"
        return {**state, "rag":"", "route":next_route}
    if chunks:
        print(f"Retrieved RAG chunks (first 500 chars) : {chunks[:500]}...")
    else:
        print("No RAG chunks retrieved. ")

    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
            "\n\nExample 1: Question: 'In what year will Amsterdam become a waste-free city?' Retrieved: 'By 2050.' -> {\"sufficient\": true}"
            "\nExample 2: Question: 'What are the plans of amsterdam gemeente to reduce nuisance?' Retrieved: 'Asmterdam is known as a free and open city.' -> {\"sufficient\": false} (Doesn't answer plans)"
            "\nExample 3: Question: 'What programme supports social enterprises?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]

    verdict: RagJudge=judge_llm.invoke(judge_messages)
    print(f"RAG Judge verdict :{verdict.sufficient}")
    print("Exiting rag_node")

    # Decide next route based on sufficiency and web_search info
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer"
        print(f"RAG nor sufficient. Web search enabled : {web_search_enabled}. Next route:{next_route}")

    return {
        **state,
        "rag":chunks,
        "route":next_route,
        "web_search_enabled": web_search_enabled
    }


# Node 3: Web search
def web_node(state: AgentState,config:RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    # Check if web search is actually enabled before performing it
    # MODIFIED: Get web_search_enabled directly from the config
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True) # <-- CHANGED LINE
    print(f"Router received web search info : {web_search_enabled}")
    if not web_search_enabled:
        print("Web search node entered but web search is disabled. Skipping actual search.")
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    print(f"Web search query: {query}")
    snippets = web_search_tool.invoke(query)
    
    if snippets.startswith("WEB_ERROR::"):
        print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
        return {**state, "web": "", "route": "answer"}

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        # If web search was disabled, the 'web' field might contain a message like "Web search was disabled..."
        # We should only include actual search results here.
        if state["web"] and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts)
    if not context.strip():
        context = "No external context was available for this query. Try to answer based on general knowledge if possible."

    prompt = f"""Please answer the user's question using the provided context.
If the context is empty or irrelevant, try to answer based on your general knowledge.

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
    print(f"Final answer generated: {ans[:200]}...")
    print("--- Exiting answer_node ---")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }

# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"

# --- Build graph ---
def build_agent():
    """Builds and compiles the LangGraph agent."""
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")
    
    g.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        }
    )
    
    g.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer": "answer",
            "web": "web_search"
        }
    )
    
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    agent = g.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()