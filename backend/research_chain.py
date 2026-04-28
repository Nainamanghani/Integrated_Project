import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_tavily import TavilySearch
except ModuleNotFoundError:
    TavilySearch = None

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# RAG integration
from backend.rag.qa import retrieve_context
from backend.rag.config import settings as rag_settings

load_dotenv()

# ==================================================
# ⚡ LLM + External Tool Setup
# ==================================================

llm_client = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

if TavilySearch and os.getenv("TAVILY_API_KEY"):
    web_lookup = TavilySearch(max_results=4)
else:
    web_lookup = None


# ==================================================
# 🧠 State Blueprint
# ==================================================

class WorkflowState(TypedDict, total=False):
    topic: str
    raw_search: str
    rag_context: str          # ← NEW: retrieved document context
    summary_notes: str
    strategic_insights: str
    final_document: str
    followups: List[str]
    review_status: str
    iteration: int
    rag_project: str          # ← NEW: which Chroma namespace to query


# ==================================================
# 1️⃣ Relevance Filter
# ==================================================

relevance_prompt = PromptTemplate.from_template("""
You are an Energy Domain Validator.

Determine whether the following topic belongs to the energy sector
(renewables, fossil fuels, grids, EVs, batteries, sustainability, etc.).

Topic: {topic}

Respond ONLY with YES or NO.
""")

relevance_chain = relevance_prompt | llm_client | StrOutputParser()


def relevance_filter(state: WorkflowState):
    result = relevance_chain.invoke({"topic": state["topic"]}).strip().upper()
    if "YES" in result:
        return {"topic": state["topic"], "review_status": "CONTINUE"}
    return {
        "topic": state["topic"],
        "final_document": "This assistant specializes in energy-related topics only.",
        "followups": [],
        "review_status": "STOP",
    }


def filter_router(state: WorkflowState):
    if state.get("review_status") == "STOP":
        return "end"
    return "rag"   # ← goes to RAG node first, then intelligence


# ==================================================
# 2️⃣ RAG Node  ← NEW
# ==================================================

def rag_node(state: WorkflowState):
    """
    Queries the local Chroma vector store for document context
    relevant to the research topic.  Gracefully returns empty
    string when no documents have been uploaded yet.
    """
    topic = state["topic"]
    project = state.get("rag_project", rag_settings.default_project)

    try:
        context = retrieve_context(topic, project=project, top_k=5)
    except Exception:
        context = ""

    return {"rag_context": context}


# ==================================================
# 3️⃣ Intelligence Gathering (web search)
# ==================================================

research_prompt = PromptTemplate.from_template("""
You are an Energy Intelligence Analyst.

Using the search data below, create structured bullet insights.

Topic: {topic}

Search Data:
{data}
""")

research_chain_llm = research_prompt | llm_client | StrOutputParser()


def intelligence_node(state: WorkflowState):
    if web_lookup is None:
        search_data = (
            "Live web search is unavailable because Tavily is not configured. "
            "Proceed using model knowledge."
        )
    else:
        try:
            search_data = web_lookup.run(state["topic"])
        except Exception:
            search_data = (
                "Live web search failed at runtime. Proceed using model knowledge."
            )

    notes = research_chain_llm.invoke({
        "topic": state["topic"],
        "data": search_data,
    })

    return {
        "raw_search": search_data,
        "summary_notes": notes,
        "iteration": 0,
    }


# ==================================================
# 4️⃣ Strategic Analysis  (now uses RAG context too)
# ==================================================

analysis_prompt = PromptTemplate.from_template("""
From the research notes and any retrieved document context below, extract:

- Emerging Trends
- Business Implications
- Risk Factors

Research Notes:
{notes}

Retrieved Document Context (from uploaded PDFs — may be empty):
{rag_context}
""")

analysis_chain = analysis_prompt | llm_client | StrOutputParser()


def strategy_node(state: WorkflowState):
    insights = analysis_chain.invoke({
        "notes": state["summary_notes"],
        "rag_context": state.get("rag_context", "No document context available."),
    })
    return {"strategic_insights": insights}


# ==================================================
# 5️⃣ Report Composer
# ==================================================

writing_prompt = PromptTemplate.from_template("""
Create a professional energy sector report with:

1. Overview
2. Market Trends
3. Strategic Outlook
4. Risks & Challenges
5. Conclusion

Insights:
{insights}

If this is a revision, improve clarity and analytical depth.
""")

writing_chain = writing_prompt | llm_client | StrOutputParser()


def composer_node(state: WorkflowState):
    document = writing_chain.invoke({"insights": state["strategic_insights"]})
    return {
        "final_document": document,
        "iteration": state.get("iteration", 0) + 1,
    }


# ==================================================
# 6️⃣ Quality Review
# ==================================================

review_prompt = PromptTemplate.from_template("""
Evaluate the quality of this report.

If it is complete and well-structured, respond PASS.
Otherwise respond FAIL.

Report:
{report}
""")

review_chain = review_prompt | llm_client | StrOutputParser()


def quality_check_node(state: WorkflowState):
    verdict = review_chain.invoke({"report": state["final_document"]})
    if "PASS" in verdict.upper() or state.get("iteration", 0) >= 2:
        return {"review_status": "APPROVED"}
    return {"review_status": "REVISE"}


# ==================================================
# 7️⃣ Follow-up Generator
# ==================================================

followup_prompt = PromptTemplate.from_template("""
Based on this energy report, generate 3 intelligent follow-up questions.
Return only the questions separated by new lines.

Report:
{report}
""")

followup_chain = followup_prompt | llm_client | StrOutputParser()


def followup_node(state: WorkflowState):
    raw = followup_chain.invoke({"report": state["final_document"]})
    questions = [q.strip() for q in raw.split("\n") if q.strip()]
    return {"followups": questions[:3]}


# ==================================================
# 🔁 Flow Control Logic
# ==================================================

def review_router(state: WorkflowState):
    if state.get("review_status") == "APPROVED":
        return "followup"
    return "compose"


# ==================================================
# 🧩 Graph Assembly
# ==================================================
#
#  filter ──► rag ──► intelligence ──► strategy ──► compose ──► review ──► followup ──► END
#         └──────────────────────────────────────────────────────────────────────────► END (off-topic)

graph = StateGraph(WorkflowState)

graph.add_node("filter", relevance_filter)
graph.add_node("rag", rag_node)              # ← NEW node
graph.add_node("intelligence", intelligence_node)
graph.add_node("strategy", strategy_node)
graph.add_node("compose", composer_node)
graph.add_node("review", quality_check_node)
graph.add_node("followup", followup_node)

graph.set_entry_point("filter")

graph.add_conditional_edges(
    "filter",
    filter_router,
    {
        "rag": "rag",          # ← was "intelligence", now goes to RAG first
        "end": END,
    },
)

graph.add_edge("rag", "intelligence")        # RAG → web search
graph.add_edge("intelligence", "strategy")   # both contexts flow into strategy
graph.add_edge("strategy", "compose")
graph.add_edge("compose", "review")

graph.add_conditional_edges(
    "review",
    review_router,
    {
        "followup": "followup",
        "compose": "compose",
    },
)

graph.add_edge("followup", END)


# ==================================================
# 🚀 Public Interface
# ==================================================

def run_full_research(
    topic: str,
    thread_id: Optional[str] = None,
    rag_project: Optional[str] = None,
) -> dict:
    """
    Run the full research pipeline.

    Args:
        topic:       The energy research question.
        thread_id:   Optional conversation session ID for memory.
        rag_project: Chroma namespace to query.  Defaults to the
                     value in rag_settings.default_project.
    """
    memory = MemorySaver()
    workflow_app = graph.compile(checkpointer=memory)

    config = {
        "configurable": {
            "thread_id": thread_id or "energy_session"
        }
    }

    initial_state: WorkflowState = {
        "topic": topic,
        "rag_project": rag_project or rag_settings.default_project,
    }

    result = workflow_app.invoke(initial_state, config=config)

    return {
        "report": result.get("final_document", "No report generated."),
        "suggestions": result.get("followups", []),
        "rag_used": bool(result.get("rag_context", "").strip()),  # surfaced to API
    }
