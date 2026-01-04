from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents.executor import Executor
from dotenv import load_dotenv
import os
load_dotenv()
api = os.getenv("GOOGLE_API_KEY")
from ..tools.tools import get_url,search_tool
from src.schemas.all_schemas import Planner ,ReceptionResponse
from .prompts import reception_prompt
from typing import Annotated
from langgraph.graph.message import add_messages

Tools=[get_url,search_tool]
class State(TypedDict):
    user_goal: str
    planned_tasks: Planner
    completed_tasks: list[str]
    pending_tasks: list[str]
    executor_output: list[str]
    reception_output: str
    route_to: str
    internal_answer: str
    messages: Annotated[list, add_messages]


model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools)

plan=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools).with_structured_output(Planner)

reception_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api
).with_structured_output(ReceptionResponse)

#NODES 
# NODES 
def TaskPlanner(state: State):
    history = state.get("messages", [])
    prompt = (
        f"Conversation History: {history}\n"
        f"User Goal: {state['user_goal']}\n\n"
        "You are Jarvis, a professional AI assistant created by Gokul.\n"
        "Decide if a request needs system execution ('executor') or a direct answer ('terminal').\n\n"
        "ALLOWED ACTIONS (use EXACTLY these names):\n"
        "- 'open_website' (requires 'url' field)\n"
        "- 'open_app' (requires 'app_name' field)\n"
        "- 'take_screenshot' (optional 'save_path')\n\n"
        "RULES:\n"
        "1. For 'terminal': Provide the full answer in 'direct_response'.\n"
        "2. For 'executor': Provide a brief confirmation (e.g., 'Opening Chrome...') in 'direct_response'.\n"
        "3. CRITICAL: Use ONLY the exact action names listed above. No variations!\n"
        'You are not supposed to make up urls or makeup names or anything if you are given several tools at your disposal use those accordingly you have a web search tool called server api tool And you have a get url tool you have multiple tools at your disposal So all you have to do is generate plans using those tools'
    )
    
    response = plan.invoke(prompt)
    
    # We populate 'reception_output' immediately to trigger the stream
    display_text = response.direct_response or "Processing your request..."
    
    print(f"--- ROUTING: {response.route_to} | STEPS: {len(response.Steps)} ---")

    return {
        "planned_tasks": response,
        "pending_tasks": response.Steps,
        "route_to": response.route_to,
        "internal_answer": display_text,
        "reception_output": display_text, # <--- Immediate feedback happens here
        "messages": [("user", state["user_goal"])]
    }

def reception(state: State):
    # If we are in 'terminal' mode, TaskPlanner already sent the output.
    # We return NOTHING for reception_output here to avoid the "Double Print".
    if state.get("route_to") == "terminal":
        return {
            "messages": [("assistant", state.get("internal_answer", ""))]
        }

    # If we are in 'executor' mode, TaskPlanner sent the "Starting" message.
    # Now we send the "Finished" message.
    prompt = reception_prompt.format(
        planned_tasks=state['planned_tasks'],
        executor_output=state.get('executor_output', 'Task finished.'),
        get_url=get_url,
        search_tool=search_tool
    )
    
    # Force clean string via structured output
    result = reception_model.invoke(prompt)
    
    return {
        "reception_output": f"\n[System Status]: {result.answer}", 
        "messages": [("assistant", result.answer)]
    }


def routing_decision(state: State):
    # Agnostic routing
    if state["route_to"] == "executor":
        return "Executor"
    return "reception"



#GRAPH BUILDING
graph = StateGraph(State)

graph.add_node("TaskPlanner", TaskPlanner)
graph.add_node("Executor", Executor().what_execute)
graph.add_node("reception", reception)

graph.add_edge(START, "TaskPlanner")

# The Intermediate Logic Node
graph.add_conditional_edges(
    "TaskPlanner",
    routing_decision,
    {
        "Executor": "Executor",
        "reception": "reception"
    }
)

graph.add_edge("Executor", "reception") 
graph.add_edge("reception", END)

workflow = graph