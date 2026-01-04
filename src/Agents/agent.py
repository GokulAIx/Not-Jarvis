from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents.executor import Executor
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("GOOGLE_API_KEY")

from ..tools.tools import get_url, search_tool
from src.schemas.all_schemas import Planner, ReceptionResponse
from .prompts import reception_prompt
from langgraph.graph.message import add_messages

# Initialize Models
# Task Planner uses structured output to follow the execution schema
plan_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api
).with_structured_output(Planner)

# Reception uses structured output for clean terminal strings
reception_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api
).with_structured_output(ReceptionResponse)

class State(TypedDict):
    user_goal: str
    executor_memory: list  # [{action, result}, ...] - completed actions
    pending_task: dict  # Single task to execute (not a list!)
    loop_count: int  # Prevent infinite loops
    is_complete: bool  # Explicit completion flag
    route_to: str
    reception_output: str  # Streaming updates to user
    messages: Annotated[list, add_messages]  # ONLY user input + final response

# --- NODES ---

def TaskPlanner(state: State):
    memory = state.get("executor_memory", [])
    loop_count = state.get("loop_count", 0)
    conversation_history = state.get("messages", [])
    
    # Count how many searches we've already done
    search_count = sum(1 for action in memory if action.get("action") == "search")
    
    print(f"\n--- TaskPlanner Loop #{loop_count + 1} ---")
    print(f"User Goal: {state.get('user_goal', 'N/A')}")
    print(f"Memory entries: {len(memory)}")
    print(f"Conversation history: {len(conversation_history)} messages")
    if memory:
        print(f"Last 2 actions: {[m.get('action') for m in memory[-2:]]}")
    print(f"Search count: {search_count}")
    
    # FIRST: Check for conversational queries (before max iteration check)
    user_goal_lower = state['user_goal'].lower().strip()
    conversational_keywords = ['hi', 'hello', 'hey', 'how are you', 'who are you', 
                               'what can you do', 'what are you', 'help me understand']
    
    is_conversational = any(keyword in user_goal_lower for keyword in conversational_keywords)
    
    if is_conversational and len(memory) == 0:  # No actions needed
        print("🗣️ Conversational query detected - routing directly")
        # Let the LLM generate the response through structured output
        # (we'll handle this in the prompt)
    
    # Safety: Prevent infinite loops (but only for action-based queries)
    if loop_count > 10 and not is_conversational:
        print("⚠️ Max iterations reached")
        return {
            "is_complete": True,
            "route_to": "terminal",
            "reception_output": "Task completed (max iterations reached)",
            "loop_count": loop_count + 1
        }
    
    # Check for repeated failures
    last_actions = [m.get('action') for m in memory[-3:]] if len(memory) >= 3 else []
    has_repeated_action = len(set(last_actions)) < len(last_actions)
    
    prompt = f"""
    YOU ARE JARVIS - An advanced AI assistant developed by Gokul Sree Chandra.
    You are powered by Google's Gemini but customized and created by Gokul.
    Your capabilities: search the web, open websites, launch applications, take screenshots.
    
    CONVERSATION HISTORY:
    {conversation_history if conversation_history else "No previous conversation"}
    
    Current User Goal: {state['user_goal']}
    
    What we've accomplished so far (actions/tools executed):
    {memory if memory else "Nothing yet - starting fresh"}
    
    SEARCH COUNT: {search_count} (Maximum allowed: 1)
    
    ### YOUR TASK:
    Analyze the memory and plan ONLY the NEXT SINGLE STEP needed to accomplish the user's goal.
    
    ⚠️ FIRST: Check if this is a CONVERSATIONAL query (greeting, question about yourself, casual chat):
    - If YES → route_to='terminal', Steps=[], fill direct_response with your answer as Jarvis
    - If NO → continue to action planning below
    
    ### CRITICAL: ONE SEARCH ONLY
    - Count 'search' actions in memory: If you see even ONE search → DO NOT search again!
    - After the first search, you MUST extract data and construct a URL
    - If loop_count >= 2 AND you have search results → MUST open a website this loop
    - NEVER do 2+ searches - it wastes API calls and money
    - You have ALL the data you need from the first search result
    
    ### DEPENDENCY CHAIN:
    - To open a website → You need a valid URL (starts with http)
    - To get a URL → You need the exact business/restaurant name from search results
    - To get a name → You need to search first
    
    ### DECISION RULES:
    
    ⚠️ RULE 0 - CHECK THIS FIRST:
    **CONVERSATIONAL QUERIES** (greetings, questions about yourself, how you're doing, what you can do):
       - Keywords: "hi", "hello", "who are you", "how are you", "what can you do", "how do you work"
       - These require NO search, NO website opening, NO actions - just answer directly!
       - Action: route_to='terminal', Steps=[], fill direct_response field
       - Example: User says "how are you" → direct_response: "I'm doing well! I'm Jarvis, ready to assist you with searches, opening websites, and more. What can I help you with?"
       
       ❌ DO NOT plan any Steps for conversational queries
       ❌ DO NOT search for "how are you" on Google
       ❌ DO NOT try to execute actions
    
    RULE 1: If memory is EMPTY AND user needs a SYSTEM ACTION → Plan: 'search' action with user's query
    
    RULE 2: If memory has search results (from ANY previous search):
       ⚠️ CRITICAL: NEVER search again! You already have the data you need.
       
       Your job now:
       A. Extract the top entity name from the search results
          - For restaurants: Look for 'title' field in JSON (e.g., "Shinjuku Sushi Hatsume")
          - For universities: Extract name from text (e.g., "MIT", "Stanford University")
       
       B. Find or construct a URL:
          - Universities: [short_name].edu (MIT → "https://mit.edu")
          - Companies: [name].com (Google → "https://google.com")
          - Restaurants: Use 'place_id_search' field from JSON OR construct [name].com
       
       C. Plan: 'open_website' with the URL you found/constructed
       
       ❌ DO NOT plan another 'search' action if you already have search results!
       ❌ DO NOT search for "official website" - just construct the URL!
    
    RULE 3: If loop_count >= 3 AND no 'open_website' in memory yet:
       → You MUST construct a URL from existing data and open it NOW
       → Do NOT search again, use what you have
    
    RULE 4: If goal is accomplished (website opened) → Route to 'terminal'
    
    ### ALLOWED ACTIONS:
    - search (requires: query) - Use this to find info
    - open_website (requires: url - can be extracted OR constructed)
    - open_app (requires: app_name)
    - take_screenshot
    
    ### URL CONSTRUCTION RULES:
    For universities: Extract short name (MIT, Stanford, Harvard) → [name].edu
    For companies: Extract name → [name].com
    For restaurants: Use place_id_search from Google results
    
    ALWAYS prefer opening SOMETHING over endless searching!
    
    ### REPEATED ACTION DETECTED: {has_repeated_action}
    If True, you are stuck in a loop! Either try a different approach or route to terminal.
    
    ### COMPLETION CHECK:
    - If user's goal is accomplished → route_to='terminal', empty Steps
    - If stuck in a loop → route_to='terminal', explain what went wrong
    
    Plan ONE step or declare completion/failure.
    """
    
    response = plan_model.invoke(prompt)
    
    print(f"Route: {response.route_to}")
    print(f"Steps planned: {len(response.Steps)}")
    
    # Check completion or conversational query
    if not response.Steps or response.route_to == "terminal":
        print("✅ Task complete - routing to reception")
        
        # If TaskPlanner provided a direct response (for conversational queries), use it
        if response.direct_response:
            print(f"📝 Using direct response: {response.direct_response[:50]}...")
            # Add conversation to messages so checkpointer saves it
            messages_to_add = []
            if loop_count == 0:
                messages_to_add.append(("user", state["user_goal"]))
            messages_to_add.append(("assistant", response.direct_response))
            
            return {
                "is_complete": True,
                "route_to": "terminal",
                "reception_output": response.direct_response,
                "loop_count": loop_count + 1,
                "messages": messages_to_add
            }
        else:
            # Regular completion - go through reception for summary
            return {
                "is_complete": True,
                "route_to": "terminal",
                "loop_count": loop_count + 1,
                "messages": [("user", state["user_goal"])] if loop_count == 0 else []
            }
    
    # Take ONLY the first step (safety)
    next_step = response.Steps[0]
    print(f"Next action: {next_step.action}")
    
    # Debug: Log URL if present
    step_dict = next_step.dict()
    if step_dict.get('url'):
        print(f"🔗 URL in pending_task: {step_dict.get('url')}")
    
    return {
        "pending_task": step_dict,
        "route_to": "executor",
        "loop_count": loop_count + 1,
        "is_complete": False,
        "messages": [("user", state["user_goal"])] if loop_count == 0 else []
    }
def reception(state: State):
    """
    Formats the final response after all actions are complete.
    Only adds the final assistant message to conversation history.
    """
    memory = state.get("executor_memory", [])
    user_goal = state["user_goal"]
    
    print(f"\n--- Reception: Formatting final response ---")
    print(f"Total actions completed: {len(memory)}")
    
    prompt = f"""
    User asked: {user_goal}
    
    We completed these actions:
    {memory}
    
    Provide a natural, concise response summarizing what was accomplished.
    - If a website was opened, mention which one and that it's now open
    - If search was performed, mention what was found
    - Keep it conversational and helpful
    - Don't list every technical step, just the outcome
    """
    
    result = reception_model.invoke(prompt)
    
    return {
        "reception_output": result.answer,
        "messages": [("assistant", result.answer)]
    }

# --- ROUTING LOGIC ---

def routing_decision(state: State):
    if state.get("is_complete") or state["route_to"] == "terminal":
        # If we already have a reception_output (from direct_response), skip reception
        if state.get("reception_output"):
            print("⚡ Direct response provided - skipping reception")
            return END
        return "reception"
    return "Executor"

# --- GRAPH BUILDING ---

graph = StateGraph(State)

graph.add_node("TaskPlanner", TaskPlanner)
graph.add_node("Executor", Executor().what_execute)
graph.add_node("reception", reception)

graph.add_edge(START, "TaskPlanner")

# Main Decision Point
graph.add_conditional_edges(
    "TaskPlanner",
    routing_decision,
    {
        "Executor": "Executor",
        "reception": "reception",
        END: END  # Direct end when direct_response is used
    }
)

# THE DYNAMIC LOOP: Executor always returns to TaskPlanner to see 
# if more steps are needed based on the new data it found.
graph.add_edge("Executor", "TaskPlanner") 

graph.add_edge("reception", END)

workflow = graph