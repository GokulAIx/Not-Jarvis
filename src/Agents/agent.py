from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents.executor import Executor
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("GOOGLE_API_KEY")

from ..tools.tools import search_tool
from src.schemas.all_schemas import Planner, ReceptionResponse
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
    # ONLY greetings and identity questions - NOT informational queries
    conversational_keywords = [
        'hi', 'hello', 'hey', 'how are you', 'who are you', 
        'what can you do', 'what are you'
    ]
    
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
    YOU ARE Not-JARVIS - An advanced AI assistant developed by Gokul Sree Chandra.
    You are powered by Google's Gemini but customized and created by Gokul.
    Your capabilities: search the web, open websites, launch applications, take screenshots.
    
    CONVERSATION HISTORY:
    {conversation_history if conversation_history else "No previous conversation"}
    
    Current User Goal: {state['user_goal']}
    
    **CRITICAL INSTRUCTION - REPEATED REQUESTS**:
    - If user asks the SAME TASK again (even if it's in conversation history), DO IT AGAIN
    - NEVER say "I already did this" or "I have already found/opened..."
    - ALWAYS execute the task fresh as if it's the first time
    - Example: If "Find MIT website" was done before, do it again when asked again
    - The conversation history is for context, NOT to skip repeating tasks
    
    **ANSWERING REPEATED QUESTIONS**:
    - If user asks the same QUESTION again, answer it fully again
    - Do NOT say "I already answered this" - just provide the answer
    - Users may want to hear it again or may have forgotten
    
    What we've accomplished so far (actions/tools executed):
    {memory if memory else "Nothing yet - starting fresh"}
    
    SEARCH COUNT: {search_count} (Maximum allowed: 2)
    
    ### YOUR TASK:
    Analyze the memory and plan ONLY the NEXT SINGLE STEP needed to accomplish the user's goal.
    
    **CRITICAL**: Return Steps as a list with EXACTLY ONE item (or empty list if done).
    - ✅ Correct: Steps=[{{action: "search", query: "..."}}]
    - ❌ Wrong: Steps=[{{...}}, {{...}}, {{...}}]  ← Multiple steps are IGNORED
    
    We execute ONE step, then re-plan based on results. Multi-step plans are wasteful.
    
    ⚠️ FIRST: Check if this is a CONVERSATIONAL query (greeting, question about yourself, casual chat):
    - If YES → route_to='terminal', Steps=[], fill direct_response with your answer as Jarvis
    - If NO → continue to action planning below
    
    ### SEARCH STRATEGY (MAX 2 SEARCHES):
    ⚠️ **HARD LIMIT ENFORCEMENT**:
    - Current search count: {search_count}
    - If search_count >= 2: **ABSOLUTELY NO MORE SEARCHES ALLOWED**
    - DO NOT plan 'search' action if search_count >= 2
    - Use existing data or route to terminal with explanation
    
    - First search: Use the user's query as-is
    - **FOLLOW-UP SEARCH ALLOWED** if first result is a ranking/aggregator site:
      • Check [EXTRACTED_URL] - if it contains: "ranking", "top", "best", "list", "topuniversities", "usnews", etc.
      • Then do ONE more search with "official site" or institution name added
      • Example: First search → got topuniversities.com → Second search: "MIT official site"
    - After 2 searches, you MUST use whatever URL you have or route to terminal
    
    ### DEPENDENCY CHAIN:
    - To open a website → You need a valid URL (starts with http)
    - To get a URL → You need the exact business/restaurant name from search results
    - To get a name → You need to search first
    
    ### DECISION RULES:
    
    ⚠️ RULE 0 - CHECK THIS FIRST:
    **CONVERSATIONAL QUERIES** (greetings, questions about yourself):
       - Keywords: "hi", "hello", "how are you", "who are you", "what can you do"
       - These require NO search, NO website opening - just answer directly!
       - Action: route_to='terminal', Steps=[], fill direct_response field
       - Example: "hi" → direct_response: "Hello! I'm Jarvis..."
       
    **INFORMATIONAL QUERIES** ("what is", "explain", etc.):
       - YOU decide if search is needed:
         • General knowledge you know → Answer directly (route to terminal)
         • Real-time/specific info → Search first
       - Examples:
         • "What is quantum computing" → You know this, answer directly
         • "What is the weather in Tokyo" → Search needed (real-time)
         • "What is the best restaurant in NYC" → Search needed (current info)
       
       ❌ DO NOT plan any Steps for greetings/identity queries
       ❌ DO use search when you need current/specific information
    
    🎯 RULE 1 - **CHECK COMPLETION FIRST (HIGHEST PRIORITY)**:
    **Before planning ANY action, check if user's goal is ALREADY accomplished:**
    
    Compare what user asked for vs what's in memory:
    - "Find X and open website" + memory has [search, open_website] = ✅ DONE → route to terminal
    - "Open website and screenshot" + memory has [open_website, take_screenshot] = ✅ DONE → route to terminal
    - "Search for X" + memory has [search] = ✅ DONE → route to terminal
    - "Open website and screenshot" + memory has [open_website] only = ❌ NOT DONE → plan screenshot
    
    **If goal is met:**
    - Set route_to='terminal'
    - Set Steps=[]
    - DO NOT plan any more actions
    
    **If goal is NOT met:**
    - Continue to RULE 2 below
    
    RULE 2: If memory is EMPTY AND user needs a SYSTEM ACTION → Plan: 'search' action with user's query
    
    RULE 3: If memory has search results (from the FIRST search):
    RULE 3: If memory has search results (from the FIRST search):
       **Check the [EXTRACTED_URL]**:
       
       A. If URL looks good (official site, not a ranking/list site):
          → Copy exact URL into open_website action
          → Example: [EXTRACTED_URL]: https://web.mit.edu → use it!
       
       B. If URL is a RANKING/AGGREGATOR SITE (and search_count < 2):
          → Detect: URL contains "ranking", "top", "best", "list", "topuniversities", "usnews"
          → Extract the entity name from search results text (e.g., "Massachusetts Institute of Technology")
          → Do ONE more search: "[entity name] official site"
          → Stream message: "🔎 Found ranking site - searching for official website...\n"
       
       C. If you already did 2 searches (search_count >= 2):
          → Use best available URL or route to terminal explaining limitation
          → **CRITICAL**: DO NOT search a third time - it's BLOCKED
       
       **CRITICAL**: Copy URLs EXACTLY from [EXTRACTED_URL] - do NOT modify!
    
    RULE 4: If loop_count >= 4 AND no 'open_website' in memory yet:
    RULE 4: If loop_count >= 4 AND no 'open_website' in memory yet:
       → You MUST construct a URL from existing data and open it NOW
       → Do NOT search again, use what you have
    
    ### REPEATED ACTION DETECTED: {has_repeated_action}
    If True, you are stuck in a loop! Either try a different approach or route to terminal.
    
    ### ALLOWED ACTIONS:
    - search (requires: query) - Searches web and auto-extracts top URL
    - open_website (requires: url - Copy from [EXTRACTED_URL] tag)
    - open_app (requires: app_name)
    - take_screenshot
    
    ### URL EXTRACTION (AUTOMATED):
    **The search tool does the work for you!**
    
    When you run search, the result includes:
    ```
    [search results text...]
    
    [EXTRACTED_URL]: https://actual-website.com
    ```
    
    Your job: Copy that URL exactly into open_website action.
    
    ❌ DO NOT construct URLs - use [EXTRACTED_URL]
    ❌ DO NOT modify the extracted URL
    ❌ DO NOT search for "official website" - first search has the URL
    
    ALWAYS prefer opening SOMETHING over endless searching!
    
    ### REPEATED ACTION DETECTED: {has_repeated_action}
    If True, you are stuck in a loop! Either try a different approach or route to terminal.
    
    ### COMPLETION CHECK:
    - **Compare user's goal with completed actions in memory**
    - Ask yourself: "Did I accomplish what the user asked for?"
    - Examples:
      • Goal: "Find MIT and open website" + Actions: [search, open_website] = ✅ DONE
      • Goal: "Open website and screenshot" + Actions: [open_website] = ❌ Need screenshot
      • Goal: "Take a screenshot" + Actions: [take_screenshot] = ✅ DONE
    - If accomplished → route_to='terminal', Steps=[]
    - If NOT accomplished → plan the next missing action
    - If stuck/confused → route_to='terminal', Steps=[], explain issue
    
    **REMEMBER**: Return Steps with EXACTLY ONE item, or empty list if complete.
    We will execute that step, analyze the result, then ask you to plan the NEXT step.
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
    
    step_dict = next_step.dict()
    
    # Generate user-friendly status message
    action_messages = {
        "search": f"🔍 Searching for: {step_dict.get('query', 'information')}...\n",
        "open_website": f"🌐 Opening website...\n",
        "open_app": f"🚀 Launching {step_dict.get('app_name', 'application')}...\n",
        "take_screenshot": "📸 Taking screenshot...\n"
    }
    
    status_message = action_messages.get(next_step.action, f"⚙️ Executing {next_step.action}...")
    
    return {
        "pending_task": step_dict,
        "route_to": "executor",
        "loop_count": loop_count + 1,
        "is_complete": False,
        "reception_output": status_message,  # ← Stream intermediate update
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

graph.add_edge("Executor", "TaskPlanner") 

graph.add_edge("reception", END)

workflow = graph