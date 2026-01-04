# Not-Jarvis: Production AI Agent Architecture (v2.0)

**Last Updated**: January 4, 2026  
**Status**: Production-Ready Multi-Turn Conversational Agent with Persistent Memory

---

## Executive Summary

**Not-Jarvis (Jarvis)** is a production-grade multi-turn conversational AI assistant that demonstrates mastery of:
- **Iterative Single-Step Planning** (LangGraph agentic workflows)
- **Persistent Conversation Memory** (PostgreSQL checkpointer with Supabase)
- **Real-Time Streaming** (Server-Sent Events for live feedback)
- **Multi-Turn Context Awareness** (Remembers conversation history across requests)
- **System Integration** (Web search, browser control, app launching)

### Architecture Paradigm: Iterative Single-Step Planning

**Key Innovation**: Unlike traditional multi-step planners that generate full task lists upfront, this system:
1. Plans ONE action at a time
2. Executes that action
3. Analyzes results and conversation history
4. Plans next action based on new information
5. Repeats until goal accomplished

**Benefits**:
- ✅ Handles ambiguous queries (adapts based on search results)
- ✅ Recovers from failures (re-plans instead of following broken plan)
- ✅ Multi-turn awareness ("open their website" works after "find restaurant")
- ✅ Cost-efficient (only 1 search per query, constructs URLs from results)

---

## Technology Stack

```
Client:        Python CLI (requests + SSE streaming)
API:           FastAPI (async, streaming responses)
Orchestration: LangGraph StateGraph (iterative loop)
LLM:           Google Gemini 2.5 Flash (structured outputs)
Tools:         SerpAPI (web search), OS commands (browser/app control)
Memory:        Supabase PostgreSQL (AsyncPostgresSaver)
Identity:      "Jarvis" - AI assistant by Gokul Sree Chandra
```

---

## System Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT (client.py)                     │
│  - Hardcoded Thread ID: GOKUL_SREE_CHANDRA                 │
│  - Maintains session across all requests                    │
│  - SSE streaming consumption                                │
└────────────────┬────────────────────────────────────────────┘
                 │ POST /not-jarvis/stream
                 │ {user_goal, thread_id}
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI SERVER (main.py)                  │
│  - Async connection pool (max 10)                           │
│  - Loads checkpointer state from Supabase                   │
│  - Resets loop_count/executor_memory per request            │
│  - Preserves messages (conversation history)                │
└────────────────┬────────────────────────────────────────────┘
                 │ astream() with config
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              LANGGRAPH WORKFLOW (agent.py)                  │
│                                                              │
│     START → TaskPlanner → [routing] → END                   │
│                  ↑            ↓                              │
│                  └───Executor─┘                              │
│                                                              │
│  Loop Logic:                                                 │
│  1. TaskPlanner reads conversation_history + executor_memory│
│  2. Decides: conversational query? → direct_response + END  │
│  3. Or plans single action → Executor                        │
│  4. Executor executes → back to TaskPlanner                 │
│  5. Repeats until is_complete=True                          │
│  6. Reception formats final response (if no direct_response)│
└─────────────────────────────────────────────────────────────┘
```

### State Schema

```python
class State(TypedDict):
    user_goal: str                        # Current user request
    executor_memory: list                 # [{action, result}, ...] - completed actions
    pending_task: dict                    # Single task to execute (not a list!)
    loop_count: int                       # Iteration counter (reset per request)
    is_complete: bool                     # Explicit completion flag
    route_to: str                         # Routing decision: "executor" | "terminal"
    reception_output: str                 # Streaming updates to user
    messages: Annotated[list, add_messages]  # Conversation history (persisted)
```

**Key Design Decisions**:
1. **`executor_memory`**: Tracks tool executions within current request (search, open_website)
2. **`messages`**: Persists full conversation across requests via checkpointer
3. **`pending_task`**: Single dict (not list) - enforces one-action-at-a-time pattern
4. **`loop_count`**: Reset to 0 per request to prevent cross-request pollution
5. **`reception_output`**: Set by TaskPlanner for direct responses OR reception node for summaries

---

## File-by-File Breakdown

### 1. `main.py` - FastAPI Server & Entry Point

**Purpose**: HTTP API server with async support, connection pooling, and SSE streaming

**Key Components**:

```python
# Connection Pool Configuration
connection_pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=10,         # Max 10 concurrent connections
    timeout=30,
    max_idle=300,        # 5 min idle timeout
    kwargs={
        "autocommit": True,
        "row_factory": dict_row,
        "prepare_threshold": None  # Prevents type OID errors
    }
)
```

**Startup Sequence**:
```python
@app.on_event("startup")
async def startup():
    await connection_pool.open()
    checkpointer = AsyncPostgresSaver(connection_pool)
    await checkpointer.setup()  # Creates checkpoints tables
    app.state.checkpointer = checkpointer
```

**Request Handler**:
```python
@app.post("/not-jarvis/stream")
async def stream_agent(request: ChatRequest):
    # Debug logging
    print(f"Thread ID: {request.thread_id}")
    print(f"User Goal: {request.user_goal}")
    
    # Check for existing session
    existing_state = await checkpointer.aget(config)
    if existing_state:
        print(f"✅ Loaded session with {len(messages)} messages")
    
    # Reset per-request state, preserve conversation history
    async for event in app_instance.astream({
        "user_goal": request.user_goal,
        "loop_count": 0,           # Fresh counter
        "executor_memory": [],     # Fresh action log
        "is_complete": False
    }, config=config):
        # Stream reception_output events to client
        if "reception_output" in values:
            yield f"data: {output}\n\n"
    
    yield "data: [DONE]\n\n"
```

**Critical Details**:
- **State Reset**: `loop_count` and `executor_memory` reset per request
- **State Preservation**: `messages` (conversation history) loaded from checkpointer
- **SSE Format**: `data: <content>\n\n` for Server-Sent Events standard

---

### 2. `client.py` - Terminal Client

**Purpose**: Interactive CLI for testing with persistent session

**Key Implementation**:

```python
# Hardcoded thread ID for consistent sessions
SESSION_ID = "GOKUL_SREE_CHANDRA"

def chat_with_jarvis(user_input):
    url = "http://localhost:8000/not-jarvis/stream"
    payload = {
        "user_goal": user_input,
        "thread_id": SESSION_ID  # Same ID = persistent memory
    }
    
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                content = line.decode('utf-8').replace("data: ", "")
                if content == "[DONE]":
                    break
                print(content, end="", flush=True)
```

**Features**:
- ✅ Persistent session across terminal restarts
- ✅ Real-time streaming (no waiting for full response)
- ✅ Type `reset` to generate new UUID session
- ✅ Type `exit` to quit

**Why Hardcoded Thread ID**:
- Enables multi-turn conversations: "find restaurant" → "open their website"
- Checkpointer remembers search results from previous request
- Conversation history available for "what was my last question" queries

---

### 3. `src/Agents/agent.py` - Core Orchestration Logic

**Purpose**: LangGraph StateGraph with iterative single-step planning

#### Node A: TaskPlanner

**Responsibility**: Analyze context and plan next single action

**Input Sources**:
```python
def TaskPlanner(state: State):
    memory = state.get("executor_memory", [])      # Actions in current request
    conversation_history = state.get("messages", [])  # Full chat history
    loop_count = state.get("loop_count", 0)
    search_count = sum(1 for a in memory if a.get("action") == "search")
```

**Conversational Query Detection**:
```python
user_goal_lower = state['user_goal'].lower().strip()
conversational_keywords = ['hi', 'hello', 'how are you', 'who are you', ...]

if is_conversational and len(memory) == 0:
    # Skip execution, respond directly
    print("🗣️ Conversational query detected")
```

**LLM Prompt Structure**:
```python
prompt = f"""
YOU ARE JARVIS - AI assistant developed by Gokul Sree Chandra.
Capabilities: search web, open websites, launch apps, take screenshots.

CONVERSATION HISTORY:
{conversation_history}

Current User Goal: {state['user_goal']}

What we've accomplished (actions executed):
{memory}

SEARCH COUNT: {search_count} (Maximum allowed: 1)

### YOUR TASK:
Plan ONLY the NEXT SINGLE STEP.

⚠️ RULE 0: CONVERSATIONAL QUERIES
If user asks "hi", "how are you", "who are you" → 
  route_to='terminal', Steps=[], fill direct_response

RULE 1: If memory EMPTY and user needs action → Plan 'search'

RULE 2: If memory has search results →
  - NEVER search again! Extract URL from results
  - Construct URL (mit.edu, stanford.edu patterns)
  - Plan 'open_website' with constructed URL

RULE 3: If loop_count >= 3 without opening website →
  - MUST construct URL from available data NOW

RULE 4: If website opened → route_to='terminal'

### ALLOWED ACTIONS (use EXACT names):
- search (requires: query)
- open_website (requires: url)
- open_app (requires: app_name)
- take_screenshot
"""
```

**Structured Output**:
```python
plan_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
).with_structured_output(Planner)

response = plan_model.invoke(prompt)
# Returns: Planner(Steps=[...], route_to="...", direct_response="...")
```

**Return Logic**:
```python
# Case 1: Conversational query with direct answer
if response.direct_response:
    return {
        "is_complete": True,
        "reception_output": response.direct_response,
        "messages": [
            ("user", state["user_goal"]),
            ("assistant", response.direct_response)
        ]
    }

# Case 2: Action needed
else:
    next_step = response.Steps[0]
    return {
        "pending_task": next_step.dict(),
        "route_to": "executor",
        "is_complete": False
    }
```

**Debug Logging**:
```python
print(f"\n--- TaskPlanner Loop #{loop_count + 1} ---")
print(f"User Goal: {state['user_goal']}")
print(f"Memory entries: {len(memory)}")
print(f"Conversation history: {len(conversation_history)} messages")
print(f"Search count: {search_count}")
```

---

#### Node B: Executor

**Responsibility**: Execute single system action and return result

**File**: `src/Agents/executor.py`

```python
class Executor:
    def what_execute(self, state):
        task = state.get("pending_task")
        
        # Execute the action
        result = self.dispatch_actions([task])
        
        # Critical: Process handoff delay
        action_type = task.get('action')
        if action_type in ['open_website', 'open_app']:
            time.sleep(3)  # Ensures browser/app survives thread termination
        
        # Append to memory
        return {
            "executor_memory": state.get("executor_memory", []) + result,
            "pending_task": None
        }
```

**Action Dispatch**:
```python
def dispatch_actions(self, steps):
    from ..tools.tools import search_tool
    
    action_handlers = {
        "search": lambda query: search_tool.run(query),
        "open_website": self.open_website,
        "open_app": self.open_app,
        "take_screenshot": self.take_screenshot
    }
    
    for step in steps:
        handler = action_handlers[step["action"]]
        
        # Extract only parameters the handler accepts
        sig = inspect.signature(handler)
        params = {k: v for k, v in step.items() 
                  if k in sig.parameters and v is not None}
        
        # Debug logging
        if step["action"] == "open_website":
            print(f"🔍 Step URL: {step.get('url')}")
            print(f"🔍 Filtered params: {params}")
        
        result = handler(**params)
```

**Action Implementations**:
```python
def open_website(self, url: str) -> str:
    print(f"🌐 open_website called with: '{url}'")
    
    # Auto-add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        print(f"🔧 Added protocol: {url}")
    
    print(f'🚀 Executing: start "" "{url}"')
    os.system(f'start "" "{url}"')  # Windows command
    return f"Opening website: {url}"

def open_app(self, app_name: str) -> str:
    subprocess.Popen(["start", "", app_name], shell=True)
    return f"Opening {app_name}..."

def take_screenshot(self, save_path: str = None) -> str:
    save_path = save_path or f"screenshot_{datetime.now().strftime('%H%M%S')}.png"
    img = ImageGrab.grab()
    img.save(save_path)
    return f"Saved to {save_path}"
```

**Why 3-Second Delay**:
```
FastAPI async response → Python thread terminates
If OS hasn't fully detached browser process → Browser dies
time.sleep(3) → Ensures clean handoff to OS
```

---

#### Node C: Reception

**Responsibility**: Format final response when no direct_response exists

```python
def reception(state: State):
    memory = state.get("executor_memory", [])
    user_goal = state["user_goal"]
    
    prompt = f"""
    User asked: {user_goal}
    We completed: {memory}
    
    Provide natural summary of what was accomplished.
    """
    
    result = reception_model.invoke(prompt)
    return {
        "reception_output": result.answer,
        "messages": [("assistant", result.answer)]
    }
```

---

#### Routing Logic

```python
def routing_decision(state: State):
    if state.get("is_complete") or state["route_to"] == "terminal":
        # If direct_response already set, skip reception
        if state.get("reception_output"):
            return END
        return "reception"
    return "Executor"
```

**Graph Construction**:
```python
graph = StateGraph(State)
graph.add_node("TaskPlanner", TaskPlanner)
graph.add_node("Executor", Executor().what_execute)
graph.add_node("reception", reception)

graph.add_edge(START, "TaskPlanner")
graph.add_conditional_edges("TaskPlanner", routing_decision, {
    "Executor": "Executor",
    "reception": "reception",
    END: END
})
graph.add_edge("Executor", "TaskPlanner")  # Loop back!
graph.add_edge("reception", END)

workflow = graph.compile()
```

**The Iterative Loop**:
```
START → TaskPlanner → Executor → TaskPlanner → Executor → ... → END
              ↓                        ↓
            (conversational)       (action complete)
              ↓                        ↓
             END                   reception → END
```

---

### 4. `src/schemas/all_schemas.py` - Pydantic Schemas

**Purpose**: Type-safe contracts for LLM structured outputs

```python
class Planner_Steps(BaseModel):
    action: Literal["open_website", "open_app", "take_screenshot", "search"] = Field(
        description="MUST be exactly one of these action names"
    )
    app_name: Optional[str] = Field(None, description="Required for open_app")
    url: Optional[str] = Field(None, description="Required for open_website")
    query: Optional[str] = Field(None, description="Required for search action")

class Planner(BaseModel):
    Steps: List[Planner_Steps]
    route_to: str = Field(description="'executor' for actions, 'terminal' for Q&A")
    direct_response: Optional[str] = Field(
        description="Fill this for conversational queries"
    )

class ReceptionResponse(BaseModel):
    answer: str = Field(description="Final natural language response")
```

**Why Literal Types**:
- Prevents LLM hallucinations ("Open Google" → rejected)
- Forces exact match: `action="open_website"` only
- Eliminates parsing errors

---

### 5. `src/tools/tools.py` - External Tools

**Purpose**: Integration with SerpAPI and URL extraction

```python
from langchain_community.utilities import SerpAPIWrapper

serpy = os.getenv("SERPAPI_API_KEY")
searching_tool = SerpAPIWrapper(serpapi_api_key=serpy)

# Tool instance for agent
search_tool = Tool(
    name="search",
    description="Search the web for information",
    func=searching_tool.run
)
```

**get_url Function** (currently unused in workflow):
```python
@tool
def get_url(site_name: str) -> str:
    """Find official website URL using web search + LLM extraction"""
    search_query = f"{site_name} official website"
    search_results = searching_tool.run(search_query)
    
    prompt = f"""
    Extract ONLY the official website URL from:
    {search_results}
    Return JUST the URL or NO_URL_FOUND.
    """
    response = model.invoke(prompt).content.strip()
    
    # Regex extraction
    url_match = re.search(r'https?://[^\s<>"()]+', response)
    return url_match.group(0) if url_match else "NO_URL_FOUND"
```

**Why Not Used**:
- Adds extra SerpAPI call (costs money)
- Current approach: TaskPlanner constructs URLs from first search result
- Simpler, faster, more cost-effective

---

### 6. `src/Agents/prompts.py` - Prompt Templates

```python
reception_prompt = """
You are Jarvis, an advanced AI assistant developed by Gokul Sree Chandra.

Your personality:
- Professional yet friendly
- Powered by Google's Gemini but customized by Gokul

The executor completed: {executor_output}

INSTRUCTIONS:
1. If executor_output is empty (conversational query):
   - Introduce yourself as Jarvis
   - Mention capabilities

2. If search results → Extract and present info

3. If error → Explain simply

EXAMPLES:
User: "hi who are you"
Response: "Hello! I'm Jarvis, developed by Gokul Sree Chandra..."

User: "find restaurant"
Executor: [search results]
Response: "I found XYZ Restaurant. Opening their website..."
"""
```

---

### 7. `clear_db.py` - Database Maintenance

**Purpose**: Clear all checkpoint data from Supabase

```python
async def clear_checkpoints():
    pool = AsyncConnectionPool(conninfo=DB_URI, max_size=1)
    await pool.open()
    
    async with pool.connection() as conn:
        await conn.cursor().execute("DELETE FROM checkpoints;")
        await conn.cursor().execute("DELETE FROM checkpoint_writes;")
        print("✅ All checkpoint data cleared!")
    
    await pool.close()
```

**When to Use**:
- Testing fresh conversations
- Debugging state pollution
- Resetting after architecture changes

---

## Key Workflows

### Workflow 1: Conversational Query

**Example**: "hi who are you"

```
1. Client sends: {user_goal: "hi who are you", thread_id: "GOKUL_SREE_CHANDRA"}
2. FastAPI: Loads messages=[] (first time), resets loop_count=0
3. TaskPlanner:
   - Detects conversational keyword "hi", "who are you"
   - Fills direct_response: "Hello! I'm Jarvis..."
   - Returns: is_complete=True, reception_output=<response>
4. Router: Sees reception_output exists → routes to END (skips reception node)
5. Stream: Yields direct_response to client
6. Checkpointer: Saves messages=[("user", "hi..."), ("assistant", "Hello...")]
```

**Total LLM Calls**: 1 (TaskPlanner only)  
**Total Duration**: ~2 seconds

---

### Workflow 2: Multi-Turn Restaurant Query

**Example**: 
- Turn 1: "Find the best restaurant in Tokyo"
- Turn 2: "open their website"

#### Turn 1:

```
Loop #1:
- TaskPlanner: memory=[], search_count=0
  → Plans: {action: "search", query: "best restaurant in Tokyo"}
- Executor: Runs SerpAPI search
  → Returns: [{action: "search", result: "...Sukiyabashi Jiro..."}]
- Back to TaskPlanner

Loop #2:
- TaskPlanner: memory=[{action:"search",...}], search_count=1
  → Extracts "Sukiyabashi Jiro" from results
  → Constructs URL: "sukiyabashijiro.com"
  → Plans: {action: "open_website", url: "https://sukiyabashijiro.com"}
- Executor: Opens browser
  → Returns: [{action: "open_website", result: "Opening..."}]
- Back to TaskPlanner

Loop #3:
- TaskPlanner: Sees website opened, goal complete
  → route_to='terminal', is_complete=True
- Reception: Formats summary
  → "I found Sukiyabashi Jiro and opened their website."
- Checkpointer saves: messages=[("user","Find..."),("assistant","I found...")]
```

#### Turn 2 (Same Session):

```
1. Client sends: {user_goal: "open their website", thread_id: <same>}
2. FastAPI: Loads messages=[previous conversation], resets loop_count=0
3. TaskPlanner: 
   - Reads conversation_history
   - Sees previous search for "Sukiyabashi Jiro"
   - Understands "their" = Sukiyabashi Jiro
   - Constructs URL from context
   - Plans: {action: "open_website", url: "..."}
4. Executor: Opens website
5. Reception: "Opening Sukiyabashi Jiro's website for you."
6. Checkpointer appends to messages
```

**Key**: Turn 2 works because `messages` preserved the search context!

---

### Workflow 3: Rate Limit Optimization

**Problem**: Multiple SerpAPI calls waste money

**Solution**: search_count tracking

```python
search_count = sum(1 for action in memory if action.get("action") == "search")

prompt = f"""
SEARCH COUNT: {search_count} (Maximum: 1)

RULE: If search_count >= 1 → NEVER search again!
Extract URL from existing results and construct it.
"""
```

**Result**: 1 search per query (66% reduction from 3→1 calls)

---

## Architecture Decisions & Rationale

### 1. Why Iterative Single-Step Planning?

**Alternative**: Generate full task list upfront
```
User: "Find restaurant and open website"
Plan: [search, get_url, open_website]
Problem: If search returns place_id_search, get_url fails
```

**Our Approach**:
```
Loop 1: search
Loop 2: Analyze search results → construct URL → open_website
Benefit: Adapts to what search actually returns
```

### 2. Why Reset loop_count Per Request?

**Problem Without Reset**:
```
Request 1: Completes in 3 loops (loop_count=3)
Request 2: Starts at loop_count=3, hits max iterations immediately
```

**Solution**:
```python
async for event in app_instance.astream({
    "loop_count": 0,        # Reset
    "executor_memory": []   # Reset
    # messages preserved from checkpointer
}, config)
```

### 3. Why Hardcoded Thread ID?

**Alternative**: UUID per request
```python
thread_id = f"session_{uuid.uuid4()}"  # NEW ID EVERY TIME
Problem: "open their website" fails (no context from "find restaurant")
```

**Our Approach**:
```python
SESSION_ID = "GOKUL_SREE_CHANDRA"  # Same ID = persistent memory
Benefit: Multi-turn conversations work seamlessly
```

### 4. Why Skip get_url Tool?

**With get_url**:
```
Loop 1: search("best restaurant Tokyo")
Loop 2: get_url("Sukiyabashi Jiro")  ← Extra SerpAPI call!
Loop 3: open_website(url)
Total: 2 SerpAPI calls
```

**Without get_url**:
```
Loop 1: search("best restaurant Tokyo")
Loop 2: TaskPlanner constructs URL from search results → open_website
Total: 1 SerpAPI call (50% cost reduction)
```

### 5. Why 3-Second Delay After open_website?

**Problem**:
```
Executor launches browser → returns immediately
FastAPI completes response → Python thread terminates
Browser process not fully detached → OS kills it
Result: Browser opens and immediately closes
```

**Solution**:
```python
os.system(f'start "" "{url}"')
time.sleep(3)  # Keep thread alive for handoff
```

---

## Current Limitations & Future Work

### Limitations

1. **Windows-Only**: `os.system('start...')` is Windows-specific
2. **No Error Recovery**: If website unreachable, doesn't fall back
3. **URL Construction Heuristics**: `mit.edu` pattern works for universities, breaks for others
4. **No Observability**: No structured logging, metrics, or tracing
5. **Single User**: Hardcoded thread ID prevents multi-user deployment

### Planned Improvements

**Phase 1: Robustness**
- [ ] Cross-platform support (macOS, Linux)
- [ ] Retry logic for failed actions
- [ ] Fallback: If constructed URL fails, try get_url
- [ ] Structured logging (JSON logs for production)

**Phase 2: Scale**
- [ ] Multi-user support (dynamic thread IDs)
- [ ] Rate limiting per user
- [ ] Caching for frequent queries
- [ ] Metrics dashboard (OpenTelemetry)

**Phase 3: Features**
- [ ] Voice input/output integration
- [ ] Multi-modal support (images, PDFs)
- [ ] Browser automation (Selenium/Playwright)
- [ ] Calendar/email integration

---

## Development Commands

```bash
# Start server
python main.py

# Start client
python client.py

# Clear database (fresh start)
python clear_db.py

# Environment variables (.env)
GOOGLE_API_KEY=<gemini-key>
SERPAPI_API_KEY=<serpapi-key>
DATABASE_URL=postgresql://...
```

---

## Portfolio Readiness Assessment

**For $100k+ Remote US AI Engineering Roles**:

✅ **Strong**:
- Production-grade architecture (async, streaming, persistence)
- Modern AI stack (LangGraph, structured outputs, checkpointing)
- Sophisticated routing logic (iterative planning)
- Cost optimization (1 search per query)
- Multi-turn context awareness

⚠️ **Needs Work**:
- Error handling and recovery
- Observability and metrics
- Multi-user scalability
- Cross-platform support
- Unit/integration tests

**Recommendation**: Add Phase 1 improvements + blog post explaining architecture decisions → Portfolio-ready!
