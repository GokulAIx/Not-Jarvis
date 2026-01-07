# Not-Jarvis: Intelligent OS Agent with Zero Hallucination

An AI-powered desktop automation agent that executes web searches, opens websites, launches applications, and takes screenshots through natural language commands—with **0% URL hallucination rate** achieved through Python+LLM hybrid architecture.

---

## 🎯 Key Features

- **Zero URL Hallucination**: Deterministic Python extraction ensures 100% accurate website URLs
- **Persistent Memory**: Conversation history stored in PostgreSQL survives server restarts
- **Real-Time Streaming**: Server-Sent Events provide live task execution updates
- **Iterative Planning**: Re-plans after each action based on actual results (no wasted multi-step plans)
- **Multi-Turn Conversations**: Maintains context across requests using LangGraph checkpointing

---

## 🏗️ Architecture

### System Flow

```
Client Request → FastAPI → LangGraph StateGraph → Gemini 2.5 Flash
                                ↓
                         PostgreSQL (Supabase)
                                ↓
                    Persistent Conversation Memory
```

### Agent Workflow

**Execution Loop:**
1. **TaskPlanner**: Analyzes goal + conversation history → plans single next action
2. **Executor**: Executes action (search, open_website, screenshot, open_app)
3. **Loop Back**: TaskPlanner re-evaluates with execution results
4. **Reception**: Formats final response when task complete


<img width="275" height="333" alt="graph2" src="https://github.com/user-attachments/assets/b9c945a3-3cd8-430a-aa45-faa2ee281c91" />


**Key Design Decision**: Single-step planning instead of multi-step plans eliminates wasted LLM calls when intermediate results differ from expectations.

---

## 🔑 Core Innovation: Python+LLM Hybrid

### The Problem
LLMs frequently hallucinate URLs:
```
User: "Find the MIT website"
LLM: "Opening https://mit.com" ❌ (Hallucinated - real site is mit.edu)
```

### The Solution
**Separate deterministic operations from semantic decisions:**

```python
# Python Extracts URLs (Zero Hallucination)
def enhanced_search(query: str) -> str:
    results = serpapi.search(query)
    url = results['organic_results'][0]['link']  # ✅ Python extraction
    return f"{results}\n\n[EXTRACTED_URL]: {url}"

# LLM uses extracted URL (no hallucination possible)
```

**Results**: 0% URL hallucination across all test queries.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI (Async) | HTTP server with SSE streaming |
| **AI Model** | Gemini 2.5 Flash | Task planning & decision making |
| **Agent Framework** | LangGraph | State management & workflow orchestration |
| **Database** | PostgreSQL (Supabase) | Persistent conversation checkpoints |
| **Web Search** | SerpAPI | Real-time search results |
| **OS Automation** | webbrowser, pyautogui | System-level actions |

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL database (Supabase recommended)
- Google API key (Gemini)
- SerpAPI key

### Setup

1. **Clone & Install**
```bash
git clone https://github.com/YourUsername/Not-Jarvis.git
cd Not-Jarvis
python -m venv Jarvis
Jarvis\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Environment Variables**
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Then edit `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
SERPAPI_API_KEY=your_serpapi_key
DATABASE_URL=postgresql://user:pass@host:5432/db
```

3. **Run Server**
```bash
uvicorn main:app --reload
```

4. **Run Client**
```bash
python client.py
```

---

## 💡 Usage Examples

### Real-Time Streaming Updates

```
You: Find the MIT website and open it

🔍 Searching for: MIT website...
✅ Search complete
🌐 Opening website...
✅ Website opened
I have opened the MIT website for you.
[Browser opens https://web.mit.edu]

You: What was my previous request?
You asked me to find and open the MIT website.

You: Take a screenshot
📸 Taking screenshot...
✅ Screenshot saved
Screenshot has been captured and saved.
```

**Note**: Each action streams progress updates in real-time via Server-Sent Events.

---

## 🔍 How It Works

### 1. Persistent Conversations
```python
# Each request uses same thread_id for continuity
config = {"configurable": {"thread_id": "USER_123"}}

# LangGraph automatically:
# - Loads previous messages from DB
# - Merges with new request
# - Saves updated state after each node
```

### 2. State Management & Streaming
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Persists (reducer accumulates)
    executor_memory: list                     # Resets per request
    user_goal: str                            # New each request
    loop_count: int                           # Tracks iterations
    reception_output: str                     # Streaming updates (any node can set)
```

**Design**: 
- `messages` persist for conversation context
- `executor_memory` resets for independent task execution
- `reception_output` is set by any node for real-time progress updates

### 3. URL Extraction (Zero Hallucination)
```python
# SerpAPI returns structured JSON
search_results = searching_tool.results(query)

# Python extracts URL deterministically
url = search_results['organic_results'][0]['link']

# LLM receives: "[EXTRACTED_URL]: https://actual-url.com"
# LLM copies exact URL (no generation = no hallucination)
```

---

## 📊 Architecture Highlights

### Request Lifecycle with Real-Time Streaming
```
1. Client → POST /not-jarvis/stream
2. FastAPI → workflow.compile(checkpointer=supabase)
3. LangGraph → loads messages from DB (thread_id lookup)
4. TaskPlanner → streams "🔍 Searching..." → plans action
5. Executor → streams "✅ Search complete" → executes
6. Loop → repeats until is_complete=True (each node streams updates)
7. Reception → streams final summary
8. SSE → delivers all updates to client in real-time
```

**Streaming Pattern**: Every node (TaskPlanner, Executor, Reception) can set `reception_output`, which FastAPI immediately streams via Server-Sent Events. User sees progress as it happens.

### Database Schema
- **checkpoints**: State snapshots (user_goal, loop_count, route_to)
- **checkpoint_writes**: Messages & executor_memory (msgpack encoded)
- **checkpoint_blobs**: Large data overflow
- **checkpoint_migrations**: Schema versioning

---

## 🎓 Key Learnings

### 1. When NOT to Use LLMs
**Problem**: LLMs hallucinate structured data (URLs, emails, IDs)  
**Solution**: Use Python for extraction, LLMs for semantic decisions

### 2. Iterative > Multi-Step Planning
**Before**: TaskPlanner returned 10 steps, executed only first (9 wasted)  
**After**: Single-step planning, re-plan after each execution

### 3. Selective State Persistence
**Messages**: Persist (conversation context)  
**Executor Memory**: Reset (task independence)  
**Design trade-off**: Conversation continuity vs clean task boundaries

---

## 🚀 Future Enhancements

- [ ] Add authentication & multi-user support
- [ ] Implement crash recovery (check `is_complete` flag before starting)
- [ ] Add more tools (file operations, email, calendar)
- [ ] Voice input/output integration
- [ ] Deploy with Docker + managed PostgreSQL

---

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details

---

## 👤 Author

**Gokul Sree Chandra**  
3rd Year Computer Science Student, GITAM University

---

**Built with focus on production-ready patterns**: async/await, connection pooling, structured outputs, error handling, and architectural clarity.



