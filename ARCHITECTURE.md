# Not-Jarvis Architecture Documentation

**Version**: 2.1  
**Last Updated**: February 6, 2026  
**Status**: Production-Ready

---

## Overview

Not-Jarvis is a text-based AI agent that executes system tasks through natural language commands. Built on iterative single-step planning, it adapts to search results dynamically rather than following rigid pre-planned sequences.

**Core Capabilities**:
- Web search with zero-hallucination URL extraction
- Browser control and website opening
- Application launching
- Screenshot capture
- Multi-turn conversation memory
- Real-time streaming responses

---

## System Architecture

### Three-Layer Design

**1. Client Layer** ([client.py](client.py))
- Terminal-based interface for user interaction
- Maintains persistent session thread ID (HAEDCODED HERE)
- Consumes Server-Sent Events for real-time streaming

**2. API Layer** ([main.py](main.py))
- FastAPI server with async PostgreSQL connection pooling
- Manages conversation state via Supabase checkpointer
- Resets per-request state (loop_count, executor_memory)
- Preserves conversation messages across requests
- Streams updates via Server-Sent Events

**3. Orchestration Layer** ([src/Agents/agent.py](src/Agents/agent.py))
- LangGraph StateGraph with iterative workflow
- Three nodes: TaskPlanner → Executor → Reception
- TaskPlanner analyzes context and plans next single action
- Executor runs system commands and tools
- Reception formats final natural language response

---

## Key Innovation: Iterative Single-Step Planning

**Traditional Approach**: Generate complete task list upfront  
**Problem**: Rigid plans fail when reality differs from expectation

**Our Approach**: Plan one action → Execute → Analyze results → Plan next action  
**Benefit**: Adapts to actual outcomes, handles unexpected responses gracefully

**Example Flow**:
1. User: "Find MIT website and open it"
2. Loop 1: TaskPlanner plans search → Executor searches MIT → Returns to TaskPlanner
3. Loop 2: TaskPlanner sees results with 4 URLs → Selects index 0 → Executor opens browser
4. Loop 3: TaskPlanner confirms completion → Routes to Reception → Formats response

---

## Zero-Hallucination URL Handling

**Problem**: LLMs hallucinate URLs when extracting from JSON

**Solution**: Python extracts URLs, LLM selects by index

### URL_MAP Implementation

**Step 1 - Search** ([src/tools/tools.py](src/tools/tools.py))
- Python parses SerpAPI JSON response
- Extracts top 4 URLs deterministically
- Creates indexed map: {0: url1, 1: url2, 2: url3, 3: url4}
- Returns formatted text + [URL_MAP] block

**Step 2 - Selection** ([src/schemas/all_schemas.py](src/schemas/all_schemas.py))
- LLM receives URL_MAP with indices [0], [1], [2], [3]
- Returns `url_index: 0` (just a number, never copies URL)
- Pydantic schema enforces url_index as Optional[int]

**Step 3 - Resolution** ([src/Agents/executor.py](src/Agents/executor.py))
- Executor reads url_index from task
- Looks up index in state['url_map'] dictionary
- Injects resolved URL into task
- Opens browser with actual URL

**Result**: Zero URL hallucination - Python handles all data extraction

---

## State Management

### State Schema

**Per-Request Fields** (reset each request):
- `loop_count`: Iteration counter
- `executor_memory`: List of completed actions
- `pending_task`: Current single task to execute
- `is_complete`: Completion flag
- `route_to`: Routing decision
- `reception_output`: Streaming messages to user

**Persistent Fields** (preserved via checkpointer):
- `messages`: Full conversation history
- `last_opened_url`: Normalized URL for duplicate prevention
- `url_map`: Current search result URLs

**Why Split State**:
- Per-request fields prevent cross-request pollution
- Persistent fields enable multi-turn awareness
- User can ask "open their website" after "find restaurant"

---

## TaskPlanner Rules

**Rule 0**: Detect conversational queries (greetings, "who are you") → respond directly without actions

**Rule 1**: Check completion FIRST - compare user goal vs executor_memory to avoid redundant actions

**Rule 2**: If memory empty and action needed → plan search action

**Rule 3**: If search results available → analyze URL_MAP:
- Official site found → open using url_index
- Aggregator site (ranking, topuniversities) AND search_count < 2 → trigger follow-up search
- search_count >= 2 → use best available URL or explain limitation

**Rule 4**: If loop_count >= 4 without opening → force action to prevent infinite loops

**Rule 5**: Always re-execute repeated requests - never say "I already did this"

---

## Intelligent Follow-Up Search

**Problem**: First search often returns ranking sites (topuniversities.com) instead of official sources

**Solution**: Two-search maximum with aggregator detection

**Flow**:
1. User: "Find best engineering university"
2. Search 1: Returns topuniversities.com ranking page
3. TaskPlanner detects "topuniversities" in URL
4. Extracts "Massachusetts Institute of Technology" from results
5. Search 2: "Massachusetts Institute of Technology official site"
6. Returns mit.edu → Opens official site

**Hard Limit**: Maximum 2 searches enforced in prompt and search_count tracking

---

## Duplicate Prevention

**Problem**: LLM sometimes plans same open_website action twice

**Solution**: URL normalization and state tracking

**Implementation**:
1. Executor stores last_opened_url in normalized form (adds https:// prefix)
2. Before opening, normalizes candidate URL
3. Compares normalized candidate with last_opened_url
4. If match: Skip execution, return "Already opened"
5. If different: Execute and update last_opened_url

**Example**: "open mit.edu" normalizes to "https://mit.edu". Later request "open https://mit.edu" also normalizes to "https://mit.edu" → Duplicate detected → Skipped.

---

## Real-Time Streaming

**TaskPlanner Streaming**: Emits start messages
- "🔍 Searching for: {query}..."
- "🌐 Opening website..."
- "🚀 Launching {app_name}..."

**Executor Streaming**: Emits completion messages
- "✅ Search complete"
- "✅ Website opened"
- "✅ Application launched"

**Implementation**: Both nodes set `reception_output` field. FastAPI yields values as Server-Sent Events. Client displays updates in real-time.

---

## Technology Stack

- **Language**: Python 3.12
- **API Framework**: FastAPI with async/await
- **Orchestration**: LangGraph StateGraph
- **LLM**: Google Gemini 2.5 Flash with structured outputs
- **Search**: SerpAPI
- **Database**: PostgreSQL (Supabase) with AsyncPostgresSaver
- **State Management**: TypedDict with add_messages reducer
- **Schemas**: Pydantic with Literal types for action validation
- **OS Integration**: webbrowser module, os.system (Windows), subprocess.Popen

---

## Production Features

### Connection Pooling
- AsyncConnectionPool with max 10 connections
- 30-second timeout, 5-minute idle timeout
- Autocommit enabled, dict_row factory

### Process Handoff
- 3-second delay after launching browser/app
- Ensures OS fully detaches process before Python thread terminates
- Prevents browser closing when FastAPI completes response

### Structured Outputs
- Pydantic schemas with Literal types
- Prevents LLM hallucinations ("Open Google" → rejected)
- Forces exact action matches: "open_website", "search", "open_app", "take_screenshot"

### Conversation Persistence
- Hardcoded thread ID enables multi-turn conversations
- Checkpointer loads conversation history from PostgreSQL
- Messages persist across requests, action memory resets

---

## System Limitations

1. **Windows-Only**: Uses Windows `start` command and webbrowser module
2. **No Error Recovery**: Failed actions don't retry automatically
3. **URL_MAP Limited to 4 Results**: Only top 4 search results available
4. **No Observability**: Missing structured logging and metrics
5. **Single-User Design**: Hardcoded session ID, no multi-user support

---

## Architectural Decisions

### Why Python-Based URL Extraction?
LLMs hallucinate URLs when extracting from JSON. Python deterministic parsing eliminates hallucination completely. LLM only makes semantic decisions (which index to select), never generates or copies URLs.

### Why Iterative Single-Step Planning?
Pre-planned task lists fail when search returns unexpected results. Iterative approach adapts dynamically to actual outcomes, recovering from failures by re-planning.

### Why Reset loop_count Per Request?
Without reset, first request completing in 3 loops would cause second request to start at loop_count=3 and hit max iterations immediately.

### Why Hardcoded Thread ID?
UUID per request would prevent multi-turn conversations. Consistent thread ID enables "open their website" to work after "find restaurant" by loading conversation history.

### Why 3-Second Delay After open_website?
FastAPI returns response and terminates Python thread immediately. If OS hasn't fully detached browser process, it gets killed. 3-second delay ensures clean handoff before thread ends.
