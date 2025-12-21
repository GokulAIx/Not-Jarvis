**Project Overview**: A small modular multi-agent assistant (Version 1) that uses a planner node (LLM-based) to produce structured plans and an executor node to perform actions locally. Planner uses a tool binding for helper utilities (e.g., URL lookup). The workflow is orchestrated with LangGraph.

**Quick Status (V1)**
- **Planner**: LLM-based planner (`src/Agents/task_planner.py`) that returns structured plans (Pydantic `Planner` model).
- **Executor**: Local executor (`src/Agents/executor.py`) that handles `open_app`, `open_website`, and `take_screenshot` actions.
- **Orchestrator**: LangGraph workflow defined in `src/Agents/orchestrator.py` (START → TaskPlanner → Executor → END).
- **Tools**: Example tool `get_url` in `src/tools/url.py` bound to the planner so the model can resolve site names to URLs during planning.

**Repository Layout**
- `src/` — application sources
	- `src/Agents/` — planner, executor, orchestrator, state definitions
		- `executor.py` — executes platform actions and returns structured results
		- `task_planner.py` — LLM planner that returns structured plans (Pydantic schema)
		- `orchestrator.py` — wires nodes into a LangGraph `workflow`
		- `State.py` — workflow state TypedDict/type (holds `user_goal`, `planned_tasks`, `executor_output`)
	- `src/schemas/` — Pydantic models describing planner output (`all_schemas.py`)
	- `src/tools/` — example tools bound to the planner (e.g., `url.py`)
	- `src/main.py` — simple CLI entrypoint to run the workflow

**Getting Started (local)**
1. Activate the project's virtual environment (example for this repo):
```
.\Jarvis\Scripts\Activate.ps1
pip install --upgrade pip
```

2. Install required packages (adjust for the LLM and tools you use). Example minimal deps:
```
pip install pydantic langchain-google-genai langgraph python-dotenv
```

3. Set up environment variables:
- Create a `.env` in the project root with your Google API key (used by the planner tool & model):
```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the app (simple CLI):
```
python -m src.main
```
Type a command such as `open youtube` when prompted.

**How the workflow runs**
- `src/main.py` prompts the user and invokes the compiled LangGraph `workflow` from `src/Agents/orchestrator.py`.
- `TaskPlanner` (LLM) receives `user_goal` and returns a Pydantic `Planner` object that contains `Steps` (each step has an `action` and optional params like `app_name` or `url`). Tools (like `get_url`) are bound to the planner, so the LLM can call them during planning.
- `Executor` receives the plan and `dispatch_actions` maps each step's `action` to a handler method (`open_app`, `open_website`, `take_screenshot`). The executor filters parameters using function signatures (via `inspect.signature`) so only allowed params are passed. The executor returns a structured `executor_output` list which is stored in the workflow state.

**State schema (V1)**
- The workflow stores state keys (examples):
	- `user_goal` (str) — the user's input goal.
	- `planned_tasks` (Pydantic `Planner`) — structured plan produced by the planner.
	- `executor_output` (list[dict]) — structured list of executed actions and results produced by the executor.

**Example plan & executor output**
- Example `planned_tasks` produced by the planner:
```
{"Steps": [{"action": "open_app", "app_name": "CHROME"}, {"action": "open_website", "url": "https://youtube.com"}]}
```
- Example `executor_output` produced by the executor after running the plan:
```
[
	{"action": "open_app", "params": {"app_name": "CHROME"}, "result": "Opening 'CHROME'..."},
	{"action": "open_website", "params": {"url": "https://youtube.com"}, "result": "Opening website: https://youtube.com"}
]
```

**Security & privacy notes (V1)**
- By default the planner uses the Google Generative API (configure via `GOOGLE_API_KEY`). Be mindful that sending prompts or transcripts to a cloud LLM or external STT/TTS provider may expose user content. For production, add privacy policies, opt-ins, and local-only modes.

**Development notes & where to extend**
- `src/Agents/executor.py`: Add new handlers for additional actions (update `action_handlers` mapping). Keep the signature-based parameter filtering to avoid TypeErrors.
- `src/Agents/task_planner.py`: The planner prompt and Pydantic schema determine what the LLM will output. If the model returns steps without needed params, update the prompt to require them or use tools to fill missing values.
- Add tests under a `tests/` folder (unit tests for dispatcher, integration tests for end-to-end flows). Consider mocking the LLM for CI.

**Planned V2 features (roadmap highlights)**
- Voice input (STT) and text-to-speech (TTS)
- Short-term memory manager (in-memory + optional persistence)
- Context-aware planning (planner uses memory)
- Policy/confirmation layer for risky actions

**Troubleshooting**
- If the executor raises `TypeError` about missing parameters, check the planner output structure against the `Planner` schema in `src/schemas/all_schemas.py` and make sure steps include required parameters (e.g., `url` for `open_website`).
- If the model doesn't call bound tools as expected, ensure tools are bound in `task_planner.py` and that the planner prompt instructs the model to use tools where appropriate.

**Contact & contribution**
- If you want me to help implement V2 or add tests, tell me which task to take next and I will provide step-by-step guidance.

---

Last updated: 2025-11-29

