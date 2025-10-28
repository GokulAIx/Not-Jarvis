from typing import TypedDict

class State(TypedDict):
    user_goal: str
    planned_tasks: list[str]
    completed_tasks: list[str]
    pending_tasks: list[str]
    
    