from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Planner_Steps(BaseModel):
    action: Literal["open_website", "open_app", "take_screenshot", "search"] = Field(
        description="The action to perform."
    )
    app_name: Optional[str] = Field(None, description="Required for open_app")
    command: Optional[str] = None
    url_index: Optional[int] = Field(None, description="If provided, refers to an index in the URL map returned by the search tool")
    query: Optional[str] = Field(None, description="Required for search action")

class Planner(BaseModel):
    Steps: List[Planner_Steps]
    route_to: str = Field(description="Set to 'executor' for system actions or 'terminal'.")
    direct_response: Optional[str] = Field(description="Response for terminal route.")

class ReceptionResponse(BaseModel):
    answer: str = Field(description="Final response.")