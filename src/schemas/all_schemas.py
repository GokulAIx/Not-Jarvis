from pydantic import BaseModel
from typing import List, Optional


class Planner_Steps(BaseModel):
    action: str
    app_name: Optional[str] = None
    command: Optional[str] = None
    url: Optional[str] = None

class Planner(BaseModel):
    Steps: list[Planner_Steps]

