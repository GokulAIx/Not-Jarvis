from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents.executor import Executor
from dotenv import load_dotenv
import os
load_dotenv()
api = os.getenv("GOOGLE_API_KEY")
from tools.tools import get_url,search_tool
from src.schemas.all_schemas import Planner
from prompts import reception_prompt


Tools=[get_url,search_tool]
class State(TypedDict):
    user_goal: str
    planned_tasks: Planner
    completed_tasks: list[str]
    pending_tasks: list[str]
    executor_output: list[str]
    reception_output: str

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools)

plan=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools).with_structured_output(Planner)

#NODES 
def TaskPlanner(state: State):
    response = plan.invoke(
        f"Based on the user's goal: {state['user_goal']}, break it down into a list of steps. "
        "Each step should be a JSON object with an 'action' key and any needed parameters. "
        "For 'open_website' actions, ALWAYS include a valid 'url' parameter. Use the get_url tool if the user does not provide a URL. "
        "Return the full plan as a JSON object with a 'Steps' key containing the list. "
        "EXAMPLE: {\"Steps\": [{\"action\": \"open_app\", \"app_name\": \"CHROME\"}, {\"action\": \"open_website\", \"url\": \"https://youtube.com\"}]}"
    )
    print(response)
    return {
    "planned_tasks": response,
    "pending_tasks": response.Steps
}


def reception(state: State):
    reception_prompt.format(
        planned_tasks=state['planned_tasks']
        get_url=get_url
        search_tool=search_tool
    )
    result=model.invoke(reception_prompt)
    return {"reception_output": result}

#GRAPH BUILDING
graph=StateGraph(State)
graph.add_node("TaskPlanner",TaskPlanner)
graph.add_node("Executor", Executor().what_execute)

graph.add_edge(START,"TaskPlanner")
graph.add_edge("TaskPlanner","Executor")
graph.add_edge("Executor",)

workflow=graph.compile()
