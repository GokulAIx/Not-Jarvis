from langgraph.graph import StateGraph , START , END
from Agents.task_planner import TaskPlanner
from Agents.executor import Executor
from Agents.State import State
from tools.url import get_url

Tools=[get_url]

graph=StateGraph()

graph.add_node("TaskPlanner",TaskPlanner)
graph.add_node("Executor",Executor)

graph.add_edge(START,"TaskPlanner")
graph.add_edge("TaskPlanner","Executor")
graph.add_edge("Executor",END)

workflow=graph.compile()


