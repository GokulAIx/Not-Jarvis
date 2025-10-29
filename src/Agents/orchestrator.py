from langgraph.graph import StateGraph , START , END
from src.Agents.task_planner import TaskPlanner
from src.Agents.executor import Executor
from src.Agents.State import State


graph=StateGraph(State)

graph.add_node("TaskPlanner",TaskPlanner)
graph.add_node("Executor", Executor().what_execute)

graph.add_edge(START,"TaskPlanner")
graph.add_edge("TaskPlanner","Executor")
graph.add_edge("Executor",END)

workflow=graph.compile()


