from Agents.task_planner import TaskPlanner
from Agents.State import State


user=input("Hey, what do you want to do today? ")
State["user_goal"]=user

planner = TaskPlanner(user)