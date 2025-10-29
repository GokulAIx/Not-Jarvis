from src.Agents.task_planner import TaskPlanner
from src.Agents.State import State
from src.Agents.orchestrator import workflow


wakeup="arise"
user_goal=input("TYPE SOMETHING ")
if user_goal.lower().startswith(wakeup):
     user_goal=user_goal[len(wakeup):].strip()
else:
    print(f"Please start your goal with the wakeup word ")
    exit(1)


print("STARTING")
result = workflow.invoke({"user_goal": user_goal})

print("\n=== Planner Output ===")
for step in result["planned_tasks"].Steps:
    print(f"→ {step.action}: {step.app_name or step.url}")
print("\n=== Executor Output ===")
for line in result.get("executor_output", []):
    print(line)
