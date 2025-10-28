from langchain_google_genai import ChatGoogleGenerativeAI
from Agents.State import State
import os
from dotenv import load_dotenv
load_dotenv()
from orchestrator import Tools
api=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools)


def TaskPlanner(state: State):
    responce=model.invoke(f"Based on the user's goal: {state['user_goal']}, plan a list of tasks to achieve this goal. Provide the tasks in a numbered list format.USE THE AVAILABLE TOOLS IF NECESSARY").content
    

    return {"planned_tasks": responce}
