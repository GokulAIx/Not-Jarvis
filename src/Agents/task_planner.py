from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents.State import State
from src.schemas.all_schemas import Planner
from src.tools.url import get_url
import os
from dotenv import load_dotenv
load_dotenv()
api=os.getenv("GOOGLE_API_KEY")

Tools=[get_url]
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api).bind_tools(Tools).with_structured_output(Planner)



def TaskPlanner(state: State):
    response = model.invoke(
        f"Based on the user's goal: {state['user_goal']}, break it down into a list of steps. "
        "Each step should be a JSON object with an 'action' key and any needed parameters. "
        "For 'open_website' actions, ALWAYS include a valid 'url' parameter. Use the get_url tool if the user does not provide a URL. "
        "Return the full plan as a JSON object with a 'Steps' key containing the list. "
        "EXAMPLE: {\"Steps\": [{\"action\": \"open_app\", \"app_name\": \"CHROME\"}, {\"action\": \"open_website\", \"url\": \"https://youtube.com\"}]}"
    )
    print(response)
    return {"planned_tasks": response}
