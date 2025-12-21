from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.tools import Tool
import os
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
load_dotenv()

api = os.getenv("GOOGLE_API_KEY")
serpy = os.getenv("SERPAPI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api)

@tool
def get_url(site_name: str) -> str:
    """this tool is used a generate the url of a website given its name. USE WHEN THE USER WANTS TO OPEN A WEBSITE BUT DOES NOT KNOW THE URL"""
    prompt = f"Generate the homepage URL for {site_name}."
    response = model.invoke(prompt).content
    return response.strip()


searching_tool=SerpAPIWrapper(serpapi_api_key=serpy)
search_tool = Tool(
    name="SerpAPI Search",
    func=searching_tool.run,  # callable method
    description="Use this tool to search for public information on the web."
)