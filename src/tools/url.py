from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

api = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api)

@tool
def get_url(site_name: str) -> str:
    """this tool is used a generate the url of a website given its name. USE WHEN THE USER WANTS TO OPEN A WEBSITE BUT DOES NOT KNOW THE URL"""
    prompt = f"Generate the homepage URL for {site_name}."
    response = model.invoke(prompt).content
    return response.strip()