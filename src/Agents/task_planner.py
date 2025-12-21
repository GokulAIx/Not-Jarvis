from src.Agents.State import State
from src.schemas.all_schemas import Planner
from src.tools.url import get_url
import os
from dotenv import load_dotenv
load_dotenv()
api=os.getenv("GOOGLE_API_KEY")

Tools=[get_url]



