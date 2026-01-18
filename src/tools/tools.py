from langchain_core.tools import Tool
import os
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
load_dotenv()

serpy = os.getenv("SERPAPI_API_KEY")

# Initialize SerpAPI wrapper
searching_tool = SerpAPIWrapper(serpapi_api_key=serpy)

def enhanced_search(query: str) -> str:
    """Search and build a URL map for top 4 results, formatted for LLM index selection."""
    results_dict = searching_tool.results(query)
    url_map = {}
    try:
        organic = results_dict.get('organic_results', [])
        # Build map for up to 4 results
        for idx in range(min(4, len(organic))):
            url = organic[idx].get('link', '')
            if url:
                url_map[idx] = url
        # Format map for LLM
        raw_text = searching_tool.run(query)
        map_str = '\n'.join([f"[{i}]: {u}" for i, u in url_map.items()])
        response = f"{raw_text}\n\n[URL_MAP]: {{\n{map_str}\n}}"
        return response
    except Exception as e:
        print(f"⚠️ URL extraction error: {e}")
        return searching_tool.run(query)

search_tool = Tool(
    name="serp_search",
    func=enhanced_search,  # Use enhanced version
    description="Search the web and automatically extract the top result's URL"
)