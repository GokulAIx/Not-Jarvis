from langchain_core.tools import Tool
import os
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
load_dotenv()

serpy = os.getenv("SERPAPI_API_KEY")

# Initialize SerpAPI wrapper
searching_tool = SerpAPIWrapper(serpapi_api_key=serpy)

def enhanced_search(query: str) -> str:
    """Search and automatically extract URL from first result"""
    # Get the raw JSON dict from SerpAPI (not formatted text)
    results_dict = searching_tool.results(query)
    
    try:
        # Check if there are organic results with links
        if 'organic_results' in results_dict and len(results_dict['organic_results']) > 0:
            first_result = results_dict['organic_results'][0]
            url = first_result.get('link', '')
            
            if url:  # Only add tag if URL exists
                # Convert dict to formatted text + add URL tag
                raw_text = searching_tool.run(query)  # Get formatted text version
                response = f"{raw_text}\n\n[EXTRACTED_URL]: {url}"
                return response
        
        # If no organic results (e.g., direct answer), return formatted text
        # This happens for "what is" queries where Google shows answer box
        return searching_tool.run(query)
    
    except (KeyError, IndexError, Exception) as e:
        # If parsing fails, return formatted text
        print(f"⚠️ URL extraction error: {e}")
        return searching_tool.run(query)

search_tool = Tool(
    name="serp_search",
    func=enhanced_search,  # Use enhanced version
    description="Search the web and automatically extract the top result's URL"
)