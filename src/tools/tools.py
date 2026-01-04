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

# Initialize SerpAPI tool FIRST (before get_url uses it)
searching_tool = SerpAPIWrapper(serpapi_api_key=serpy)

@tool
def get_url(site_name: str) -> str:
    """Find the official website URL for a given business/restaurant name using web search"""
    try:
        if not site_name or site_name.strip() == "":
            return "Error: No site name provided. Please provide a restaurant or business name."
        
        # Search for the official website
        search_query = f"{site_name} official website"
        search_results = searching_tool.run(search_query)
        
        # Use LLM to extract the URL from search results
        prompt = f"""
        Search results for "{site_name}":
        {search_results}
        
        Extract ONLY the official website URL (must start with http:// or https://). 
        Return JUST the URL, nothing else.
        
        If no official website is found in the results, return exactly: "NO_URL_FOUND"
        Do not return explanations, just the URL or NO_URL_FOUND.
        """
        
        response = model.invoke(prompt).content.strip()
        
        # Check if no URL was found
        if "NO_URL_FOUND" in response or not response:
            return f"Error: Could not find official website for {site_name}. Try a direct search instead."
        
        # Clean up the response (remove quotes, extra text, trailing punctuation)
        import re
        # Match URL but stop before trailing punctuation like ), ., etc
        url_match = re.search(r'https?://[^\s<>"()]+(?:[^\s<>"().,;!?)])?', response)
        if url_match:
            url = url_match.group(0)
            # Remove any trailing dots, ellipsis, or closing parens
            url = re.sub(r'[.)]+$', '', url)
            return f"URL: {url}"
        
        return f"Error: Found text but no valid URL: {response[:100]}"
        
    except Exception as e:
        return f"Error: Failed to find URL - {str(e)}"

search_tool = Tool(
    name="serp_search",  # CHANGE: Removed space, used underscore
    func=searching_tool.run, 
    description="Use this tool to search for public information on the web."
)