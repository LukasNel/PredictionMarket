from pydantic import BaseModel
from datetime import datetime, timedelta
from tavily import TavilyClient
import os

class QuestionToAnswerT(BaseModel):
    question: str
    description: str
    outcomes: list[str]
    winning_outcome: str
    time_of_resolution: datetime


def search_the_web(query: str, only_results_before: datetime) -> list[dict]:
    """
    Search the web using Tavily API and return only pages published before the specified datetime.
    
    Args:
        query: The search query string
        only_results_before: Only return results published before this datetime
        
    Returns:
        List of filtered search results (dictionaries containing title, url, content, etc.)
        
    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set
    """
    # Initialize Tavily client with API key from environment variable
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")
    
    tavily_client = TavilyClient(api_key=api_key)
    
    # Convert datetime to YYYY-MM-DD format for the API
    end_date_str = only_results_before.strftime("%Y-%m-%d")
    
    # Perform the search with end_date parameter to filter results
    response = tavily_client.search(
        query=query,
        end_date=end_date_str
    )
    
    # Return the filtered results (API handles filtering by date)
    return response.get('results', [])

print(search_the_web("When was Tokyo built?", datetime.now() - timedelta(months=7)))
