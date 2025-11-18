"""
Wikipedia Navigation Game using GPT-OSS Executor.

The AI must navigate from a starting Wikipedia page to Hitler's page
by following links. Uses the wikipedia-api library for cleaner API access.

Install requirements:
    pip install openai-harmony transformers torch accelerate triton==3.4 kernels
    pip install wikipedia-api pydantic
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime
import json
import re
import wikipediaapi
from typing import Any
import traceback
from io import StringIO
import sys
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime, timedelta
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import wandb
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort,
    Content,
)
try:
    from google.colab import userdata
    os.environ['TAVILY_API_KEY'] = userdata.get('TAVILY_API_KEY')
except ImportError:
    # Not in Colab, load from .env
    load_dotenv()

class QuestionToAnswerT(BaseModel):
    question: str
    description: str
    outcomes: list[str]
    winning_outcome: str
    time_of_resolution: datetime


class PredictionResult(BaseModel):
    question: QuestionToAnswerT
    predicted_outcome: str | None
    is_correct: bool
    conversation: list[dict]
    reasoning: str


class ExecutorResult(BaseModel):
    """Generic result from GPTOSSExecutor."""
    final_content: str | None
    conversation: list[dict]
    full_reasoning: list[str]


class BaseTool(ABC):
    """Abstract base class for tools that can be used by the executor."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """Return the JSON schema for the tool's parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with the given arguments and return results as JSON string."""
        pass
    
    def to_harmony_tool_description(self) -> ToolDescription:
        """Convert this tool to a Harmony ToolDescription."""
        return ToolDescription.new(
            self.name,
            self.description,
            parameters=self.parameters_schema
        )


class SearchTool(BaseTool):
    """Web search tool with date cutoff support."""
    
    def __init__(self, cutoff_date: datetime):
        self.cutoff_date = cutoff_date
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set.")
        self.tavily = TavilyClient(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "search_the_web"
    
    @property
    def description(self) -> str:
        return (
            "Search the web for information relevant to the question. "
            "Results are restricted to content published before the resolution date."
        )
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str) -> str:
        """Execute a web search and return results as JSON."""
        try:
            end_date_str = self.cutoff_date.strftime("%Y-%m-%d")
            response = self.tavily.search(query=query, end_date=end_date_str, topic="news")
            results = [
                {"title": result['title'], "content": result['content']}
                for result in response.get('results', [])[:5]  # Limit to 5 results
            ]
            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


class GPTOSSExecutor:
    """
    Core GPT-OSS executor that handles model interaction.
    Takes user message, system prompt, and tools - returns raw results.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize the executor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def _build_conversation(
        self,
        user_message: str,
        system_prompt: str,
        tools: list[BaseTool],
        cutoff_date: datetime | None = None
    ) -> Conversation:
        """Build initial conversation using Harmony library."""
        # System message with optional cutoff date
        system_msg = SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
        if cutoff_date:
            system_msg = system_msg.with_conversation_start_date(cutoff_date.strftime("%Y-%m-%d"))

        # Developer message with instructions and tools
        tool_descriptions = [tool.to_harmony_tool_description() for tool in tools]
        developer_msg = (
            DeveloperContent.new()
            .with_instructions(system_prompt)
            .with_function_tools(tool_descriptions)
        )

        return Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_msg),
            Message.from_role_and_content(Role.DEVELOPER, developer_msg),
            Message.from_role_and_content(Role.USER, user_message),
        ])

    def _execute_tool(self, name: str, arguments: dict, tools: list[BaseTool]) -> str:
        """Execute the specified tool by name."""
        for tool in tools:
            if tool.name == name:
                try:
                    return tool.execute(**arguments)
                except Exception as e:
                    return json.dumps({"error": f"Tool execution failed: {str(e)}"})
        
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _conversation_to_dicts(self, convo: Conversation) -> list[dict]:
        """Convert Harmony Conversation to list of dicts."""
        return convo.to_dict()['messages']

    def render_content_to_text(self, content: list[Content]) -> str:
        """Render content list to plain text."""
        return "".join(cont.text for cont in content)

    def execute(
        self,
        user_message: str,
        system_prompt: str,
        tools: list[BaseTool],
        cutoff_date: datetime | None = None,
        max_tool_calls: int = 10
    ) -> ExecutorResult:
        """
        Execute GPT-OSS model with given configuration.

        Args:
            user_message: The user's query/prompt
            system_prompt: Instructions for the model
            tools: List of available tools
            cutoff_date: Optional date for system message
            max_tool_calls: Maximum number of tool call iterations

        Returns:
            ExecutorResult with final_content, conversation, and full_reasoning
        """
        convo = self._build_conversation(user_message, system_prompt, tools, cutoff_date)
        full_reasoning = []

        iteration = 0

        while iteration < max_tool_calls:
            # Render conversation to token IDs using Harmony encoding
            input_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            stop_tokens = self.encoding.stop_tokens_for_assistant_actions()

            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.model.device)

            # Generate response
            outputs = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                eos_token_id=stop_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Extract newly generated tokens
            new_tokens = outputs[0][len(input_ids):].tolist()

            # Remove stop token if present
            if new_tokens and new_tokens[-1] in stop_tokens:
                new_tokens = new_tokens[:-1]

            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Generated {len(new_tokens)} tokens")

            # Parse using Harmony encoding
            parsed_messages = self.encoding.parse_messages_from_completion_tokens(
                new_tokens,
                Role.ASSISTANT
            )

            print(f"Parsed {len(parsed_messages)} messages")

            has_tool_call = False
            final_content = None

            for msg in parsed_messages:
                msg_dict = msg.to_dict()
                channel = msg_dict.get("channel", "")
                content = msg_dict.get("content", "")
                recipient = msg_dict.get("recipient", "")
                print(msg)
                print(f"Channel: {channel}, Recipient: {recipient}")
                if len(content) > 200:
                    print(f"Content: {content[:200]}...")
                else:
                    print(f"Content: {content}")

                # Collect reasoning from analysis channel
                if channel == "analysis":
                    full_reasoning.append(f"[Analysis] {content}")

                # Check for tool call
                if recipient and str(recipient).startswith("functions."):
                    has_tool_call = True
                    func_name = str(recipient).replace("functions.", "")

                    try:
                        arguments = json.loads(self.render_content_to_text(msg.content))
                    except json.JSONDecodeError:
                        arguments = {}

                    print(f"Tool call: {func_name}({arguments})")

                    # Execute tool
                    tool_result = self._execute_tool(func_name, arguments, tools)
                    print(f"Tool result: {tool_result[:300]}...")

                    # Add assistant's tool call message
                    convo = Conversation.from_messages(
                        list(convo.messages) + [msg]
                    )

                    # Add tool response
                    convo = Conversation.from_messages(
                        list(convo.messages) + [
                            Message.from_author_and_content(
                                Author.new(Role.TOOL, f"functions.{func_name}"),
                                tool_result
                            ).with_channel("commentary")
                        ]
                    )

                elif channel == "final":
                    # This is the final answer
                    final_content = self.render_content_to_text(msg.content)
                    full_reasoning.append(f"[Final] {content}")

                    # Add final message to conversation
                    convo = Conversation.from_messages(
                        list(convo.messages) + [msg]
                    )
                else:
                    # Other message (commentary without tool call, etc.)
                    if content and channel:
                        convo = Conversation.from_messages(
                            list(convo.messages) + [msg]
                        )

            if not has_tool_call:
                # No more tool calls, return results
                return ExecutorResult(
                    final_content=final_content,
                    conversation=self._conversation_to_dicts(convo),
                    full_reasoning=full_reasoning
                )

            iteration += 1

        # Max iterations reached
        return ExecutorResult(
            final_content=None,
            conversation=self._conversation_to_dicts(convo),
            full_reasoning=full_reasoning + ["[Max tool calls reached]"]
        )



class WikipediaNavigator:
    """Handles Wikipedia navigation and link extraction using wikipedia-api."""
    
    def __init__(self, target_page: str = "Adolf_Hitler"):
        self.target_page = target_page
        self.current_page = None
        self.history = []
        self.max_steps = 20
        
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='WikipediaGame/1.0 (Educational Project)',
            language='en'
        )
  
    
    def get_random_page(self) -> str:
        """Get a random Wikipedia page title."""
        import requests
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": "0",  # Main namespace (articles only)
            "rnlimit": "1"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            random_page = data["query"]["random"][0]["title"]
            print(f"Random page selected: {random_page}")
            return random_page
            
        except Exception as e:
            print(f"Error getting random page: {e}")
            # Fallback to a default page
            return "Philosophy"

    def get_page_links(self, page_title: str) -> list[str]:
        """Get all valid Wikipedia links from a page."""
        try:
            page = self.wiki.page(page_title)
            
            if not page.exists():
                print(f"Page does not exist: {page_title}")
                return []
            
            # Get links as a list of titles
            links = list(page.links.keys())
            
            # Filter out special pages (those with colons, which are usually namespace prefixes)
            filtered_links = [
                link for link in links 
                if ':' not in link and link.strip()
            ]
            
            print(f"Found {len(filtered_links)} links on {page_title}")
            return filtered_links[:100]  # Limit to first 100 links
            
        except Exception as e:
            print(f"Error fetching links from {page_title}: {e}")
            return []
    
    def navigate(self, page_title: str) -> dict:
        """Navigate to a Wikipedia page and return info about it."""
        if len(self.history) >= self.max_steps:
            return {
                "success": False,
                "error": f"Maximum steps ({self.max_steps}) reached",
                "history": self.history
            }
        
        # Check if page exists
        page = self.wiki.page(page_title)
        if not page.exists():
            return {
                "success": False,
                "error": f"Page does not exist: {page_title}",
                "current_page": self.current_page,
                "history": self.history
            }
        
        # Record the navigation
        self.history.append(page_title)
        self.current_page = page_title
        
        # Check if we've reached the target
        if page_title == self.target_page:
            return {
                "success": True,
                "message": f"SUCCESS! Reached {self.target_page} in {len(self.history)} steps",
                "current_page": page_title,
                "history": self.history,
                "links": [],
                "reached_target": True
            }
        
        # Get links from this page
        links = self.get_page_links(page_title)
        
        if not links:
            return {
                "success": False,
                "error": f"Could not fetch links from page: {page_title}",
                "current_page": page_title,
                "history": self.history
            }
        
        return {
            "success": True,
            "current_page": page_title,
            "links": links,
            "steps_taken": len(self.history),
            "steps_remaining": self.max_steps - len(self.history),
            "history": self.history,
            "reached_target": False
        }
    
    def reset(self, start_page: str):
        """Reset the navigator to a new starting page."""
        self.current_page = start_page
        self.history = []


class CodeExecutorTool(BaseTool):
    """Tool that executes Python code with access to navigate() function."""
    
    def __init__(self, navigator: WikipediaNavigator):
        self.navigator = navigator
        
    @property
    def name(self) -> str:
        return "execute_code"
    
    @property
    def description(self) -> str:
        return (
            "Execute Python code that calls the navigate(page_title) function. "
            "The navigate function takes a Wikipedia page title (from the available links) "
            "and returns information about that page including its links. "
            "Your code should strategically navigate from the current page to the target page."
        )
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can call navigate(page_title) to move to a new page."
                }
            },
            "required": ["code"]
        }
    
    def execute(self, code: str) -> str:
        """Execute Python code with access to the navigate function."""
        # Create a restricted namespace with only navigate available
        namespace = {
            "navigate": self.navigator.navigate,
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "bool": bool,
                "list": list,
                "dict": dict,
                "range": range,
                "enumerate": enumerate,
                "True": True,
                "False": False,
                "None": None,
            }
        }
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        result = {
            "success": False,
            "output": "",
            "error": None,
            "navigation_result": None
        }
        
        try:
            # Execute the code
            exec(code, namespace)
            
            # Check if navigation happened and get the last result
            if "result" in namespace:
                result["navigation_result"] = namespace["result"]
            elif "nav_result" in namespace:
                result["navigation_result"] = namespace["nav_result"]
            
            result["success"] = True
            result["output"] = captured_output.getvalue()
            
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        finally:
            sys.stdout = old_stdout
        
        return json.dumps(result, indent=2)


class WikipediaGame:
    """Main game coordinator using GPT-OSS Executor."""
    
    def __init__(self, executor, start_page: str, target_page: str = "Adolf_Hitler"):
        """
        Initialize the Wikipedia navigation game.
        
        Args:
            executor: GPTOSSExecutor instance
            start_page: Wikipedia page title to start from
            target_page: Target page to reach
        """
        self.executor = executor
        self.start_page = start_page
        self.target_page = target_page
        self.navigator = WikipediaNavigator(target_page=target_page)
        self.code_tool = CodeExecutorTool(self.navigator)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the Wikipedia game."""
        return f"""You are playing the Wikipedia game. Your goal is to navigate from the starting page to the target page ({self.target_page}) by following Wikipedia links.

RULES:
1. You can only navigate by clicking links that exist on the current page
2. You have a maximum of 20 steps to reach the target
3. Use the execute_code tool to call navigate(page_title) with your chosen link
4. Each navigation returns: current_page, available links, steps taken, and whether you've reached the target

STRATEGY:
1. Look at the available links on the current page
2. Choose the link that seems most likely to lead toward {self.target_page}
3. Consider: Historical topics? Geographic connections? Time period? Categories?
4. Think about "hub" pages (like "World War II", "Germany", "Politics") that might connect to many topics
5. If stuck, try broader category pages

IMPORTANT:
- The navigate() function expects the EXACT page title from the links list
- Page titles may use underscores or spaces (both work)
- Don't navigate to the same page twice
- After each navigation, analyze the new links before choosing the next step

Your code should:
1. Call navigate(page_title) with your chosen link
2. Store the result in a variable named 'result'
3. Print your reasoning for why you chose that link

Example code:
```python
# I'm choosing "World War II" because it's a major historical event connected to Hitler
result = navigate("World War II")
print(f"Navigated to: {{result['current_page']}}")
print(f"Reached target: {{result.get('reached_target', False)}}")
```"""
    
    def _build_user_message(self, navigation_result: dict | None = None) -> str:
        """Build user message based on current navigation state."""
        if navigation_result is None:
            # Initial message
            initial_info = self.navigator.navigate(self.start_page)
            
            if not initial_info.get('success'):
                return f"""Failed to load starting page: {self.start_page}
Error: {initial_info.get('error', 'Unknown error')}
Please check that the page title is correct."""
            
            links = initial_info.get('links', [])
            links_preview = links[:20]  # Show first 20 links
            
            return f"""Let's play the Wikipedia game!

STARTING PAGE: {self.start_page}
TARGET PAGE: {self.target_page}
MAXIMUM STEPS: {self.navigator.max_steps}

CURRENT PAGE LINKS (first 20 of {len(links)}):
{chr(10).join(f"  - {link}" for link in links_preview)}

Analyze these links and choose the best one to navigate toward {self.target_page}.
Write Python code that calls navigate() with your chosen page title."""
        
        else:
            # Subsequent message based on navigation result
            if navigation_result.get("reached_target"):
                return f"""🎉 SUCCESS! You reached {self.target_page} in {len(navigation_result['history'])} steps!

Path taken: {' → '.join(navigation_result['history'])}"""
            
            elif not navigation_result.get("success"):
                return f"""Navigation failed: {navigation_result.get('error', 'Unknown error')}

Current progress: {' → '.join(navigation_result.get('history', []))}
Please try a different approach or page."""
            
            else:
                links = navigation_result.get('links', [])
                current = navigation_result['current_page']
                steps_taken = navigation_result['steps_taken']
                steps_remaining = navigation_result['steps_remaining']
                
                links_preview = links[:20]
                
                return f"""CURRENT PAGE: {current}
STEPS TAKEN: {steps_taken} / {self.navigator.max_steps}
STEPS REMAINING: {steps_remaining}

PATH SO FAR: {' → '.join(navigation_result['history'])}

AVAILABLE LINKS (first 20 of {len(links)}):
{chr(10).join(f"  - {link}" for link in links_preview)}

Choose your next link to get closer to {self.target_page}."""
    
    def play(self, max_iterations: int = 25) -> dict:
        """
        Play the game until target is reached or max iterations.
        
        Returns:
            Game result with path and success status
        """
        system_prompt = self._build_system_prompt()
        tools = [self.code_tool]
        
        # Initialize with starting page
        self.navigator.reset(self.start_page)
        
        # Validate starting page
        print(f"Validating starting page: {self.start_page}")
        test_nav = self.navigator.navigate(self.start_page)
        if not test_nav.get('success'):
            print(f"ERROR: Could not load starting page {self.start_page}")
            print(f"Error: {test_nav.get('error', 'Unknown error')}")
            return {
                "success": False,
                "path": [],
                "steps": 0,
                "iterations": 0,
                "target_reached": False,
                "error": f"Could not load starting page: {test_nav.get('error')}"
            }
        
        # Reset after validation (since validation added to history)
        self.navigator.reset(self.start_page)
        user_message = self._build_user_message()
        
        iteration = 0
        game_complete = False
        
        while iteration < max_iterations and not game_complete:
            print(f"\n{'='*60}")
            print(f"Game Iteration {iteration + 1}")
            print(f"{'='*60}")
            
            # Execute with GPT-OSS
            result = self.executor.execute(
                user_message=user_message,
                system_prompt=system_prompt,
                tools=tools,
                max_tool_calls=1  # One navigation per iteration
            )
            
            print(f"Final content: {result.final_content}")
            if result.full_reasoning:
                print(f"Last reasoning: {result.full_reasoning[-1][:200]}...")
            
            # Check if we reached the target
            if self.navigator.current_page == self.target_page:
                game_complete = True
                break
            
            # Check if we hit max steps
            if len(self.navigator.history) >= self.navigator.max_steps:
                print("Maximum steps reached!")
                break
            
            # Get links for current page
            if not self.navigator.current_page:
                print("Warning: No current page set")
                break
                
            current_links = self.navigator.get_page_links(self.navigator.current_page)
            
            if not current_links:
                print(f"Warning: No links found on page {self.navigator.current_page}")
                break
            
            # Build next user message based on current state
            current_state = {
                "success": True,
                "current_page": self.navigator.current_page,
                "links": current_links,
                "steps_taken": len(self.navigator.history),
                "steps_remaining": self.navigator.max_steps - len(self.navigator.history),
                "history": self.navigator.history,
                "reached_target": self.navigator.current_page == self.target_page
            }
            
            user_message = self._build_user_message(current_state)
            iteration += 1
        
        return {
            "success": game_complete,
            "path": self.navigator.history,
            "steps": len(self.navigator.history),
            "iterations": iteration,
            "target_reached": self.navigator.current_page == self.target_page
        }


# Example usage
if __name__ == "__main__":
    executor = GPTOSSExecutor(model_name="openai/gpt-oss-20b")
    
    # Create and play the game
    game = WikipediaGame(
        executor=executor,
        start_page=WikipediaNavigator().get_random_page(),
        target_page="Adolf Hitler"
    )
    
    print("🎮 Starting Wikipedia Navigation Game!")
    print(f"Start: Philosophy")
    print(f"Target: Adolf_Hitler")
    print(f"Max steps: 20\n")
    
    result = game.play(max_iterations=25)
    
    print("\n" + "="*60)
    print("GAME COMPLETE")
    print("="*60)
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps']}")
        print(f"Path: {' → '.join(result['path'])}")
        
        if result['success']:
            print(f"\n🎉 Successfully reached {game.target_page}!")
        else:
            print(f"\n😞 Did not reach target in {game.navigator.max_steps} steps")