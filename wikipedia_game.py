"""
Wikipedia Navigation Game using GPT-OSS Executor.

The AI must navigate from a starting Wikipedia page to Hitler's page
by following links. Uses the wikipedia-api library for cleaner API access.

Install requirements:
    pip install openai-harmony transformers torch accelerate triton==3.4 kernels
    pip install wikipedia-api pydantic trl datasets wandb tavily-python python-dotenv
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime
import json
import wikipediaapi
import traceback
from io import StringIO
import sys
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import wandb
import torch.nn.functional as F
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
    print("Not in Colab, loading from .env")
    # load_dotenv()

TARGET_PAGE = "Adolf Hitler"
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
    
    def __init__(self, target_page: str = TARGET_PAGE):
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
        headers = {'User-Agent': '2084Collective (lukas@nelsoftware.com)'}
        req = requests.get("https://en.wikipedia.org/api/rest_v1/page/random/summary", headers=headers)
        title = json.loads(req.content)["title"]
        return title

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
    
    def __init__(self, executor, start_page: str, target_page: str = TARGET_PAGE):
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
            "target_reached": self.navigator.current_page == self.target_page,
            "conversation": result.conversation
        }


def compute_grpo_loss(
    logprobs: torch.Tensor,  # [group_size, seq_len] - log probs from model
    ref_logprobs: torch.Tensor,  # [group_size, seq_len] - log probs from reference model
    rewards: torch.Tensor,  # [group_size] - reward for each completion
    clip_ratio: float = 0.2,
    beta: float = 0.0  # KL penalty (default 0.0 as per recent GRPO practices)
) -> dict:
    """
    Compute GRPO loss with group relative advantages.
    
    Args:
        logprobs: Log probabilities from current policy [group_size, seq_len]
        ref_logprobs: Log probabilities from reference policy [group_size, seq_len]
        rewards: Rewards for each completion in the group [group_size]
        clip_ratio: PPO clipping ratio (epsilon)
        beta: KL divergence penalty coefficient
    
    Returns:
        Dictionary with loss and statistics
    """
    # Normalize rewards within the group (Group Relative)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Sum log probs over sequence length to get per-completion log prob
    logprob_sum = logprobs.sum(dim=1)  # [group_size]
    ref_logprob_sum = ref_logprobs.sum(dim=1)  # [group_size]
    
    # Compute probability ratio
    log_ratio = logprob_sum - ref_logprob_sum
    ratio = torch.exp(log_ratio)
    
    # Clipped surrogate objective
    clip_ratio_low = 1.0 - clip_ratio
    clip_ratio_high = 1.0 + clip_ratio
    
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
    loss_policy = torch.max(loss_unclipped, loss_clipped).mean()
    
    # KL divergence (optional, default beta=0.0)
    kl_div = (log_ratio).mean()
    loss_kl = beta * kl_div
    
    total_loss = loss_policy + loss_kl
    
    # Statistics
    clip_frac_low = ((ratio < clip_ratio_low).float().mean()).item()
    clip_frac_high = ((ratio > clip_ratio_high).float().mean()).item()
    
    return {
        "loss": total_loss,
        "loss_policy": loss_policy.item(),
        "loss_kl": loss_kl.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_reward": rewards.mean().item(),
        "mean_ratio": ratio.mean().item(),
        "clip_frac_low": clip_frac_low,
        "clip_frac_high": clip_frac_high,
        "kl_div": kl_div.item()
    }


def extract_logprobs_from_conversation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    encoding,
    conversation: list[dict]
) -> torch.Tensor:
    """
    Extract log probabilities from a conversation by reconstructing it.
    Returns log probs for all assistant tokens (excluding tool outputs which are masked).
    
    IMPORTANT: Tool outputs are NOT trained on - they come from the environment.
    Only assistant reasoning, tool calls, and final answers are trained.
    """
    # Convert conversation dict back to Harmony Conversation
    convo = Conversation.from_dict({"messages": conversation})
    
    # Collect all assistant message tokens with their positions
    all_logprobs = []
    
    # Find assistant messages (skip tool messages)
    for i, msg in enumerate(convo.messages):
        msg_dict = msg.to_dict()
        role = msg_dict.get("role")
        
        # Only train on assistant messages, NOT tool responses
        if role == "assistant":
            # Get conversation up to this point
            context_convo = Conversation.from_messages(list(convo.messages)[:i])
            
            # Render context
            context_ids = encoding.render_conversation_for_completion(context_convo, Role.ASSISTANT)
            
            # Get the assistant message tokens
            assistant_ids = encoding.render_message_for_completion(msg)
            
            # Full input: context + assistant message
            full_ids = context_ids + assistant_ids
            
            # Get model logprobs
            with torch.no_grad():
                input_tensor = torch.tensor([full_ids], dtype=torch.long).to(model.device)
                outputs = model(input_tensor)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                
                # Get log probs for assistant tokens
                # We want logprobs for positions [len(context_ids):len(full_ids)-1]
                # predicting tokens at positions [len(context_ids)+1:len(full_ids)]
                start_idx = len(context_ids)
                end_idx = len(full_ids)
                
                for pos in range(start_idx, end_idx - 1):
                    next_token = full_ids[pos + 1]
                    token_logprobs = F.log_softmax(logits[pos], dim=-1)
                    logprob = token_logprobs[next_token]
                    all_logprobs.append(logprob)
        
        # Skip tool messages - they are environment responses, not model outputs
        elif role == "tool":
            continue
    
    if not all_logprobs:
        # No assistant tokens found, return dummy
        return torch.tensor([0.0], device=model.device)
    
    return torch.stack(all_logprobs)


def train(
    model_name: str = "openai/gpt-oss-20b",
    num_iterations: int = 100,
    num_starting_pages: int = 50,
    group_size: int = 4,  # Number of games to play per starting page
    learning_rate: float = 1.41e-5,
    num_epochs_per_iteration: int = 4,
    max_game_iterations: int = 25,
    wandb_project: str = "wikipedia-game-grpo",
    save_dir: str = "./trained_wikipedia_model",
    clip_ratio: float = 0.2,
    beta: float = 0.0  # KL penalty
):
    """
    Train the model to play the Wikipedia game using GRPO.
    
    Args:
        model_name: HuggingFace model identifier
        num_iterations: Number of training iterations
        num_starting_pages: Number of different starting pages to use
        group_size: Number of games per starting page (for group relative comparison)
        learning_rate: Learning rate for optimizer
        num_epochs_per_iteration: Number of optimization epochs per iteration
        max_game_iterations: Max iterations per game
        wandb_project: W&B project name
        save_dir: Directory to save trained model
        clip_ratio: PPO clipping ratio
        beta: KL divergence penalty coefficient
    """
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "model_name": model_name,
            "num_iterations": num_iterations,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "num_epochs_per_iteration": num_epochs_per_iteration,
            "max_game_iterations": max_game_iterations,
            "clip_ratio": clip_ratio,
            "beta": beta
        }
    )
    
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Create reference model (frozen copy)
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create executor
    executor = GPTOSSExecutor(model_name=model_name)
    executor.model = model
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Generate dataset of starting pages
    print(f"Generating {num_starting_pages} random starting pages...")
    navigator = WikipediaNavigator()
    starting_pages = []
    for _ in range(num_starting_pages):
        page = navigator.get_random_page()
        starting_pages.append(page)
    
    print(f"Starting GRPO training for {num_iterations} iterations...")
    
    # Training loop
    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*80}")
        
        # Sample a starting page
        start_page = starting_pages[iteration % len(starting_pages)]
        
        print(f"Starting page: {start_page}")
        print(f"Playing {group_size} games from this page...")
        
        # Play multiple games from same starting page (generate group of completions)
        group_results = []
        group_rewards = []
        
        for g in range(group_size):
            print(f"\n--- Game {g+1}/{group_size} ---")
            
            game = WikipediaGame(
                executor=executor,
                start_page=start_page,
                target_page=TARGET_PAGE
            )
            
            result = game.play(max_iterations=max_game_iterations)
            
            # Calculate reward
            if result.get("error"):
                reward = -10.0
            elif result["target_reached"]:
                steps = result["steps"]
                reward = 100.0 - (steps * 2.0)
            else:
                reward = -5.0
            
            group_results.append(result)
            group_rewards.append(reward)
            
            print(f"Result: {'SUCCESS' if result.get('target_reached') else 'FAILED'}")
            print(f"Steps: {result.get('steps', 0)}")
            print(f"Reward: {reward:.2f}")
        
        # Extract log probs from conversations
        print("\nExtracting log probabilities from conversations...")
        group_logprobs = []
        group_ref_logprobs = []
        
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        
        for result in group_results:
            conversation = result.get("conversation", [])
            
            # Get logprobs from current model
            logprobs = extract_logprobs_from_conversation(
                model, tokenizer, encoding, conversation
            )
            group_logprobs.append(logprobs)
            
            # Get logprobs from reference model
            ref_logprobs = extract_logprobs_from_conversation(
                ref_model, tokenizer, encoding, conversation
            )
            group_ref_logprobs.append(ref_logprobs)
        
        # Pad sequences to same length
        max_len = max(lp.shape[0] for lp in group_logprobs)
        padded_logprobs = []
        padded_ref_logprobs = []
        
        for lp, ref_lp in zip(group_logprobs, group_ref_logprobs):
            if lp.shape[0] < max_len:
                padding = torch.zeros(max_len - lp.shape[0], device=lp.device)
                lp = torch.cat([lp, padding])
                ref_lp = torch.cat([ref_lp, padding])
            padded_logprobs.append(lp)
            padded_ref_logprobs.append(ref_lp)
        
        logprobs_tensor = torch.stack(padded_logprobs)  # [group_size, max_len]
        ref_logprobs_tensor = torch.stack(padded_ref_logprobs)  # [group_size, max_len]
        rewards_tensor = torch.tensor(group_rewards, device=model.device)  # [group_size]
        
        # Multiple epochs over same data
        print(f"\nOptimizing for {num_epochs_per_iteration} epochs...")
        for epoch in range(num_epochs_per_iteration):
            optimizer.zero_grad()
            
            # Compute GRPO loss
            loss_dict = compute_grpo_loss(
                logprobs_tensor,
                ref_logprobs_tensor,
                rewards_tensor,
                clip_ratio=clip_ratio,
                beta=beta
            )
            
            loss = loss_dict["loss"]
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs_per_iteration}: Loss = {loss.item():.4f}")
        
        # Log to wandb
        wandb.log({
            "iteration": iteration + 1,
            "start_page": start_page,
            "mean_reward": loss_dict["mean_reward"],
            "loss": loss_dict["loss"],
            "loss_policy": loss_dict["loss_policy"],
            "loss_kl": loss_dict["loss_kl"],
            "mean_advantage": loss_dict["mean_advantage"],
            "mean_ratio": loss_dict["mean_ratio"],
            "clip_frac_low": loss_dict["clip_frac_low"],
            "clip_frac_high": loss_dict["clip_frac_high"],
            "kl_div": loss_dict["kl_div"],
            "successes": sum(1 for r in group_results if r.get("target_reached", False)),
            "mean_steps": sum(r.get("steps", 0) for r in group_results) / len(group_results)
        })
        
        # Save checkpoint every 10 iterations
        if (iteration + 1) % 10 == 0:
            checkpoint_dir = f"{save_dir}/checkpoint-{iteration + 1}"
            print(f"\nSaving checkpoint to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    
    # Save final model
    print(f"\nSaving final model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    wandb.finish()
    print("Training complete!")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if we should train or just play
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        print("🎓 Starting GRPO Training...")
        train(
            num_iterations=100,
            num_starting_pages=50,
            group_size=4,
            learning_rate=1.41e-5,
            num_epochs_per_iteration=4,
            max_game_iterations=25,
            wandb_project="wikipedia-game-grpo",
            save_dir="./trained_wikipedia_model",
            clip_ratio=0.2,
            beta=0.0
        )
    else:
        # Demo mode - just play one game
        executor = GPTOSSExecutor(model_name="openai/gpt-oss-20b")
        
        # Create and play the game
        game = WikipediaGame(
            executor=executor,
            start_page=WikipediaNavigator().get_random_page(),
            target_page=TARGET_PAGE
        )
        
        print("🎮 Starting Wikipedia Navigation Game!")
        print(f"Target: {TARGET_PAGE}")
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


