"""
Wikipedia Navigation Game using GPT-OSS Executor.

The AI must navigate from a starting Wikipedia page to WW2's page
by following links. Uses the wikipedia-api library for cleaner API access.

Install requirements:
    pip install openai-harmony transformers torch accelerate triton==3.4 kernels
    pip install wikipedia-api pydantic
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime
import json
import traceback
from io import StringIO
import sys
from tavily import TavilyClient
import os
from typing import cast

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Mxfp4Config
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort,
    Content,
)

import wikipediaapi

load_dotenv()


# Custom exception for terminating execution
class TerminateException(Exception):
    """Raised when the navigation reaches the target page."""
    pass


# System prompts as globals
WIKIPEDIA_GAME_SYSTEM_PROMPT_TEMPLATE = """You are playing the Wikipedia game. Your goal is to navigate from the starting page to the target page ({target_page}) by following Wikipedia links.

RULES:
1. You can only navigate by clicking links that exist on the current page
2. You have a maximum of 20 steps to reach the target
3. Use the execute_code tool to call navigate(page_title) with your chosen link
4. Each navigation returns: current_page, available links, steps taken, and whether you've reached the target

STRATEGY:
1. Look at the available links on the current page
2. Choose the link that seems most likely to lead toward {target_page}
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
print(f"Navigated to: {{result.current_page}}")
print(f"Reached target: {{result.reached_target}}")
```"""


# Pydantic Models for return types
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


class NavigationResult(BaseModel):
    """Result from navigating to a Wikipedia page."""
    success: bool
    current_page: str | None = None
    links: list[str] = []
    steps_taken: int = 0
    steps_remaining: int = 0
    history: list[str] = []
    reached_target: bool = False
    error: str | None = None
    message: str | None = None


class CodeExecutionResult(BaseModel):
    """Result from executing Python code."""
    success: bool
    output: str = ""
    error: str | None = None
    navigation_result: dict[str, object] | None = None


class GameResult(BaseModel):
    """Result from playing the Wikipedia game."""
    success: bool
    path: list[str]
    steps: int
    iterations: int
    target_reached: bool
    error: str | None = None


class SuccessResultT(BaseModel):
    """Result from an AbstractGym environment run."""
    reward: float
    conversation: list[dict]
    prompt: str
    terminated: bool
    reached_max_steps: bool

class AbstractGym(ABC):
    """Abstract base class for gym-like environments."""
    
    @abstractmethod
    def run(self, max_iterations: int = 25) -> SuccessResultT:
        """
        Run the environment for up to max_iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run
            
        Returns:
            SuccessResultT containing reward, conversation, prompt, and termination status
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the environment to initial state."""
        pass


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
    def parameters_schema(self) -> dict[str, object]:
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
        self.cutoff_date: datetime = cutoff_date
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set.")
        self.tavily: TavilyClient = TavilyClient(api_key=api_key)

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
    def parameters_schema(self) -> dict[str, object]:
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

        quantization_config = Mxfp4Config(dequantize=True)
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            use_cache=False,
            offload_folder="offload",  # Optional: disk offloading directory
            offload_state_dict=True, 
        )
        self.encoding: HarmonyEncoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        # self.model.generation_config.cache_implementation = "static"

        # self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

    def _build_conversation(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[BaseTool],
        cutoff_date: datetime | None = None
    ) -> Conversation:
        """Build conversation using Harmony library with existing messages."""
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
            *messages
        ])

    def _execute_tool(self, name: str, arguments: dict[str, object], tools: list[BaseTool]) -> str:
        """Execute the specified tool by name. May raise TerminateException."""
        for tool in tools:
            if tool.name == name:
                try:
                    return tool.execute(**arguments)
                except TerminateException:
                    # Re-raise TerminateException to be handled by caller
                    raise
                except Exception as e:
                    return json.dumps({"error": f"Tool execution failed: {str(e)}"})

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _conversation_to_dicts(self, convo: Conversation) -> list[dict[str, object]]:
        """Convert Harmony Conversation to list of dicts."""
        return cast(list[dict[str, object]], convo.to_dict()['messages'])

    def render_content_to_text(self, content: list[Content]) -> str:
        """Render content list to plain text."""
        return "".join(cont.text for cont in content)

    def execute(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[BaseTool],
        cutoff_date: datetime | None = None,
        max_tool_calls: int = 10
    ) -> ExecutorResult:
        """
        Execute GPT-OSS model with given configuration.

        Args:
            messages: List of Message objects (typically starts with user message)
            system_prompt: Instructions for the model
            tools: List of available tools
            cutoff_date: Optional date for system message
            max_tool_calls: Maximum number of tool call iterations

        Returns:
            ExecutorResult with final_content, conversation, and full_reasoning
        """
        convo = self._build_conversation(messages, system_prompt, tools, cutoff_date)
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
                use_cache=True
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

                    # Execute tool - may raise TerminateException
                    try:
                        tool_result = self._execute_tool(func_name, arguments, tools)
                        print(f"Tool result: {tool_result[:300]}...")
                    except TerminateException as e:
                        print(f"Termination triggered: {e}")
                        # Add the final tool call message
                        convo = Conversation.from_messages(
                            list(convo.messages) + [msg]
                        )
                        # Add success message as tool response
                        success_message = json.dumps({"success": True, "message": str(e), "terminated": True})
                        convo = Conversation.from_messages(
                            list(convo.messages) + [
                                Message.from_author_and_content(
                                    Author.new(Role.TOOL, f"functions.{func_name}"),
                                    success_message
                                ).with_channel("commentary")
                            ]
                        )
                        # Return early with success
                        return ExecutorResult(
                            final_content=str(e),
                            conversation=self._conversation_to_dicts(convo),
                            full_reasoning=full_reasoning + [f"[Terminated] {e}"]
                        )

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
        self.target_page: str = target_page
        self.current_page: str | None = None
        self.history: list[str] = []
        self.max_steps: int = 20

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
        headers = {
            "User-Agent": "WikipediaGame/1.0 (Educational Project)"
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
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
            return filtered_links  # Limit to first 100 links

        except Exception as e:
            print(f"Error fetching links from {page_title}: {e}")
            return []

    def navigate(self, page_title: str) -> NavigationResult:
        """Navigate to a Wikipedia page and return info about it."""
        if len(self.history) >= self.max_steps:
            return NavigationResult(
                success=False,
                error=f"Maximum steps ({self.max_steps}) reached",
                history=self.history,
                current_page=self.current_page
            )

        # Check if page exists
        page = self.wiki.page(page_title)
        if not page.exists():
            return NavigationResult(
                success=False,
                error=f"Page does not exist: {page_title}",
                current_page=self.current_page,
                history=self.history
            )

        # Record the navigation
        self.history.append(page_title)
        self.current_page = page_title

        # Check if we've reached the target
        if page_title == self.target_page:
            return NavigationResult(
                success=True,
                message=f"SUCCESS! Reached {self.target_page} in {len(self.history)} steps",
                current_page=page_title,
                history=self.history,
                links=[],
                reached_target=True,
                steps_taken=len(self.history),
                steps_remaining=self.max_steps - len(self.history)
            )

        # Get links from this page
        links = self.get_page_links(page_title)

        if not links:
            return NavigationResult(
                success=False,
                error=f"Could not fetch links from page: {page_title}",
                current_page=page_title,
                history=self.history
            )

        return NavigationResult(
            success=True,
            current_page=page_title,
            links=links,
            steps_taken=len(self.history),
            steps_remaining=self.max_steps - len(self.history),
            history=self.history,
            reached_target=False
        )

    def reset(self, start_page: str):
        """Reset the navigator to a new starting page."""
        self.current_page = start_page
        self.history = []


class CodeExecutorTool(BaseTool):
    """Tool that executes Python code with access to navigate() function."""

    def __init__(self, navigator: WikipediaNavigator):
        self.navigator: WikipediaNavigator = navigator

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
    def parameters_schema(self) -> dict[str, object]:
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

        result = CodeExecutionResult(success=False)

        try:
            # Execute the code
            exec(code, namespace)

            # Check if navigation happened and get the last result
            nav_result = None
            if "result" in namespace:
                nav_result_obj = namespace["result"]
                # Convert NavigationResult to dict for JSON serialization
                if isinstance(nav_result_obj, NavigationResult):
                    nav_result = nav_result_obj.model_dump()
                else:
                    nav_result = nav_result_obj
            elif "nav_result" in namespace:
                nav_result_obj = namespace["nav_result"]
                if isinstance(nav_result_obj, NavigationResult):
                    nav_result = nav_result_obj.model_dump()
                else:
                    nav_result = nav_result_obj

            result = CodeExecutionResult(
                success=True,
                output=captured_output.getvalue(),
                navigation_result=nav_result
            )

            # Check if target was reached by checking navigator state directly
            if self.navigator.current_page == self.navigator.target_page:
                raise TerminateException(f"Target reached: {self.navigator.current_page}")

        except TerminateException:
            # Re-raise to signal completion
            raise
        except Exception as e:
            result = CodeExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )

        finally:
            sys.stdout = old_stdout

        return result.model_dump_json(indent=2)


class WikipediaGame(AbstractGym):
    """Main game coordinator using GPT-OSS Executor."""

    def __init__(self, executor: "GPTOSSExecutor", start_page: str, target_page: str = "Adolf_Hitler"):
        """
        Initialize the Wikipedia navigation game.

        Args:
            executor: GPTOSSExecutor instance
            start_page: Wikipedia page title to start from
            target_page: Target page to reach
        """
        self.executor: GPTOSSExecutor = executor
        self.start_page: str = start_page
        self.target_page: str = target_page
        self.navigator: WikipediaNavigator = WikipediaNavigator(target_page=target_page)
        self.code_tool: CodeExecutorTool = CodeExecutorTool(self.navigator)
        self._last_conversation: list[dict[str, object]] = []

    def _build_system_prompt(self) -> str:
        """Build system prompt for the Wikipedia game."""
        return WIKIPEDIA_GAME_SYSTEM_PROMPT_TEMPLATE.format(target_page=self.target_page)

    def _build_user_message(self, navigation_result: NavigationResult | None = None) -> str:
        """Build user message based on current navigation state."""
        if navigation_result is None:
            # Initial message
            initial_info = self.navigator.navigate(self.start_page)

            if not initial_info.success:
                return f"""Failed to load starting page: {self.start_page}
Error: {initial_info.error or 'Unknown error'}
Please check that the page title is correct."""

            links = initial_info.links
            links_preview = links  # Show first 20 links

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
            if navigation_result.reached_target:
                return f"""🎉 SUCCESS! You reached {self.target_page} in {len(navigation_result.history)} steps!

Path taken: {' → '.join(navigation_result.history)}"""

            elif not navigation_result.success:
                return f"""Navigation failed: {navigation_result.error or 'Unknown error'}

Current progress: {' → '.join(navigation_result.history)}
Please try a different approach or page."""

            else:
                links = navigation_result.links
                current = navigation_result.current_page
                steps_taken = navigation_result.steps_taken
                steps_remaining = navigation_result.steps_remaining

                links_preview = links[:20]

                return f"""CURRENT PAGE: {current}
STEPS TAKEN: {steps_taken} / {self.navigator.max_steps}
STEPS REMAINING: {steps_remaining}

PATH SO FAR: {' → '.join(navigation_result.history)}

AVAILABLE LINKS (first 20 of {len(links)}):
{chr(10).join(f"  - {link}" for link in links_preview)}

Choose your next link to get closer to {self.target_page}."""

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.navigator.reset(self.start_page)
        self._last_conversation = []

    def _calculate_reward(self, terminated: bool, steps: int, max_steps: int) -> float:
        """Calculate reward based on termination and number of steps."""
        if not terminated:
            return 0.0
        # Reward is higher for fewer steps (normalized between 0 and 1)
        # Perfect score (1 step) = 1.0, max steps = 0.5, failed = 0.0
        return 1.0 - (0.5 * (steps / max_steps))

    def run(self, max_iterations: int = 25) -> SuccessResultT:
        """
        Run the environment for up to max_iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run
            
        Returns:
            SuccessResultT containing reward, conversation, prompt, and termination status
        """
        system_prompt = self._build_system_prompt()
        tools: list[BaseTool] = [self.code_tool]

        # Initialize with starting page
        self.reset()

        # Validate starting page
        print(f"Validating starting page: {self.start_page}")
        test_nav = self.navigator.navigate(self.start_page)
        if not test_nav.success:
            print(f"ERROR: Could not load starting page {self.start_page}")
            print(f"Error: {test_nav.error or 'Unknown error'}")
            # Return failed result
            return SuccessResultT(
                reward=0.0,
                conversation=[],
                prompt=system_prompt,
                terminated=False,
                reached_max_steps=False
            )

        # Reset after validation (since validation added to history)
        self.reset()
        user_message_text = self._build_user_message()

        print(f"\n{'='*60}")
        print("Starting Wikipedia Game")
        print(f"{'='*60}")

        # Build messages list starting with initial user message
        messages: list[Message] = [Message.from_role_and_content(Role.USER, user_message_text)]
        
        iteration = 0
        game_complete = False
        result: ExecutorResult | None = None
        
        while iteration < max_iterations and not game_complete:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Execute with current messages
            result = self.executor.execute(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                max_tool_calls=1  # One tool call per iteration
            )
            
            # Check if we reached the target
            if self.navigator.current_page == self.navigator.target_page:
                game_complete = True
                print("Target reached!")
                break
            
            # Check if we hit max steps
            if len(self.navigator.history) >= self.navigator.max_steps:
                print("Maximum steps reached!")
                break
            
            # Extract the assistant's messages from the result
            # We need to get the new messages that were added in this iteration
            full_convo = Conversation.from_json(json.dumps({"messages": result.conversation}))
            # Get messages after the system/developer/previous messages
            # The new assistant and tool messages are at the end
            num_existing = len(messages) + 2  # +2 for system and developer
            new_messages = list(full_convo.messages)[num_existing:]
            
            # Add new messages to our message list
            messages.extend(new_messages)
            
            # Build next user message based on current state
            if self.navigator.current_page:
                current_links = self.navigator.get_page_links(self.navigator.current_page)
                if not current_links:
                    print(f"Warning: No links found on page {self.navigator.current_page}")
                    break
                
                current_state = NavigationResult(
                    success=True,
                    current_page=self.navigator.current_page,
                    links=current_links,
                    steps_taken=len(self.navigator.history),
                    steps_remaining=self.navigator.max_steps - len(self.navigator.history),
                    history=self.navigator.history,
                    reached_target=False
                )
                
                next_user_message = self._build_user_message(current_state)
                messages.append(Message.from_role_and_content(Role.USER, next_user_message))
            
            iteration += 1

        # Get final conversation
        if result is None:
            # No iterations were run
            return SuccessResultT(
                reward=0.0,
                conversation=[],
                prompt=system_prompt,
                terminated=False,
                reached_max_steps=False
            )
        
        final_conversation = result.conversation
        self._last_conversation = final_conversation

        print(f"\nFinal content: {result.final_content}")
        if result.full_reasoning:
            print(f"Last reasoning: {result.full_reasoning[-1][:200]}...")

        # Check termination status
        terminated = self.navigator.current_page == self.target_page
        reached_max_steps = len(self.navigator.history) >= self.navigator.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(terminated, len(self.navigator.history), self.navigator.max_steps)

        print(f"\nGame complete:")
        print(f"  Terminated: {terminated}")
        print(f"  Reached max steps: {reached_max_steps}")
        print(f"  Steps taken: {len(self.navigator.history)}")
        print(f"  Path: {' → '.join(self.navigator.history)}")
        print(f"  Reward: {reward:.3f}")

        # Get complete prompt using openai_harmony
        complete_prompt = ""
        if final_conversation:
            # Reconstruct conversation and render it
            convo_data: dict[str, object] = {"messages": final_conversation}
            convo = Conversation.from_json(json.dumps(convo_data))
            # Use the encoding to render the full prompt
            encoding = self.executor.encoding
            input_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            complete_prompt = str(self.executor.tokenizer.decode(input_ids))

        return SuccessResultT(
            reward=reward,
            conversation=final_conversation,
            prompt=complete_prompt,
            terminated=terminated,
            reached_max_steps=reached_max_steps
        )


def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import wandb
    from peft import LoraConfig, get_peft_model

    wandb.init(
        project="testinggrpoforwikipediagame"
    )
    INNER_LOOP=1
    OUTPUT_LOOP=1


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )

    executor = GPTOSSExecutor(model_name="openai/gpt-oss-20b")
    executor.model = get_peft_model(executor.model, peft_config)
    executor.model.print_trainable_parameters()

    executor.model.train()
    optimizer = optim.Adam(executor.model.parameters(), lr=0.001)
    for _ in range(OUTPUT_LOOP):
        random_page = WikipediaNavigator().get_random_page()
        # Create and play the game
        prompts = []
        for __ in range(INNER_LOOP):
            game = WikipediaGame(
            executor=executor,
            start_page=random_page,
            target_page="World War II"
            )
            result = game.run(max_iterations=25)
            prompts.append((result.prompt, result.reward))
        rewards = torch.tensor([p[1] for p in prompts])
        tokenized_prompts = executor.tokenizer([p[0] for p in prompts], return_tensors="pt").to(executor.model.device)
        logprobs = F.log_softmax(executor.model(**tokenized_prompts).logits)
        mean = rewards.mean()

        std = rewards.std()
        loss = (-(rewards - mean)/std*torch.exp(logprobs - logprobs.detach())).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"loss": loss.item(), "mean_reward": mean, "std":rewards.std()})


