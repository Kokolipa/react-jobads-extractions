from typing import Annotated, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.prompts import JOB_ADVERTISEMENT_SECTION_EXTRACTION, WRITE_TODOS_DESCRIPTION
from agent.state import JobAdContent, ReActConversationState, Todo


@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: List[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command: 
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command: Command to update agent state with new TODO list
    """
    #? Returning command directly result an update to the state.
    #? this limits the context of the agent - the LLM doesn't get access this tool at this stage
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ]
        }
    )

@tool(parse_docstring=True)
def read_todos(
    state: Annotated[ReActConversationState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str: 
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list 
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """ 
    # Get todo items from state
    todos = state.get("todos", [])
    if not todos: 
        return "Currently, there aren't any todos in the list."
    
    result = "Current TODO list:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"])
        result = f"{i}. {emoji} {todo['content']}: {todo['status']}\n"
    
    return result.strip() #? This will be inserted into the messages key within the state



@tool(description=JOB_ADVERTISEMENT_SECTION_EXTRACTION, parse_docstring=True)
def update_content(
    job_ads: List[JobAdContent], 
    tool_call_id: Annotated[str, InjectedToolCallId], # insert custom id to the tool during execution
) -> Command: 
    """
    This tool update the agent's 'job_ads' key with the raw context of the job advertisement. 

    Args: 
        job_ads: A list of dictionaries. EACH dictionary MUST contain the key 'ad_text' 
                 with the full text of the job advertisement as its value.
        tool_call_id: Tool call identifier for message response
    
    Returns: 
        Command to update the agent state with the raw job_ads text
    """
    return Command(
        update={
            "job_ads": job_ads,
            "messages": [
                ToolMessage(content=f"The job_ad was successfuly updated!\n{job_ads}", tool_call_id=tool_call_id)
            ]
        }
    )