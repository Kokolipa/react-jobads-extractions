from typing import Annotated, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.state import (ReActConversationState, RequirementsState,
                         ResponsibilityState)


@tool(description="Extracts core responsibilities directly from a job advertisement text without paraphrasing or altering phrasing.", parse_docstring=True)
def extract_responsibilities(
    state: Annotated[ReActConversationState, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command: 
    """
    Extracts and stores job responsibilities from a job advertisement.

    This function uses an LLM to identify and extract all primary duties and responsibilities 
    mentioned in the provided job advertisement text. The extraction is performed rigorously and precisely
    preserving the original phrasing, order, and structure without paraphrasing or summarization. 
    Each responsibility is returned as a clean Python list of strings.

    Args:
        state: The injected agent state containing the "job_ads" key with a list of dictionaries which contains the "ad_text"
        tool_call_id: The injected identifier for tracking this specific tool invocation.

    Returns:
        Command: A Command object instructing the ReAct agent to update the state with 
        the extracted list of responsibilities and log a summary message.
    """

    # Define the prompt 
    RESPONSIBILITIES_EXTRACTION_PROMPT: str = """
    ## Task: 
    Extract core responsibilities from the job advertisement below, extract all **primary duties and responsibilities** exactly as they appear or are clearly implied.

    ### Guidelines
    - **Comprehensive:** Include every distinct duty, task, or area of accountability mentioned.
    - **Constrains:** Do not paraphrase, reword, or infer beyond what is stated.
    - **Faithful to Source:** Preserve the original phrasing and order as much as possible.
    - **Specific:** Capture full statements of responsibility, not fragments or general summaries.
    - **Format:** Return the results as a clean, unnumbered python list only — no commentary, headings, or additional text.

    ---

    {job_advertisement}
    """

    # Instantiate the model 
    llm = ChatOllama(
        model="qwen2.5:latest",
        temperature=0,
        verbose=True
    ) 

    # Create LLMChain 
    chain = PromptTemplate.from_template(RESPONSIBILITIES_EXTRACTION_PROMPT) | llm.with_structured_output(ResponsibilityState)

    # Get job_ads raw text
    job_ads = state.get("job_ads", [])
    if not job_ads: 
        return Command(
            update={
                "messages": [
                    ToolMessage(content="No job_ads were found in the state!", tool_call_id=tool_call_id)
                ]
            })
    
    # Extract raw job advertisement text
    advertisements: List[str] =  [raw_text["ad_text"] for raw_text in job_ads]
    responsiblity_dict: Dict[str, List[str]] = {}
    for ad in advertisements: 
        # Invoke the model
        responsibilities = chain.invoke(ad).responsibilities # type: ignore
        responsiblity_dict["responsibilities"] = responsibilities # update the dictionary

    # Update the state 
    return Command(
        update={
            "responsibilities": responsiblity_dict["responsibilities"], 
            "messages": [ToolMessage(content=f"{len(responsibilities)} were extracted from the job advertisment!", tool_call_id=tool_call_id)]
        }
    )


@tool(description="Extracts job requirements, qualifications, and criteria directly from a job advertisement, including skills only if they are explicitly presented as requirements.", parse_docstring=True)
def extract_requirements(
    state: Annotated[ReActConversationState, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command: 
    """
    Extracts and stores job requirements from a job advertisement.

    This function uses an LLM to identify and extract all explicit or clearly implied 
    job requirements mentioned in the provided job advertisement. These may include 
    qualifications, experience, education, certifications, or personal attributes. 
    Skills are included only if they are explicitly stated as requirements within the text.
    
    The extraction is performed verbatim — preserving the original phrasing, order, 
    and structure without paraphrasing, rewording, or summarization. Each requirement 
    is returned as a clean Python list of strings.

    Args:
        state: The injected agent state containing the "job_ads" key with a list of dictionaries which contains the "ad_text"
        tool_call_id: The injected identifier for tracking this specific tool invocation.

    Returns:
        Command: A Command object instructing the ReAct agent to update the state with 
        the extracted list of requirements and log a summary message.
    """

    # Define the prompt 
    REQUIREMENTS_EXTRACTION_PROMPT: str = """
    ## Task: Extract Core Requirements

    From the job advertisement below, extract all **requirements, qualifications, and essential criteria** exactly as they appear or are clearly implied.

    ### Guidelines
    - **Comprehensive:** Include every stated or implied requirement such as experience, education, certifications, or personal attributes.
    - **Skills:** Include a skill only if it is explicitly presented as a requirement (e.g., "must have experience with Python" or "required to manage databases").
    - **Constrains:** Do not paraphrase, reword, or infer beyond what is explicitly written.
    - **Faithful to Source:** Preserve the original phrasing and order as much as possible.
    - **Specific:** Capture full requirement statements, not fragments or summaries.
    - **Format:** Return the results as a clean, unnumbered Python list only — no commentary, headings, or additional text.

    ---

    {job_advertisement}
    """

    # Instantiate the model 
    llm = ChatOllama(
        model="qwen2.5:latest",
        temperature=0,
        verbose=True
    ) 

    # Create LLMChain 
    chain = PromptTemplate.from_template(REQUIREMENTS_EXTRACTION_PROMPT) | llm.with_structured_output(RequirementsState)

    # Get job_ads raw text
    job_ads = state.get("job_ads", [])
    if not job_ads: 
        return Command(
            update={
                "messages": [
                    ToolMessage(content="No job_ads were found in the state!", tool_call_id=tool_call_id)
                ]
            })
    
    # Extract raw job advertisement text
    advertisements: List[str] =  [raw_text["ad_text"] for raw_text in job_ads]
    requirements_dict: Dict[str, List[str]] = {}
    for ad in advertisements: 
        # Invoke the model
        requirements = chain.invoke(ad).requirements # type: ignore
        requirements_dict["requirements"] = requirements # update the dictionary

    # Update the state 
    return Command(
        update={
            "requirements": requirements_dict["requirements"], 
            "messages": [ToolMessage(content=f"{len(requirements)} were extracted from the job advertisment!", tool_call_id=tool_call_id)]
        }
    )
