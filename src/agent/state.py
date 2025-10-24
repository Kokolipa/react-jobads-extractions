from typing import Annotated, List, Literal, NotRequired, Required

from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

######################################
# <<<< Structured Outputs States >>>>
######################################

class SkillTypes(BaseModel): 
    """Schema for categorizing and storing extracted skills into three distinct types: hard (technical), soft (interpersonal), and both (hybrid)"""
    hard: List[str] = Field(description="A list of demonstrable, quantifiable abilities and technical expertise (e.g., Python, SQL, Cloud Architecture). These are skills acquired through training or education", default_factory=list)
    soft: List[str] = Field(description="A list of personal, non-technical attributes and interpersonal qualities that define work style and interaction (e.g., Teamwork, Communication, Leadership)", default_factory=list)
    both: List[str] = Field(description="A list of skills that can function as either technical or interpersonal depending on context", default_factory=list)
    

class RequirementsState(BaseModel): 
    """State used to extract and list specific, non-negotiable prerequisites or qualifications (e.g., degree level, years of experience) found in a job add"""
    requirements: List[str] = Field(description="A  list of specific, required qualifications, educational background, or minimum experience levels mentioned in a job add.", default_factory=list)


class ResponsibilityState(BaseModel): 
    """State used to extract and list the core duties, tasks, or functions expected of the role being described in the job add"""
    responsibilities: List[str] = Field(description="A list of containing primary day-to-day duties, tasks, and performance expectations for the role", default_factory=list)





############################################
# <<<< ReAct Structured Outputs States >>>>
############################################
class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """
    content: str
    status: Literal["pending", "in_progress", "completed"]


class JobAdContent(TypedDict):
    """A structured output state to extract raw ad_text from a job advertisement.

    Attributes:
        ad_text: The raw text of the job advertisement
    """
    ad_text: str



######################################
# <<<< Main Conversational State >>>>
######################################
class ReActConversationState(AgentState): #* AgentState = Contains remainning_steps & messages with add_messages reducer
    """A comprehensive state object for the ReAct agent, combining conversational history, structured data extraction, and the full Thought-Action-Observation loop memory"""
    user_id: Annotated[Required[str], "Default user ID for all conversational threads"] # "user-dEf@ulT-1"
    skills: SkillTypes
    job_ads: List[JobAdContent]
    todos: NotRequired[List[Todo]] 
    requirements: List[str]
    responsibilities: List[str]