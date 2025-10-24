from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.prompts import GENERAL_SYSTEM_PROMPT
from agent.skill_utils import (check_for_bothskills, extract_hard_skills,
                               extract_soft_skills)
from agent.state import ReActConversationState
from agent.todo_utils import read_todos, update_content, write_todos

# ------ Model Initialisation ------ 
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    verbose=True
) 


# ------ Tool Sequence ------ 
tools = [
    update_content,
    write_todos,
    read_todos,
    extract_soft_skills,
    extract_hard_skills,
    check_for_bothskills
]


# ------ Compile and build the agent ------ 
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=GENERAL_SYSTEM_PROMPT,
    state_schema=ReActConversationState,
    checkpointer=MemorySaver(),
)