
import json
from typing import Annotated, Any, Dict, List

import langextract as lx
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.state import ReActConversationState


@tool(description="use this tool to extract soft skills from a job advertisement", parse_docstring=True)
def extract_soft_skills(
    state: Annotated[ReActConversationState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command: 
    """
    Extracts soft skills (e.g., Communication, Leadership) from a given job advertisement and updates the 'soft' list within the global conversation state.

    This tool utilises a language model (via LangExtract) to identify entities in 
    the input job add that correspond to soft skill requirements in a job advertisement. 
    
    Args: 
        state: The injected agent state containing the "job_ads" key with a list of dictionaries which contains the "ad_text"
        tool_call_id: Injected tool call identifier for message tracking 
        
    Returns: 
        Command: A Command object instructing the ReAct agent framework to update 
        the top-level 'skills' field in the ReActConversationState with the 
        new, complete SkillTypes structure.
    """
    
    # 1. Define a promopt for extraction
    EXTRACTION_PROMPT: str = """
    ## Task
    Extract and idenfity soft skills from the job add provided. Use attributes to group soft skills related information! 

    ## Scope & Practices to implement: 
    1. Extract entities in the order they appear in the text.
    2. Use the exacted text for extractions.
    3. Identified entities should not be paraphrased or overlap.
    4. Soft skill attributes should always have the key "skill_type" and the value "soft". 
    """

    # 2. Load examples from JSON file
    with open("../src/data/soft_skills.json", "r") as file: 
        softskills_json: List[Dict[str, Any]] = json.load(file)

    # 3. Formulate high quality examples for LangExtract
    examples = []
    for example in softskills_json: 
        text = example.get("text")
        extractions = example.get("extractions", [])
        if extractions and isinstance(extractions, list) and len(extractions) > 0: 
            for extraction in extractions: 
                examples.append(
                    lx.data.ExampleData(
                        text=text,
                        extractions=[
                            lx.data.Extraction(
                                extraction_class=extraction["extraction_class"],
                                extraction_text=extraction["extraction_text"],
                                attributes=extraction["attributes"]
                            )
                        ]
                    )
                )
    
    # Get job_ads raw text
    job_ads = state.get("job_ads", [])
    if not job_ads: 
        return Command(
            update={
                "messages": [
                    ToolMessage(content="No job_ads were found in the state!", tool_call_id=tool_call_id)
                ]
            }    
        )
    # Extract raw job advertisement text
    advertisements: List[str] =  [raw_text["ad_text"] for raw_text in job_ads]

    updated_dictionary: Dict[str, List[str]] = {}
    for ad in advertisements: 
        # 4. Extract results
        results = lx.extract(
            text_or_documents=ad,
            prompt_description=EXTRACTION_PROMPT,
            examples=examples,
            model_id="qwen2.5:latest",
            model_url="http://localhost:11434"
        )

        # 5. Extract the softskills entities from the result object 
        softskills: List[str] = [r.extraction_class for r in results.extractions]
            #* Normalise soft skills 
        softskills_normalised: List[str] = [" ".join(skill.split()).lower() if "_" in skill else skill.lower() for skill in softskills]

        # 6. Combine unique and current softskills
        current_skill_state = state.get("skills")
        if current_skill_state:
            updated_skill_data_model = current_skill_state.model_copy(deep=True) #* Creating a deep copy of the Pydantic model
            current_softskills = updated_skill_data_model.soft 
            updated_softskills = list(set(current_softskills + softskills_normalised))
        
            # 7. Update the soft skills within the state dictionary 
            updated_skill_data_model.soft = updated_softskills
            updated_skills_dict_value = updated_skill_data_model.model_dump() # Dump the Pydantic model 
            updated_dictionary = updated_skills_dict_value

    return Command(
        update={
            "skills": updated_dictionary,
            "messages": [
                ToolMessage(f"Successfully extracted {len(updated_softskills)} new soft skills. State updated under the 'skills' key.", tool_call_id=tool_call_id)
            ]
        }
    )




@tool(description="use this tool to extract hard skills from a job advertisement", parse_docstring=True)
def extract_hard_skills(
    state: Annotated[ReActConversationState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command: 
    """
    Extracts hard skills (e.g., Communication, Leadership) from a given job advertisement and updates the 'hard' list within the global conversation state.

    This tool utilises a language model (via LangExtract) to identify entities in 
    the input job add that correspond to hard skill requirements in a job advertisement. 
    
    Args: 
        state: The injected agent state containing the "job_ads" key with a list of dictionaries which contains the "ad_text"
        tool_call_id: Injected tool call identifier for message tracking 
        
    Returns: 
        Command: A Command object instructing the ReAct agent framework to update 
        the top-level 'skills' field in the ReActConversationState with the 
        new, complete SkillTypes structure.
    """
    
    # 1. Define a promopt for extraction
    EXTRACTION_PROMPT: str = """
    ## Task
    Extract and idenfity hard skills from the job add provided. Use attributes to group hard skills related information! 

    ## Scope & Practices to implement: 
    1. Extract entities in the order they appear in the text.
    2. Use the exacted text for extractions.
    3. Identified entities should not be paraphrased or overlap.
    4. hard skill attributes should always have the key "skill_type" and the value "hard". 
    """

    # 2. Load examples from JSON file
    with open("../src/data/hard_skills.json", "r") as file: 
        hardskills_json: List[Dict[str, Any]] = json.load(file)

    # 3. Formulate high quality examples for LangExtract
    examples = []
    for example in hardskills_json: 
        text = example.get("text")
        extractions = example.get("extractions", [])
        if extractions and isinstance(extractions, list) and len(extractions) > 0: 
            for extraction in extractions: 
                examples.append(
                    lx.data.ExampleData(
                        text=text,
                        extractions=[
                            lx.data.Extraction(
                                extraction_class=extraction["extraction_class"],
                                extraction_text=extraction["extraction_text"],
                                attributes=extraction["attributes"]
                            )
                        ]
                    )
                )
    
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

    updated_dictionary: Dict[str, List[str]] = {}
    for ad in advertisements: 
        # 4. Extract results
        results = lx.extract(
            text_or_documents=ad,
            prompt_description=EXTRACTION_PROMPT,
            examples=examples,
            model_id="qwen2.5:latest",
            model_url="http://localhost:11434"
        )

        # 5. Extract the hardskills entities from the result object 
        hardskills: List[str] = [r.extraction_class for r in results.extractions]
                    #* Normalise soft skills 
        hardskills_normalised: List[str] = [" ".join(skill.split()).lower() if "_" in skill else skill.lower() for skill in hardskills]

        # 6. Combine unique and current hardskills
        current_skill_state = state.get("skills")
        if current_skill_state: 
            updated_skill_data_model = current_skill_state.model_copy(deep=True) #* Creating a deep copy of the Pydantic model
            current_hardskills = updated_skill_data_model.hard 
            updated_hardskills = list(set(current_hardskills + hardskills_normalised))
            
            # 7. Update the hard skills within the state dictionary 
            updated_skill_data_model.hard = updated_hardskills
            updated_skills_dict_value = updated_skill_data_model.model_dump() # Dump the Pydantic model 
            updated_dictionary = updated_skills_dict_value
        
    return Command(
        update={
            "skills": updated_dictionary,
            "messages": [
                ToolMessage(f"Successfully extracted {len(updated_hardskills)} new hard skills. State updated under the 'skills' key.", tool_call_id=tool_call_id)
            ]
        }
    )




@tool(description="use this tool to extract hard skills from a job advertisement", parse_docstring=True)
def check_for_bothskills(
    state: Annotated[ReActConversationState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command: 
    """
    Validates and resolves overlapping skills that were categorized as both 'hard' and 'soft'.

    This function compares the lists of hard skills and soft skills currently stored in the agent's 
    state. Any skill found in both lists is moved to a dedicated 'both' category, and removed 
    from the original 'hard' and 'soft' lists. This is a naive but effective approach to 
    disambiugate skill types after an LLM extraction process, where the LLM might classify 
    the same term (e.g., 'communication') differently depending on context or prompt variation.
    
    Args: 
        state: The injected agent state containing the 'skills' field (of type SkillTypes) 
               which holds the 'hard', 'soft', and 'both' skill lists.
        tool_call_id: Injected tool call identifier for message tracking.
        
    Returns: 
        Command: A Command object instructing the ReAct agent framework to update 
        the top-level 'skills' field with the refined SkillTypes structure.
    """
        
    # Extract hard & soft skills from state
    #* Create a deep copy for SkillTypes
    skill_state = state.get("skills")
    if skill_state: 
        existing_skill_state = skill_state.model_copy(deep=True)
        hard_skills = existing_skill_state.hard
        soft_skills = existing_skill_state.soft

    #Evaluation logic for both - Naive approach (can also be used with LangExtract)
    #* Create sets
    hard_set: set = set(hard_skills) 
    soft_set: set = set(soft_skills)
    both_skills = []

    #* Iterate over skill intersection to check if existing hard/soft skills match
    if len(hard_set.intersection(soft_set)) > 0:
        intersection: set = hard_set.intersection(soft_set)
        for skill in intersection:
            both_skills.append(skill)
            hard_set.remove(skill) # remove matching hard skill from hard_set
            soft_set.remove(skill) # remove matching soft skill from soft_set
    else: 
        return Command(
            update={
                "messages": [ToolMessage(content="Hard & Soft skills were validated, all values are unique!", tool_call_id=tool_call_id)]
            }
        )
    
    #* Update hard, soft, and both skills accordingly
    hard_skills_updated: List[str] = list(hard_set)
    soft_skills_updated: List[str] = list(soft_set)
    existing_skill_state.hard = hard_skills_updated
    existing_skill_state.soft = soft_skills_updated
    existing_skill_state.both = both_skills

    #* Dump the Pydantic model with the updated skills
    updated_skills_dict_value = existing_skill_state.model_dump()
    
    return Command(
            update={
                "skills": updated_skills_dict_value,
                "messages": [
                    ToolMessage(f"Found {len(both_skills)} matching skills when validated HARD and SOFT skills. State updated under the 'skills' key.", tool_call_id=tool_call_id)
                ]
            }
        )