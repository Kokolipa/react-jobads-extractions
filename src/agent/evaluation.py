from typing import Annotated, List, Tuple

from jinja2 import Template
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import InjectedToolCallId, tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.state import ReActConversationState


# Define the structured output class
class EvalResults(BaseModel): 
    "Schema definition for G-Eval results"
    reasoning: List[str] = Field(description="Step-by-step explanation for the scores for each item in the extracted results.", default_factory=list)
    score: List[float] = Field(description="A list of scores, one for each item in the extracted results in the order they were provided.", default_factory=list) 


parser = JsonOutputParser(pydantic_object=EvalResults)
format_instructions = parser.get_format_instructions() # We need these instructions to ensure the output format is aligned with LangChain for jinja2 

class GEvalTemplate:
    """
    A class to encapsulate and format a G-Eval prompt using a Jinja2 template.
    """
    def __init__(
            self,
            prompt: str,
            input_variables:List[str],
            evaluation_steps: List[str],
            eval_criteria: str,
            criteria_definition: str,
            score_range: Tuple[int, int],
            task: str
        ):
        self.prompt = Template(prompt)
        self.input_variables = input_variables
        self.evaluation_steps = evaluation_steps
        self.eval_criteria = eval_criteria
        self.criteria_definition = criteria_definition
        self.score_range = score_range 
        self.task = task 
    
    def format(self, **kwargs):
        """Renders the Jinja2 template with the provided keyword arguments."""
        return self.prompt.render(**kwargs)


# Jinja2 - Flexible prompt for G-Eval
G_EVAL_PROMPT = """
<role>You are an expert and meticulous HR data quality auditor. Your primary directive is to adhere strictly to the provided task, evaluation steps, and output format.</role> 

<task>
Your task is to evaluate a set of extracted {{task}} against the original job advertisement based on the following evaluation scoring criterion:

## Scoring Criterion
Rate **EACH** of the generated results based on **{{eval_criteria}}** on a scale of {{ score_min }} to {{ score_max }} based on the following criterion definition: 
{{criteria_definition}}

**Scoring Reference:**
- Score Range {{ score_max - 2 }}-{{score_max}}: High compliance with all aspects of the criterion (e.g., the skill is explicitly and accurately mentioned in the ad).
- Score {{ score_min + 3 }}-{{score_max - 2}}: Medium compliance (e.g., the skill is implied or a close synonym is used; issues present).
- Score {{ score_min }}-{{ score_min + 3 }}: Complete failure to meet the core criterion (e.g., the skill is a hallucination and not mentioned or is a severe misrepresentation).
</task>

<eval_steps>
## Evaluation Steps
{% for eval_step in evaluation_steps %}
{{ loop.index}}. {{eval_step}}
{% endfor %}
</eval_steps>

## Evaluation Context
**Job Advertisement:**
{{ job_ad }}

**{{task.title()}} Extracted for Evaluation:** 
{% for output in extracted_outputs | default([]) %}
- {{ output | default("N/A") }}
{% endfor %}

## Ouput Format
Please make sure to **only** return a single JSON object. The 'score' field must be a **list of numbers**, where **EACH number corresponds to the rating of a specific extracted item** following the order they are listed in the '{{task.title()}} Extracted for Evaluation' section.

**JSON Schema Instructions:**
{{ format_instructions }}
"""



@tool(description="Use this tool to perform G-Eval for 'Correctness' criterion.", parse_docstring=True)
def evaluate_correctness(
    state: Annotated[ReActConversationState, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Performs G-Eval on extracted job data (skills, requirements, responsibilities)
    for the 'Correctness' criterion.

    This function iterates through all job advertisements and extracted lists ('skills', 
    'requirements', 'responsibilities') stored in the agent's state. It uses a structured 
    LLM call (G-Eval) to score the factual accuracy of each extracted item against the 
    original job advertisement text. The results, including reasoning and a normalized 
    scores (0.0 to 1.0), are stored back into the agent's state.

    Args:
        state: The injected agent state containing the "job_ads" key with a list of dictionaries which contains the "ad_text". 
        tool_call_id: The injected identifier for tracking this specific tool invocation.

    Returns:
        Command: A Command object instructing the ReAct agent to update the state 
        with the evaluation results and log a completion message.
    """
    
    # Instantiate the model
    llm_with_struct_output = ChatOllama(
        model="qwen2.5:latest",
        temperature=0,
        verbose=True
    ).with_structured_output(EvalResults)

    # Criteria definition 
    criteria: str = """
    Correctness (0-10) - the overall factual accuracy of the extracted skills. This dimension measures the alignment between the model's output and the source job advertisement.
    The skill list must not contain any skills that are completely hallucinated (not mentioned) or misrepresent a requirement (e.g., extracting 'Python' when the ad only mentions 'R').
    """

    # Evaluation steps 
    evaluation_steps: List[str] =[
        "Read the Job Advertisement carefully and identify all explicit skill requirements.",
        "Review the entire list of Extracted Skills.",
        "For EACH extracted skill, verify that an equivalent or exact term is present in the Job Advertisement. If a skill is not mentioned, mark it as an hallucination.",
        "Check if the extracted skill names accurately reflect the ad (e.g., if the ad says 'AWS proficiency', the output should not just be 'Cloud').",
        "Assign a score for Correctness on a scale of 0 to 10, where 1 is the lowest (high hallucination/misrepresentation) and 5 is the highest (perfect factual accuracy) based on the Evaluation Criteria."
    ]

    # Score Range
    eval_criteria = "Correctness"
    score_range = (0, 10)

    # Input variables definition
    input_variables = ["job_ad", "extracted_outputs"]

    # define the TASKS
    tasks = ["skills", "requirements", "responsibilities"]

    
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

    # Iterate over the state and advertisements to get the values required for evaluation
    result_evaluation = {}
    for ad in advertisements: 
        for task in tasks: 
            state_values = state.get(task)
            #? Skills G-Eval results
            if task == "skills":
                if state_values: 
                    hard: List[str] = state_values.hard #type: ignore
                    soft: List[str] = state_values.soft #type: ignore
                    both: List[str] = state_values.both #type: ignore
                    all_skills_outputs = [hard, soft, both]

                    # Create GEval Template instance
                    geval_template_instance = GEvalTemplate(
                        prompt=G_EVAL_PROMPT,
                        input_variables=input_variables,
                        evaluation_steps=evaluation_steps,
                        eval_criteria=eval_criteria,
                        criteria_definition=criteria,
                        score_range=score_range,
                        task=task
                    )

                    skill_results = []
                    for skills, name in zip(all_skills_outputs, ["hard", "soft", "both"]): 
                        #* ensure skill is not empty
                        if skills: 
                            # Format the prompt
                            formatted_prompt = geval_template_instance.format(
                                job_ad=ad,
                                extracted_outputs=skills,
                                evaluation_steps=geval_template_instance.evaluation_steps,
                                eval_criteria=geval_template_instance.eval_criteria,
                                criteria_definition=geval_template_instance.criteria_definition,
                                task=geval_template_instance.task,
                                score_min=score_range[0],
                                score_max=score_range[1],
                                format_instructions=format_instructions
                            )

                            # Invoke the model
                            result: EvalResults = llm_with_struct_output.invoke(formatted_prompt) #type: ignore
                            reasoning = result.reasoning
                            scores = result.score
                            #* Normalising scores
                            scores = [score / score_range[1] for score in scores]
                            skill_results.append({name: {"reasoning": reasoning, "score": scores}})
                    
                    # Append all skill results to result_dictionary 
                    result_evaluation[task] = skill_results
            else:
                if state_values:
                    # Create GEval Template instance
                    geval_template_instance = GEvalTemplate(
                        prompt=G_EVAL_PROMPT,
                        input_variables=input_variables,
                        evaluation_steps=evaluation_steps,
                        eval_criteria=eval_criteria,
                        criteria_definition=criteria,
                        score_range=score_range,
                        task=task
                    )
                    
                    # Format the prompt
                    formatted_prompt = geval_template_instance.format(
                        job_ad=ad,
                        extracted_outputs=state_values,
                        evaluation_steps=geval_template_instance.evaluation_steps,
                        eval_criteria=geval_template_instance.eval_criteria,
                        criteria_definition=geval_template_instance.criteria_definition,
                        task=geval_template_instance.task,
                        score_min=score_range[0],
                        score_max=score_range[1],
                        format_instructions=format_instructions
                    )
                    # Invoke the model
                    result: EvalResults = llm_with_struct_output.invoke(formatted_prompt) # type: ignore 
                    reasoning = result.reasoning
                    scores = result.score
                    #* Normalising scores
                    scores = [score / score_range[1] for score in scores]
                    result_evaluation[task] = {"reasoning": reasoning, "score": scores}  # type: ignore 
                    
    # Update the state
    return Command(
        update={
            "evaluation": result_evaluation,
            "messages": [
                ToolMessage(content=f"G-Eval for creterion: {eval_criteria} was conducted successfuly!", tool_call_id=tool_call_id)
                ]
        }
    )
