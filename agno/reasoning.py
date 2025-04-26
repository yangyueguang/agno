from enum import Enum
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Union
from agno.models import Model, Message
from agno.tools import Function, Toolkit
from agno.run import RunMessages


class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
    action: Optional[str] = Field(None, description="The action derived from this step. Talk in first person like I will ... ")
    result: Optional[str] = Field(None, description="The result of executing the action. Talk in first person like I did this and got ... ")
    reasoning: Optional[str] = Field(None, description="The thought process and considerations behind this step")
    next_action: Optional[NextAction] = Field(None,
        description="Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer")
    confidence: Optional[float] = Field(None, description="Confidence score for this step (0.0 to 1.0)")


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="A list of reasoning steps")


def get_next_action(reasoning_step: ReasoningStep) -> NextAction:
    next_action = reasoning_step.next_action or NextAction.FINAL_ANSWER
    if isinstance(next_action, str):
        try:
            return NextAction(next_action)
        except ValueError:
            print(f"Reasoning error. Invalid next action: {next_action}")
            return NextAction.FINAL_ANSWER
    return next_action


def update_messages_with_reasoning(run_messages: RunMessages,
    reasoning_messages: List[Message]) -> None:
    run_messages.messages.append(Message(role="assistant",
            content="I have worked through this problem in-depth, running all necessary tools and have included my raw, step by step research. ",
            add_to_agent_memory=False))
    for message in reasoning_messages:
        message.add_to_agent_memory = False
    run_messages.messages.extend(reasoning_messages)
    run_messages.messages.append(Message(role="assistant",
            content="Now I will summarize my reasoning and provide a final answer. I will skip any tool calls already executed and steps that are not relevant to the final answer.",
            add_to_agent_memory=False))


def get_default_reasoning_agent(reasoning_model: Model,
    min_steps: int,
    max_steps: int,
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None,
    use_json_mode: bool = False,
    monitoring: bool = False,
    telemetry: bool = True,
    debug_mode: bool = False) -> Optional["Agent"]:
    from agno.agent import Agent
    agent = Agent(model=reasoning_model,
        description="You are a meticulous, thoughtful, and logical Reasoning Agent who solves complex problems through clear, structured, step-by-step analysis.",
        instructions=dedent(f"""
        Step 1 - Problem Analysis:
        - Restate the user's task clearly in your own words to ensure full comprehension.
        - Identify explicitly what information is required and what tools or resources might be necessary.
        Step 2 - Decompose and Strategize:
        - Break down the problem into clearly defined subtasks.
        - Develop at least two distinct strategies or approaches to solving the problem to ensure thoroughness.
        Step 3 - Intent Clarification and Planning:
        - Clearly articulate the user's intent behind their request.
        - Select the most suitable strategy from Step 2, clearly justifying your choice based on alignment with the user's intent and task constraints.
        - Formulate a detailed step-by-step action plan outlining the sequence of actions needed to solve the problem.
        Step 4 - Execute the Action Plan:
        For each planned step, document:
        1. **Title**: Concise title summarizing the step.
        2. **Action**: Explicitly state your next action in the first person ('I will...').
        3. **Result**: Execute your action using necessary tools and provide a concise summary of the outcome.
        4. **Reasoning**: Clearly explain your rationale, covering:
            - Necessity: Why this action is required.
            - Considerations: Highlight key considerations, potential challenges, and mitigation strategies.
            - Progression: How this step logically follows from or builds upon previous actions.
            - Assumptions: Explicitly state any assumptions made and justify their validity.
        5. **Next Action**: Clearly select your next step from:
            - **continue**: If further steps are needed.
            - **validate**: When you reach a potential answer, signaling it's ready for validation.
            - **final_answer**: Only if you have confidently validated the solution.
            - **reset**: Immediately restart analysis if a critical error or incorrect result is identified.
        6. **Confidence Score**: Provide a numeric confidence score (0.0–1.0) indicating your certainty in the step’s correctness and its outcome.
        Step 5 - Validation (mandatory before finalizing an answer):
        - Explicitly validate your solution by:
            - Cross-verifying with alternative approaches (developed in Step 2).
            - Using additional available tools or methods to independently confirm accuracy.
        - Clearly document validation results and reasoning behind the validation method chosen.
        - If validation fails or discrepancies arise, explicitly identify errors, reset your analysis, and revise your plan accordingly.
        Step 6 - Provide the Final Answer:
        - Once thoroughly validated and confident, deliver your solution clearly and succinctly.
        - Restate briefly how your answer addresses the user's original intent and resolves the stated task.
        General Operational Guidelines:
        - Ensure your analysis remains:
            - **Complete**: Address all elements of the task.
            - **Comprehensive**: Explore diverse perspectives and anticipate potential outcomes.
            - **Logical**: Maintain coherence between all steps.
            - **Actionable**: Present clearly implementable steps and actions.
            - **Insightful**: Offer innovative and unique perspectives where applicable.
        - Always explicitly handle errors and mistakes by resetting or revising steps immediately.
        - Adhere strictly to a minimum of {min_steps} and maximum of {max_steps} steps to ensure effective task resolution.
        - Execute necessary tools proactively and without hesitation, clearly documenting tool usage.
        """),
        tools=tools,
        show_tool_calls=False,
        response_model=ReasoningSteps,
        use_json_mode=use_json_mode,
        monitoring=monitoring,
        telemetry=telemetry,
        debug_mode=debug_mode)
    agent.model.show_tool_calls = False
    return agent


def get_openai_reasoning_agent(reasoning_model: Model, **kwargs) -> "Agent":
    from agno.agent import Agent
    return Agent(model=reasoning_model, **kwargs)


def get_openai_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:
    from agno.run import RunResponse
    try:
        reasoning_agent_response: RunResponse = reasoning_agent.run(messages=messages)
    except Exception as e:
        print(f"Reasoning error: {e}")
        return None
    reasoning_content: str = ""
    if reasoning_agent_response.content is not None:
        content = reasoning_agent_response.content
        if "<think>" in content and "</think>" in content:
            start_idx = content.find("<think>") + len("<think>")
            end_idx = content.find("</think>")
            reasoning_content = content[start_idx:end_idx].strip()
        else:
            reasoning_content = content
    return Message(role="assistant", content=f"<thinking>\n{reasoning_content}\n</thinking>", reasoning_content=reasoning_content)


async def aget_openai_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:
    from agno.run import RunResponse
    for message in messages:
        if message.role == "developer":
            message.role = "system"
    try:
        reasoning_agent_response: RunResponse = await reasoning_agent.arun(messages=messages)
    except Exception as e:
        print(f"Reasoning error: {e}")
        return None
    reasoning_content: str = ""
    if reasoning_agent_response.content is not None:
        content = reasoning_agent_response.content
        if "<think>" in content and "</think>" in content:
            start_idx = content.find("<think>") + len("<think>")
            end_idx = content.find("</think>")
            reasoning_content = content[start_idx:end_idx].strip()
        else:
            reasoning_content = content
    return Message(role="assistant", content=f"<thinking>\n{reasoning_content}\n</thinking>", reasoning_content=reasoning_content)


def get_deepseek_reasoning_agent(reasoning_model: Model, monitoring: bool = False) -> "Agent":
    from agno.agent import Agent
    return Agent(model=reasoning_model, monitoring=monitoring)


def get_deepseek_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:
    from agno.run import RunResponse
    for message in messages:
        if message.role == "developer":
            message.role = "system"
    try:
        reasoning_agent_response: RunResponse = reasoning_agent.run(messages=messages)
    except Exception as e:
        print(f"Reasoning error: {e}")
        return None
    reasoning_content: str = ""
    if reasoning_agent_response.messages is not None:
        for msg in reasoning_agent_response.messages:
            if msg.reasoning_content is not None:
                reasoning_content = msg.reasoning_content
                break
    return Message(role="assistant", content=f"<thinking>\n{reasoning_content}\n</thinking>", reasoning_content=reasoning_content)


async def aget_deepseek_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:
    from agno.run import RunResponse
    for message in messages:
        if message.role == "developer":
            message.role = "system"
    try:
        reasoning_agent_response: RunResponse = await reasoning_agent.arun(messages=messages)
    except Exception as e:
        print(f"Reasoning error: {e}")
        return None
    reasoning_content: str = ""
    if reasoning_agent_response.messages is not None:
        for msg in reasoning_agent_response.messages:
            if msg.reasoning_content is not None:
                reasoning_content = msg.reasoning_content
                break
    return Message(role="assistant", content=f"<thinking>\n{reasoning_content}\n</thinking>", reasoning_content=reasoning_content)
