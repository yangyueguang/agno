from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Union, Any
from agno.tools import Function, Toolkit
from agno.media import AudioArtifact, AudioResponse, ImageArtifact, VideoArtifact
from agno.models import Citations, Message, MessageReferences, Model


@dataclass
class RunMessages:
    messages: List[Message] = field(default_factory=list)
    system_message: Optional[Message] = None
    user_message: Optional[Message] = None
    extra_messages: Optional[List[Message]] = None

    def get_input_messages(self) -> List[Message]:
        input_messages = []
        if self.system_message is not None:
            input_messages.append(self.system_message)
        if self.user_message is not None:
            input_messages.append(self.user_message)
        if self.extra_messages is not None:
            input_messages.extend(self.extra_messages)
        return input_messages


class NextAction(str, Enum):
    CONTINUE = 'continue'
    VALIDATE = 'validate'
    FINAL_ANSWER = 'final_answer'
    RESET = 'reset'


class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description='A concise title summarizing the step"s purpose')
    action: Optional[str] = Field(None, description='The action derived from this step. Talk in first person like I will ... ')
    result: Optional[str] = Field(None, description='The result of executing the action. Talk in first person like I did this and got ... ')
    reasoning: Optional[str] = Field(None, description='The thought process and considerations behind this step')
    next_action: Optional[NextAction] = Field(None, description='Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer')
    confidence: Optional[float] = Field(None, description='Confidence score for this step (0.0 to 1.0)')


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description='A list of reasoning steps')


def get_next_action(reasoning_step: ReasoningStep) -> NextAction:
    next_action = reasoning_step.next_action or NextAction.FINAL_ANSWER
    if isinstance(next_action, str):
        try:
            return NextAction(next_action)
        except ValueError:
            print(f'Reasoning error. Invalid next action: {next_action}')
            return NextAction.FINAL_ANSWER
    return next_action


def update_messages_with_reasoning(run_messages: RunMessages, reasoning_messages: List[Message]) -> None:
    run_messages.messages.append(Message(role='assistant', content='I have worked through this problem in-depth, running all necessary tools and have included my raw, step by step research. ', add_to_agent_memory=False))
    for message in reasoning_messages:
        message.add_to_agent_memory = False
    run_messages.messages.extend(reasoning_messages)
    run_messages.messages.append(Message(role='assistant', content='Now I will summarize my reasoning and provide a final answer. I will skip any tool calls already executed and steps that are not relevant to the final answer.', add_to_agent_memory=False))


def get_default_reasoning_agent(reasoning_model: Model, min_steps: int, max_steps: int, tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None, use_json_mode: bool = False, monitoring: bool = False, telemetry: bool = True, debug_mode: bool = False) -> Optional['Agent']:
    from agno.agent import Agent
    agent = Agent(model=reasoning_model, description='You are a meticulous, thoughtful, and logical Reasoning Agent who solves complex problems through clear, structured, step-by-step analysis.', instructions=dedent(f'''
步骤1-问题分析：\n-用你自己的话清楚地重述用户的任务，以确保完全理解。\n-明确指出需要哪些信息以及可能需要哪些工具或资源。
第2步-分解和制定战略：\n-将问题分解为明确定义的子任务。\n-制定至少两种不同的策略或方法来解决问题，以确保彻底性。
第3步-意图澄清和规划：\n-清楚地表达用户请求背后的意图。\n-从步骤2中选择最合适的策略，根据与用户意图和任务约束的一致性清楚地证明你的选择。\n-制定详细的分步行动计划，概述解决问题所需的行动顺序。
步骤4-执行行动计划：
对于每个计划步骤，记录：\n1.**标题**：概括步骤的简明标题。\n2.**行动**：以第一人称明确说明你的下一个行动（“我会……”）。\n3.**结果**：使用必要的工具执行行动，并提供结果的简明摘要。\n4.**推理**：清楚地解释你的理由，包括：
-必要性：为什么需要采取这一行动。\n-注意事项：强调关键考虑因素、潜在挑战和缓解策略。\n-进展：这一步如何从逻辑上遵循或建立在之前的行动之上。\n-假设：明确说明所做的任何假设，并证明其有效性。
5.**下一步行动**：从以下选项中明确选择下一步：
-**继续**：如果需要进一步的步骤。\n-**验证**：当你得到一个潜在的答案时，表明它已经准备好进行验证。\n-**最终答案**：只有当您自信地验证了解决方案时。\n-**重置**：如果发现严重错误或不正确的结果，请立即重新开始分析。
6.**置信度分数**：提供一个数字置信度分数（0.0-1.0），表明您对步骤的正确性及其结果的确定性。
步骤5-验证（在最终确定答案之前必须进行）：
-通过以下方式明确验证您的解决方案：\n-与替代方法进行交叉验证（在步骤2中开发）。\n-使用其他可用工具或方法独立确认准确性。\n-清楚地记录验证结果和所选验证方法背后的推理。\n-如果验证失败或出现差异，明确指出错误，重置分析，并相应地修改计划。
第6步-提供最终答案：
-一旦经过彻底验证并充满信心，就可以清晰简洁地交付您的解决方案。\n-简要重述你的答案如何满足用户的初衷并解决所述任务。
一般操作指南：
-确保您的分析保持不变：
-**完成**：解决任务的所有要素。\n-**全面**：探索不同的观点并预测潜在的结果。\n-**逻辑**：保持所有步骤之间的连贯性。\n-**可操作**：提出明确可执行的步骤和行动。\n-**富有洞察力**：在适用的情况下提供创新和独特的视角。
-始终通过立即重置或修改步骤来明确处理错误和失误。\n-严格遵守最小{min_steps}和最大{max_steps}步数，以确保有效的任务解决。
-主动毫不犹豫地执行必要的工具，清楚地记录工具的使用情况。'''), tools=tools, show_tool_calls=False, response_model=ReasoningSteps, use_json_mode=use_json_mode, monitoring=monitoring, telemetry=telemetry, debug_mode=debug_mode)
    agent.model.show_tool_calls = False
    return agent


def get_openai_reasoning_agent(reasoning_model: Model, **kwargs) -> 'Agent':
    from agno.agent import Agent
    return Agent(model=reasoning_model, **kwargs)


def get_openai_reasoning(reasoning_agent: 'Agent', messages: List[Message]) -> Optional[Message]:
    try:
        reasoning_agent_response: RunResponse = reasoning_agent.run(messages=messages)
    except Exception as e:
        print(f'Reasoning error: {e}')
        return None
    reasoning_content: str = ''
    if reasoning_agent_response.content is not None:
        content = reasoning_agent_response.content
        if '<think>' in content and '</think>' in content:
            start_idx = content.find('<think>') + len('<think>')
            end_idx = content.find('</think>')
            reasoning_content = content[start_idx:end_idx].strip()
        else:
            reasoning_content = content
    return Message(role='assistant', content=f'<thinking>\n{reasoning_content}\n</thinking>', reasoning_content=reasoning_content)


async def aget_openai_reasoning(reasoning_agent: 'Agent', messages: List[Message]) -> Optional[Message]:
    for message in messages:
        if message.role == 'developer':
            message.role = 'system'
    try:
        reasoning_agent_response: RunResponse = await reasoning_agent.arun(messages=messages)
    except Exception as e:
        print(f'Reasoning error: {e}')
        return None
    reasoning_content: str = ''
    if reasoning_agent_response.content is not None:
        content = reasoning_agent_response.content
        if '<think>' in content and '</think>' in content:
            start_idx = content.find('<think>') + len('<think>')
            end_idx = content.find('</think>')
            reasoning_content = content[start_idx:end_idx].strip()
        else:
            reasoning_content = content
    return Message(role='assistant', content=f'<thinking>\n{reasoning_content}\n</thinking>', reasoning_content=reasoning_content)


def get_deepseek_reasoning_agent(reasoning_model: Model, monitoring: bool = False) -> 'Agent':
    from agno.agent import Agent
    return Agent(model=reasoning_model, monitoring=monitoring)


def get_deepseek_reasoning(reasoning_agent: 'Agent', messages: List[Message]) -> Optional[Message]:
    for message in messages:
        if message.role == 'developer':
            message.role = 'system'
    try:
        reasoning_agent_response: RunResponse = reasoning_agent.run(messages=messages)
    except Exception as e:
        print(f'Reasoning error: {e}')
        return None
    reasoning_content: str = ''
    if reasoning_agent_response.messages is not None:
        for msg in reasoning_agent_response.messages:
            if msg.reasoning_content is not None:
                reasoning_content = msg.reasoning_content
                break
    return Message(role='assistant', content=f'<thinking>\n{reasoning_content}\n</thinking>', reasoning_content=reasoning_content)


async def aget_deepseek_reasoning(reasoning_agent: 'Agent', messages: List[Message]) -> Optional[Message]:
    for message in messages:
        if message.role == 'developer':
            message.role = 'system'
    try:
        reasoning_agent_response: RunResponse = await reasoning_agent.arun(messages=messages)
    except Exception as e:
        print(f'Reasoning error: {e}')
        return None
    reasoning_content: str = ''
    if reasoning_agent_response.messages is not None:
        for msg in reasoning_agent_response.messages:
            if msg.reasoning_content is not None:
                reasoning_content = msg.reasoning_content
                break
    return Message(role='assistant', content=f'<thinking>\n{reasoning_content}\n</thinking>', reasoning_content=reasoning_content)


class RunEvent(str, Enum):
    run_started = 'RunStarted'
    run_response = 'RunResponse'
    run_completed = 'RunCompleted'
    run_error = 'RunError'
    run_cancelled = 'RunCancelled'
    tool_call_started = 'ToolCallStarted'
    tool_call_completed = 'ToolCallCompleted'
    reasoning_started = 'ReasoningStarted'
    reasoning_step = 'ReasoningStep'
    reasoning_completed = 'ReasoningCompleted'
    updating_memory = 'UpdatingMemory'
    workflow_started = 'WorkflowStarted'
    workflow_completed = 'WorkflowCompleted'


@dataclass
class RunResponseExtraData:
    references: Optional[List[MessageReferences]] = None
    add_messages: Optional[List[Message]] = None
    reasoning_steps: Optional[List['ReasoningStep']] = None
    reasoning_messages: Optional[List[Message]] = None

    def to_dict(self) -> Dict[str, Any]:
        _dict = {}
        if self.add_messages is not None:
            _dict['add_messages'] = [m.to_dict() for m in self.add_messages]
        if self.reasoning_messages is not None:
            _dict['reasoning_messages'] = [m.to_dict() for m in self.reasoning_messages]
        if self.reasoning_steps is not None:
            _dict['reasoning_steps'] = [rs.model_dump() for rs in self.reasoning_steps]
        if self.references is not None:
            _dict['references'] = [r.model_dump() for r in self.references]
        return _dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResponseExtraData':
        add_messages = data.pop('add_messages', None)
        add_messages = [Message.model_validate(message) for message in add_messages] if add_messages else None
        history = data.pop('history', None)
        history = [Message.model_validate(message) for message in history] if history else None
        reasoning_steps = data.pop('reasoning_steps', None)
        reasoning_steps = [ReasoningStep.model_validate(step) for step in reasoning_steps] if reasoning_steps else None
        reasoning_messages = data.pop('reasoning_messages', None)
        reasoning_messages = ([Message.model_validate(message) for message in reasoning_messages] if reasoning_messages else None)
        references = data.pop('references', None)
        references = [MessageReferences.model_validate(reference) for reference in references] if references else None
        return cls(add_messages=add_messages, reasoning_steps=reasoning_steps, reasoning_messages=reasoning_messages, references=references)


@dataclass
class RunResponse:
    content: Optional[Any] = None
    content_type: str = 'str'
    thinking: Optional[str] = None
    event: str = RunEvent.run_response.value
    messages: Optional[List[Message]] = None
    metrics: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    formatted_tool_calls: Optional[List[str]] = None
    images: Optional[List[ImageArtifact]] = None
    videos: Optional[List[VideoArtifact]] = None
    audio: Optional[List[AudioArtifact]] = None
    response_audio: Optional[AudioResponse] = None
    citations: Optional[Citations] = None
    extra_data: Optional[RunResponseExtraData] = None
    created_at: int = field(default_factory=lambda: int(time()))

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in asdict(self).items()
            if v is not None and k not in ['messages', 'extra_data', 'images', 'videos', 'audio', 'response_audio']}
        if self.messages is not None:
            _dict['messages'] = [m.to_dict() for m in self.messages]
        if self.extra_data is not None:
            _dict['extra_data'] = (self.extra_data.to_dict() if isinstance(self.extra_data, RunResponseExtraData) else self.extra_data)
        if self.images is not None:
            _dict['images'] = [img.model_dump(exclude_none=True) for img in self.images]
        if self.videos is not None:
            _dict['videos'] = [vid.model_dump(exclude_none=True) for vid in self.videos]
        if self.audio is not None:
            _dict['audio'] = [aud.model_dump(exclude_none=True) for aud in self.audio]
        if self.response_audio is not None:
            _dict['response_audio'] = (self.response_audio.to_dict() if isinstance(self.response_audio, AudioResponse) else self.response_audio)
        if isinstance(self.content, BaseModel):
            _dict['content'] = self.content.model_dump(exclude_none=True)
        return _dict

    def to_json(self) -> str:
        import json
        _dict = self.to_dict()
        return json.dumps(_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResponse':
        messages = data.pop('messages', None)
        messages = [Message.model_validate(message) for message in messages] if messages else None
        return cls(messages=messages, **data)

    def get_content_as_string(self, **kwargs) -> str:
        import json
        from pydantic import BaseModel
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, **kwargs)


@dataclass
class TeamRunResponse:
    event: str = RunEvent.run_response.value
    content: Optional[Any] = None
    content_type: str = 'str'
    thinking: Optional[str] = None
    messages: Optional[List[Message]] = None
    metrics: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    member_responses: List[Union['TeamRunResponse', RunResponse]] = field(default_factory=list)
    run_id: Optional[str] = None
    team_id: Optional[str] = None
    session_id: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    formatted_tool_calls: Optional[List[str]] = None
    images: Optional[List[ImageArtifact]] = None
    videos: Optional[List[VideoArtifact]] = None
    audio: Optional[List[AudioArtifact]] = None
    response_audio: Optional[AudioResponse] = None
    citations: Optional[Citations] = None
    extra_data: Optional[RunResponseExtraData] = None
    created_at: int = field(default_factory=lambda: int(time()))

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in asdict(self).items()
            if v is not None and k not in ['messages', 'extra_data', 'images', 'videos', 'audio', 'response_audio']}
        if self.messages is not None:
            _dict['messages'] = [m.to_dict() for m in self.messages]
        if self.extra_data is not None:
            _dict['extra_data'] = self.extra_data.to_dict()
        if self.images is not None:
            _dict['images'] = [img.model_dump(exclude_none=True) for img in self.images]
        if self.videos is not None:
            _dict['videos'] = [vid.model_dump(exclude_none=True) for vid in self.videos]
        if self.audio is not None:
            _dict['audio'] = [aud.model_dump(exclude_none=True) for aud in self.audio]
        if self.response_audio is not None:
            _dict['response_audio'] = self.response_audio.to_dict()
        if self.member_responses:
            _dict['member_responses'] = [response.to_dict() for response in self.member_responses]
        if isinstance(self.content, BaseModel):
            _dict['content'] = self.content.model_dump(exclude_none=True)
        return _dict

    def to_json(self) -> str:
        import json
        _dict = self.to_dict()
        return json.dumps(_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamRunResponse':
        messages = data.pop('messages', None)
        messages = [Message.model_validate(message) for message in messages] if messages else None
        member_responses = data.pop('member_responses', None)
        parsed_member_responses: List[Union['TeamRunResponse', RunResponse]] = []
        if member_responses is not None:
            for response in member_responses:
                if 'agent_id' in response:
                    parsed_member_responses.append(RunResponse.from_dict(response))
                else:
                    parsed_member_responses.append(cls.from_dict(response))
        extra_data = data.pop('extra_data', None)
        if extra_data is not None:
            extra_data = RunResponseExtraData.from_dict(extra_data)
        images = data.pop('images', None)
        images = [ImageArtifact.model_validate(image) for image in images] if images else None
        videos = data.pop('videos', None)
        videos = [VideoArtifact.model_validate(video) for video in videos] if videos else None
        audio = data.pop('audio', None)
        audio = [AudioArtifact.model_validate(audio) for audio in audio] if audio else None
        response_audio = data.pop('response_audio', None)
        response_audio = AudioResponse.model_validate(response_audio) if response_audio else None
        return cls(messages=messages, member_responses=parsed_member_responses, extra_data=extra_data, images=images, videos=videos, audio=audio, response_audio=response_audio, **data)

    def get_content_as_string(self, **kwargs) -> str:
        import json
        from pydantic import BaseModel
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, **kwargs)

    def add_member_run(self, run_response: Union['TeamRunResponse', RunResponse]) -> None:
        self.member_responses.append(run_response)
        if run_response.images is not None:
            if self.images is None:
                self.images = []
            self.images.extend(run_response.images)
        if run_response.videos is not None:
            if self.videos is None:
                self.videos = []
            self.videos.extend(run_response.videos)
        if run_response.audio is not None:
            if self.audio is None:
                self.audio = []
            self.audio.extend(run_response.audio)
