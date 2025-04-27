import json
from enum import Enum
from time import time
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from agno.media import AudioArtifact, AudioResponse, ImageArtifact, VideoArtifact
from agno.models import Citations, Message, MessageReferences


class RunMessages:
    def __init__(self, messages: List[Message] = None, system_message: Message = None, user_message: Message = None, extra_messages: List[Message] = None):
        self.messages = messages or []
        self.system_message = system_message
        self.user_message = user_message
        self.extra_messages = extra_messages or []

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


class RunResponseExtraData:
    def __init__(self, references: Optional[List[MessageReferences]] = None, add_messages: Optional[List[Message]] = None, reasoning_steps: Optional[List['ReasoningStep']] = None, reasoning_messages: Optional[List[Message]] = None):
        self.references = references
        self.add_messages = add_messages
        self.reasoning_steps = reasoning_steps
        self.reasoning_messages = reasoning_messages

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


class RunResponse:
    def __init__(self, content=None, content_type: str = 'str', thinking: str = None, event: str = RunEvent.run_response.value,
            messages: List[Message] = None, metrics: Dict[str, Any] = None,
            model: str = None, run_id: str = None, agent_id: str = None, session_id: str = None,
            workflow_id: str = None, tools: Optional[List[Dict[str, Any]]] = None,
            formatted_tool_calls: List[str] = None, images: List[ImageArtifact] = None,
            videos: List[VideoArtifact] = None, audio: List[AudioArtifact] = None,
            response_audio: AudioResponse = None, citations: Citations = None,
            extra_data: RunResponseExtraData = None, created_at: int = 0):
        self.content = content
        self.content_type = content_type
        self.thinking = thinking
        self.event = event
        self.messages = messages
        self.metrics = metrics
        self.model = model
        self.run_id = run_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.workflow_id = workflow_id
        self.tools = tools
        self.formatted_tool_calls = formatted_tool_calls
        self.images = images
        self.videos = videos
        self.audio = audio
        self.response_audio = response_audio
        self.citations = citations
        self.extra_data = extra_data
        self.created_at = created_at or int(time())

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in self.__dict__.items()
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
        _dict = self.to_dict()
        return json.dumps(_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResponse':
        messages = data.pop('messages', None)
        messages = [Message.model_validate(message) for message in messages] if messages else None
        return cls(messages=messages, **data)

    def get_content_as_string(self, **kwargs) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, **kwargs)


class TeamRunResponse(RunResponse):
    def __init__(self, member_responses: List[Union['TeamRunResponse', RunResponse]] = None, team_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.member_responses = member_responses or []
        self.team_id = team_id

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in self.__dict__.items()
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
