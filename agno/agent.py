from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Set, Type, Union, cast, overload
import re
import os
import json
import string
import warnings
from uuid import uuid4
from textwrap import dedent
from dataclasses import dataclass, asdict
from collections import ChainMap, defaultdict, deque
from pydantic import BaseModel, ValidationError
from agno.knowledge import AgentKnowledge
from agno.media import Audio, AudioArtifact, AudioResponse, File, Image, ImageArtifact, Video, VideoArtifact
from agno.models import Model, Citations, Message, MessageReferences, MessageMetrics, ModelResponse, ModelResponseEvent, Timer
from agno.storage import Storage, AgentSession
from agno.memory import Memory, SessionSummary, AgentMemory, AgentRun
from agno.reader import Document
from agno.tools import Function, Toolkit
from agno.run import RunEvent, RunResponse, RunResponseExtraData, TeamRunResponse, RunMessages, NextAction, ReasoningStep, ReasoningSteps, get_deepseek_reasoning, get_openai_reasoning, aget_deepseek_reasoning, get_deepseek_reasoning_agent, get_default_reasoning_agent, get_next_action, update_messages_with_reasoning, aget_openai_reasoning, get_openai_reasoning_agent


@dataclass
class SessionMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[dict] = None
    completion_tokens_details: Optional[dict] = None
    additional_metrics: Optional[dict] = None
    time: Optional[float] = None
    time_to_first_token: Optional[float] = None
    timer: Optional[Timer] = None

    def start_timer(self):
        if self.timer is None:
            self.timer = Timer()
        self.timer.start()

    def stop_timer(self, set_time: bool = True):
        if self.timer is not None:
            self.timer.stop()
            if set_time:
                self.time = self.timer.elapsed

    def set_time_to_first_token(self):
        if self.timer is not None:
            self.time_to_first_token = self.timer.elapsed

    def __add__(self, other: Union['SessionMetrics', 'MessageMetrics']) -> 'SessionMetrics':
        result = SessionMetrics(input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens, total_tokens=self.total_tokens + other.total_tokens, prompt_tokens=self.prompt_tokens + other.prompt_tokens, completion_tokens=self.completion_tokens + other.completion_tokens)
        if self.prompt_tokens_details or other.prompt_tokens_details:
            result.prompt_tokens_details = {}
            if self.prompt_tokens_details:
                result.prompt_tokens_details.update(self.prompt_tokens_details)
            if other.prompt_tokens_details:
                for key, value in other.prompt_tokens_details.items():
                    result.prompt_tokens_details[key] = result.prompt_tokens_details.get(key, 0) + value
        if self.completion_tokens_details or other.completion_tokens_details:
            result.completion_tokens_details = {}
            if self.completion_tokens_details:
                result.completion_tokens_details.update(self.completion_tokens_details)
            if other.completion_tokens_details:
                for key, value in other.completion_tokens_details.items():
                    result.completion_tokens_details[key] = result.completion_tokens_details.get(key, 0) + value
        if self.additional_metrics or other.additional_metrics:
            result.additional_metrics = {}
            if self.additional_metrics:
                result.additional_metrics.update(self.additional_metrics)
            if other.additional_metrics:
                result.additional_metrics.update(other.additional_metrics)
        if self.time is not None and other.time is not None:
            result.time = self.time + other.time
        elif self.time is not None:
            result.time = self.time
        elif other.time is not None:
            result.time = other.time
        result.time_to_first_token = self.time_to_first_token or other.time_to_first_token
        return result

    def __radd__(self, other: Union['SessionMetrics', 'MessageMetrics']) -> 'SessionMetrics':
        if other == 0:
            return self
        return self + other


@dataclass(init=False)
class Agent:
    model: Optional[Model] = None
    name: Optional[str] = None
    agent_id: Optional[str] = None
    introduction: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    add_context: bool = False
    resolve_context: bool = True
    memory: Optional[AgentMemory] = None
    add_history_to_messages: bool = False
    num_history_responses: Optional[int] = None
    num_history_runs: int = 3
    knowledge: Optional[AgentKnowledge] = None
    add_references: bool = False
    retriever: Optional[Callable[..., Optional[List[Dict]]]] = None
    references_format: Literal['json', 'yaml'] = 'json'
    storage: Optional[Storage] = None
    extra_data: Optional[Dict[str, Any]] = None
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None
    show_tool_calls: bool = True
    tool_call_limit: Optional[int] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    reasoning: bool = False
    reasoning_model: Optional[Model] = None
    reasoning_agent: Optional['Agent'] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10
    read_chat_history: bool = False
    search_knowledge: bool = True
    update_knowledge: bool = False
    read_tool_call_history: bool = False
    system_message: Optional[Union[str, Callable, Message]] = None
    system_message_role: str = 'system'
    create_default_system_message: bool = True
    description: Optional[str] = None
    goal: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    expected_output: Optional[str] = None
    additional_context: Optional[str] = None
    markdown: bool = False
    add_name_to_instructions: bool = False
    add_datetime_to_instructions: bool = False
    add_state_in_messages: bool = False
    add_messages: Optional[List[Union[Dict, Message]]] = None
    user_message: Optional[Union[List, Dict, str, Callable, Message]] = None
    user_message_role: str = 'user'
    create_default_user_message: bool = True
    retries: int = 0
    delay_between_retries: int = 1
    exponential_backoff: bool = False
    response_model: Optional[Type[BaseModel]] = None
    parse_response: bool = True
    structured_outputs: bool = False
    use_json_mode: bool = False
    save_response_to_file: Optional[str] = None
    stream: Optional[bool] = None
    stream_intermediate_steps: bool = False
    team: Optional[List['Agent']] = None
    team_data: Optional[Dict[str, Any]] = None
    role: Optional[str] = None
    respond_directly: bool = False
    add_transfer_instructions: bool = True
    team_response_separator: str = '\n'
    team_session_id: Optional[str] = None
    team_id: Optional[str] = None
    debug_mode: bool = False
    monitoring: bool = False
    telemetry: bool = True

    def __init__(self, *, model: Optional[Model] = None, name: Optional[str] = None, agent_id: Optional[str] = None, introduction: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, session_name: Optional[str] = None, session_state: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, add_context: bool = False, resolve_context: bool = True, memory: Optional[AgentMemory] = None, add_history_to_messages: bool = False, num_history_responses: Optional[int] = None, num_history_runs: int = 3, knowledge: Optional[AgentKnowledge] = None, add_references: bool = False, retriever: Optional[Callable[..., Optional[List[Dict]]]] = None, references_format: Literal['json', 'yaml'] = 'json', storage: Optional[Storage] = None, extra_data: Optional[Dict[str, Any]] = None, tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None, show_tool_calls: bool = True, tool_call_limit: Optional[int] = None, tool_choice: Optional[Union[str, Dict[str, Any]]] = None, reasoning: bool = False, reasoning_model: Optional[Model] = None, reasoning_agent: Optional['Agent'] = None, reasoning_min_steps: int = 1, reasoning_max_steps: int = 10, read_chat_history: bool = False, search_knowledge: bool = True, update_knowledge: bool = False, read_tool_call_history: bool = False, system_message: Optional[Union[str, Callable, Message]] = None, system_message_role: str = 'system', create_default_system_message: bool = True, description: Optional[str] = None, goal: Optional[str] = None, instructions: Optional[Union[str, List[str], Callable]] = None, expected_output: Optional[str] = None, additional_context: Optional[str] = None, markdown: bool = False, add_name_to_instructions: bool = False, add_datetime_to_instructions: bool = False, add_state_in_messages: bool = False, add_messages: Optional[List[Union[Dict, Message]]] = None, user_message: Optional[Union[List, Dict, str, Callable, Message]] = None, user_message_role: str = 'user', create_default_user_message: bool = True, retries: int = 0, delay_between_retries: int = 1, exponential_backoff: bool = False, response_model: Optional[Type[BaseModel]] = None, parse_response: bool = True, structured_outputs: Optional[bool] = None, use_json_mode: bool = False, save_response_to_file: Optional[str] = None, stream: Optional[bool] = None, stream_intermediate_steps: bool = False, team: Optional[List['Agent']] = None, team_data: Optional[Dict[str, Any]] = None, role: Optional[str] = None, respond_directly: bool = False, add_transfer_instructions: bool = True, team_response_separator: str = '\n', debug_mode: bool = False, monitoring: bool = False, telemetry: bool = True):
        self.model = model
        self.name = name
        self.agent_id = agent_id
        self.introduction = introduction
        self.user_id = user_id
        self.session_id = session_id
        self.session_name = session_name
        self.session_state = session_state
        self.context = context
        self.add_context = add_context
        self.resolve_context = resolve_context
        self.memory = memory
        self.add_history_to_messages = add_history_to_messages
        self.num_history_responses = num_history_responses
        self.num_history_runs = num_history_runs
        if num_history_responses is not None:
            self.num_history_runs = num_history_responses
        self.knowledge = knowledge
        self.add_references = add_references
        self.retriever = retriever
        self.references_format = references_format
        self.storage = storage
        self.extra_data = extra_data
        self.tools = tools
        self.show_tool_calls = show_tool_calls
        self.tool_call_limit = tool_call_limit
        self.tool_choice = tool_choice
        self.reasoning = reasoning
        self.reasoning_model = reasoning_model
        self.reasoning_agent = reasoning_agent
        self.reasoning_min_steps = reasoning_min_steps
        self.reasoning_max_steps = reasoning_max_steps
        self.read_chat_history = read_chat_history
        self.search_knowledge = search_knowledge
        self.update_knowledge = update_knowledge
        self.read_tool_call_history = read_tool_call_history
        self.system_message = system_message
        self.system_message_role = system_message_role
        self.create_default_system_message = create_default_system_message
        self.description = description
        self.goal = goal
        self.instructions = instructions
        self.expected_output = expected_output
        self.additional_context = additional_context
        self.markdown = markdown
        self.add_name_to_instructions = add_name_to_instructions
        self.add_datetime_to_instructions = add_datetime_to_instructions
        self.add_state_in_messages = add_state_in_messages
        self.add_messages = add_messages
        self.user_message = user_message
        self.user_message_role = user_message_role
        self.create_default_user_message = create_default_user_message
        self.retries = retries
        self.delay_between_retries = delay_between_retries
        self.exponential_backoff = exponential_backoff
        self.response_model = response_model
        self.parse_response = parse_response
        if structured_outputs is not None:
            self.structured_outputs = structured_outputs
        self.use_json_mode = use_json_mode
        self.save_response_to_file = save_response_to_file
        self.stream = stream
        self.stream_intermediate_steps = stream_intermediate_steps
        self.team = team
        self.team_data = team_data
        self.role = role
        self.respond_directly = respond_directly
        self.add_transfer_instructions = add_transfer_instructions
        self.team_response_separator = team_response_separator
        self.debug_mode = debug_mode
        self.monitoring = monitoring
        self.telemetry = telemetry
        self.session_metrics: Optional[SessionMetrics] = None
        self.run_id: Optional[str] = None
        self.run_input: Optional[Union[str, List, Dict, Message]] = None
        self.run_messages: Optional[RunMessages] = None
        self.run_response: Optional[RunResponse] = None
        self.images: Optional[List[ImageArtifact]] = None
        self.audio: Optional[List[AudioArtifact]] = None
        self.videos: Optional[List[VideoArtifact]] = None
        self.agent_session: Optional[AgentSession] = None
        self._tools_for_model: Optional[List[Dict]] = None
        self._functions_for_model: Optional[Dict[str, Function]] = None
        self._formatter: Optional[SafeFormatter] = None

    def set_agent_id(self) -> str:
        if self.agent_id is None:
            self.agent_id = str(uuid4())
        print(f'Agent ID: {self.agent_id}')
        return self.agent_id

    def set_session_id(self) -> str:
        if self.session_id is None or self.session_id == '':
            self.session_id = str(uuid4())
        print(f'Session ID: {self.session_id}')
        return self.session_id

    def set_debug(self) -> None:
        if self.debug_mode or os.getenv('AGNO_DEBUG', 'false').lower() == 'true':
            self.debug_mode = True

    def set_storage_mode(self):
        if self.storage is not None:
            if self.storage.mode in ['workflow', 'team']:
                print(f'您不应该以多种模式使用存储。当前模式为 {self.storage.mode}.')
            self.storage.mode = 'agent'

    def set_monitoring(self) -> None:
        monitor_env = os.getenv('AGNO_MONITOR')
        if monitor_env is not None:
            self.monitoring = monitor_env.lower() == 'true'
        telemetry_env = os.getenv('AGNO_TELEMETRY')
        if telemetry_env is not None:
            self.telemetry = telemetry_env.lower() == 'true'

    def initialize_agent(self) -> None:
        self.set_storage_mode()
        self.set_debug()
        self.set_agent_id()
        self.set_session_id()
        if self.memory is None:
            self.memory = AgentMemory()
        if self._formatter is None:
            self._formatter = SafeFormatter()

    @property
    def is_streamable(self) -> bool:
        return self.response_model is None

    @property
    def has_team(self) -> bool:
        return self.team is not None and len(self.team) > 0

    def _run(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: bool = False, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, **kwargs: Any) -> Iterator[RunResponse]:
        """运行代理并生成RunResponse。
        步骤：
        1.让代理人做好运行准备
        2.更新模型并解析上下文
        3.从存储中读取现有会话
        4.准备运行信息
        5.如果启用推理，则说明任务的原因
        6.通过生成RunStarted事件来启动Run
        7.从模型生成响应（包括运行函数调用）
        8.更新RunResponse
        9.更新代理内存
        10.计算会话度量
        11.将会话保存到存储
        12.如果设置了Save_response_to_file，则将输出保存到文件
        """
        self.initialize_agent()
        self.memory = cast(AgentMemory, self.memory)
        self.stream = self.stream or (stream and self.is_streamable)
        self.stream_intermediate_steps = self.stream_intermediate_steps or (stream_intermediate_steps and self.stream)
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)
        print(f'Agent Run Start: {self.run_response.run_id}')
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self.resolve_run_context()
        self.read_from_storage()
        run_messages: RunMessages = self.get_run_messages(message=message, audio=audio, images=images, videos=videos, files=files, messages=messages, **kwargs)
        if len(run_messages.messages) == 0:
            print('没有消息要发送到模型.')
        self.run_messages = run_messages
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self.reason(run_messages=run_messages)
            if self.stream:
                yield from reasoning_generator
            else:
                deque(reasoning_generator, maxlen=0)
        index_of_last_user_message = len(run_messages.messages)
        if self.stream_intermediate_steps:
            yield self.create_run_response('Run started', event=RunEvent.run_started)
        model_response: ModelResponse
        self.model = cast(Model, self.model)
        if self.stream:
            model_response = ModelResponse()
            for model_response_chunk in self.model.response_stream(messages=run_messages.messages):
                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                    if model_response_chunk.content is not None:
                        model_response.content = (model_response.content or '') + model_response_chunk.content
                        self.run_response.content = model_response.content
                    if model_response_chunk.thinking is not None:
                        model_response.thinking = (model_response.thinking or '') + model_response_chunk.thinking
                        self.run_response.thinking = model_response.thinking
                    if model_response_chunk.redacted_thinking is not None:
                        model_response.redacted_thinking = (model_response.redacted_thinking or '') + model_response_chunk.redacted_thinking
                        self.run_response.thinking = model_response.redacted_thinking
                    if model_response_chunk.citations is not None:
                        self.run_response.citations = model_response_chunk.citations
                    if model_response_chunk.content is not None or model_response_chunk.thinking is not None or model_response_chunk.redacted_thinking is not None or model_response_chunk.citations is not None:
                        yield self.create_run_response(content=model_response_chunk.content, thinking=model_response_chunk.thinking, redacted_thinking=model_response_chunk.redacted_thinking, citations=model_response_chunk.citations, created_at=model_response_chunk.created_at)
                    if model_response_chunk.audio is not None:
                        if model_response.audio is None:
                            model_response.audio = AudioResponse(id=str(uuid4()), content='', transcript='')
                        if model_response_chunk.audio.id is not None:
                            model_response.audio.id = model_response_chunk.audio.id
                        if model_response_chunk.audio.content is not None:
                            model_response.audio.content += model_response_chunk.audio.content
                        if model_response_chunk.audio.transcript is not None:
                            model_response.audio.transcript += model_response_chunk.audio.transcript
                        if model_response_chunk.audio.expires_at is not None:
                            model_response.audio.expires_at = model_response_chunk.audio.expires_at
                        if model_response_chunk.audio.mime_type is not None:
                            model_response.audio.mime_type = model_response_chunk.audio.mime_type
                        model_response.audio.sample_rate = model_response_chunk.audio.sample_rate
                        model_response.audio.channels = model_response_chunk.audio.channels
                        self.run_response.response_audio = AudioResponse(id=model_response_chunk.audio.id, content=model_response_chunk.audio.content, transcript=model_response_chunk.audio.transcript, sample_rate=model_response_chunk.audio.sample_rate, channels=model_response_chunk.audio.channels)
                        self.run_response.created_at = model_response_chunk.created_at
                        yield self.run_response
                    if model_response_chunk.image is not None:
                        self.add_image(model_response_chunk.image)
                        yield self.run_response
                elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                    tool_calls_list = model_response_chunk.tool_calls
                    if tool_calls_list is not None:
                        if self.run_response.tools is None:
                            self.run_response.tools = tool_calls_list
                        else:
                            self.run_response.tools.extend(tool_calls_list)
                        self.run_response.formatted_tool_calls = format_tool_calls(self.run_response.tools)
                    if self.stream_intermediate_steps:
                        yield self.create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_started)
                elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                    tool_calls_list = model_response_chunk.tool_calls
                    if tool_calls_list is not None:
                        if self.run_response.tools:
                            tool_call_index_map = {tc['tool_call_id']: i for i, tc in enumerate(self.run_response.tools) if tc.get('tool_call_id')}
                            for tool_call_dict in tool_calls_list:
                                tool_call_id = tool_call_dict.get('tool_call_id')
                                index = tool_call_index_map.get(tool_call_id)
                                if index is not None:
                                    self.run_response.tools[index] = tool_call_dict
                        else:
                            self.run_response.tools = tool_calls_list
                    if self.stream_intermediate_steps:
                        yield self.create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_completed)
        else:
            model_response = self.model.response(messages=run_messages.messages)
            if model_response.tool_calls:
                self.run_response.formatted_tool_calls = format_tool_calls(model_response.tool_calls)
            if self.response_model is not None and model_response.parsed is not None:
                if self.model.structured_outputs:
                    self.run_response.content = model_response.parsed
                    self.run_response.content_type = self.response_model.__name__
            else:
                self.run_response.content = model_response.content
            if model_response.thinking is not None:
                self.run_response.thinking = model_response.thinking
            if model_response.redacted_thinking is not None:
                if self.run_response.thinking is None:
                    self.run_response.thinking = model_response.redacted_thinking
                else:
                    self.run_response.thinking += model_response.redacted_thinking
            if model_response.citations is not None:
                self.run_response.citations = model_response.citations
            if model_response.tool_calls is not None:
                if self.run_response.tools is None:
                    self.run_response.tools = model_response.tool_calls
                else:
                    self.run_response.tools.extend(model_response.tool_calls)
            if model_response.audio is not None:
                self.run_response.response_audio = model_response.audio
            if model_response.image is not None:
                self.add_image(model_response.image)
            self.run_response.messages = run_messages.messages
            self.run_response.created_at = model_response.created_at
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        self.run_response.messages = messages_for_run_response
        self.run_response.metrics = self.aggregate_metrics_from_messages(messages_for_run_response)
        if self.stream and model_response.audio is not None:
            self.run_response.response_audio = model_response.audio
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role=self.system_message_role)
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        if self.stream_intermediate_steps:
            yield self.create_run_response(content='Memory updated', event=RunEvent.updating_memory)
        agent_run = AgentRun(response=self.run_response)
        agent_run.message = run_messages.user_message
        if self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            self.memory.update_memory(input=run_messages.user_message.get_content_string())
        if messages is not None and len(messages) > 0:
            for _im in messages:
                mp = None
                if isinstance(_im, Message):
                    mp = _im
                elif isinstance(_im, dict):
                    try:
                        mp = Message(**_im)
                    except Exception as e:
                        print(f'消息验证失败: {e}')
                else:
                    print(f'不支持的消息类型: {type(_im)}')
                    continue
                if mp:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(mp)
                    if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        self.memory.update_memory(input=mp.get_content_string())
                else:
                    print('无法将消息添加到内存中')
        self.memory.add_run(agent_run)
        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            self.memory.update_summary()
        self.session_metrics = self.calculate_session_metrics(self.memory.messages)
        self.write_to_storage()
        self.save_run_response_to_file(message=message)
        if message is not None:
            if isinstance(message, str):
                self.run_input = message
            elif isinstance(message, Message):
                self.run_input = message.to_dict()
            else:
                self.run_input = message
        elif messages is not None:
            self.run_input = [m.to_dict() if isinstance(m, Message) else m for m in messages]
        self._log_agent_run()
        print(f'Agent Run End: {self.run_response.run_id}')
        if self.stream_intermediate_steps:
            yield self.create_run_response(content=self.run_response.content, event=RunEvent.run_completed)
        if not self.stream:
            yield self.run_response

    @overload
    def run(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: Literal[False] = False, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, retries: Optional[int] = None, **kwargs: Any) -> RunResponse: ...

    @overload
    def run(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: Literal[True] = True, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, retries: Optional[int] = None, **kwargs: Any) -> Iterator[RunResponse]: ...

    def run(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: Optional[bool] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, retries: Optional[int] = None, **kwargs: Any) -> Union[RunResponse, Iterator[RunResponse]]:
        if retries is None:
            retries = self.retries
        if stream is None:
            stream = False if self.stream is None else self.stream
        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            try:
                if self.response_model is not None and self.parse_response:
                    print('Setting stream=False as response_model is set')
                    self.stream = False
                    run_response: RunResponse = next(self._run(message=message, stream=False, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs))
                    if isinstance(run_response.content, self.response_model):
                        return run_response
                    if isinstance(run_response.content, str):
                        try:
                            structured_output = parse_response_model_str(run_response.content, self.response_model)
                            if structured_output is not None:
                                run_response.content = structured_output
                                run_response.content_type = self.response_model.__name__
                                if self.run_response is not None:
                                    self.run_response.content = structured_output
                                    self.run_response.content_type = self.response_model.__name__
                            else:
                                print('无法将响应转换为response_model')
                        except Exception as e:
                            print(f'无法将响应转换为输出模型: {e}')
                    else:
                        print('出了点问题。运行响应内容不是字符串')
                    return run_response
                else:
                    if stream and self.is_streamable:
                        resp = self._run(message=message, stream=True, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
                        return resp
                    else:
                        resp = self._run(message=message, stream=False, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
                        return next(resp)
            except ModelProviderError as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
                last_exception = e
                if attempt < num_attempts - 1:
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time
                    time.sleep(delay)
            except KeyboardInterrupt:
                cancelled_response = RunResponse(run_id=self.run_id or str(uuid4()), session_id=self.session_id, agent_id=self.agent_id, content='Operation cancelled by user', event=RunEvent.run_cancelled)
                return cancelled_response
        if last_exception is not None:
            print(f'尝试{num_attempts}次后失败。上次错误使用 {last_exception.model_name}({last_exception.model_id})')
            raise last_exception
        else:
            raise Exception(f'{num_attempts}次后失败')

    async def _arun(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: bool = False, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, **kwargs: Any) -> AsyncIterator[RunResponse]:
        """运行代理并生成RunResponse。
        步骤：
        1.让代理人做好跑步准备
        2.更新模型并解析上下文
        3.从存储中读取现有会话
        4.准备跑步信息
        5.如果启用推理，则说明任务的原因
        6.通过生成RunStarted事件来启动Run
        7.从模型生成响应（包括运行函数调用）
        8.更新RunResponse
        9.更新代理内存
        10.计算会话度量
        11.将会话保存到存储
        12.如果设置了Save_response_to_file，则将输出保存到文件
        """
        self.initialize_agent()
        self.memory = cast(AgentMemory, self.memory)
        self.stream = self.stream or (stream and self.is_streamable)
        self.stream_intermediate_steps = self.stream_intermediate_steps or (stream_intermediate_steps and self.stream)
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)
        print(f'Async Agent Run Start: {self.run_response.run_id}')
        self.update_model(async_mode=True)
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self.resolve_run_context()
        self.read_from_storage()
        run_messages: RunMessages = self.get_run_messages(message=message, audio=audio, images=images, videos=videos, files=files, messages=messages, **kwargs)
        if len(run_messages.messages) == 0:
            print('没有消息要发送到模型。')
        self.run_messages = run_messages
        if self.reasoning or self.reasoning_model is not None:
            areason_generator = self.areason(run_messages=run_messages)
            if self.stream:
                async for item in areason_generator:
                    yield item
            else:
                async for _ in areason_generator:
                    pass
        index_of_last_user_message = len(run_messages.messages)
        if self.stream_intermediate_steps:
            yield self.create_run_response('Run started', event=RunEvent.run_started)
        model_response: ModelResponse
        self.model = cast(Model, self.model)
        if stream and self.is_streamable:
            model_response = ModelResponse(content='')
            model_response_stream = self.model.aresponse_stream(messages=run_messages.messages)
            async for model_response_chunk in model_response_stream:
                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                    if model_response_chunk.content is not None:
                        model_response.content = (model_response.content or '') + model_response_chunk.content
                        self.run_response.content = model_response.content
                    if model_response_chunk.thinking is not None:
                        model_response.thinking = (model_response.thinking or '') + model_response_chunk.thinking
                        self.run_response.thinking = model_response.thinking
                    if model_response_chunk.redacted_thinking is not None:
                        model_response.redacted_thinking = (model_response.redacted_thinking or '') + model_response_chunk.redacted_thinking
                        self.run_response.thinking = model_response.redacted_thinking
                    if model_response_chunk.citations is not None:
                        self.run_response.citations = model_response_chunk.citations
                    if model_response_chunk.content is not None or model_response_chunk.thinking is not None or model_response_chunk.redacted_thinking is not None or model_response_chunk.citations is not None:
                        yield self.create_run_response(content=model_response_chunk.content, thinking=model_response_chunk.thinking, redacted_thinking=model_response_chunk.redacted_thinking, citations=model_response_chunk.citations, created_at=model_response_chunk.created_at)
                    if model_response_chunk.audio is not None:
                        if model_response.audio is None:
                            model_response.audio = AudioResponse(id=str(uuid4()), content='', transcript='')
                        if model_response_chunk.audio.id is not None:
                            model_response.audio.id = model_response_chunk.audio.id
                        if model_response_chunk.audio.content is not None:
                            model_response.audio.content += model_response_chunk.audio.content
                        if model_response_chunk.audio.transcript is not None:
                            model_response.audio.transcript += model_response_chunk.audio.transcript
                        if model_response_chunk.audio.expires_at is not None:
                            model_response.audio.expires_at = model_response_chunk.audio.expires_at
                        if model_response_chunk.audio.mime_type is not None:
                            model_response.audio.mime_type = model_response_chunk.audio.mime_type
                        model_response.audio.sample_rate = model_response_chunk.audio.sample_rate
                        model_response.audio.channels = model_response_chunk.audio.channels
                        self.run_response.response_audio = AudioResponse(id=model_response_chunk.audio.id, content=model_response_chunk.audio.content, transcript=model_response_chunk.audio.transcript, sample_rate=model_response_chunk.audio.sample_rate, channels=model_response_chunk.audio.channels)
                        self.run_response.created_at = model_response_chunk.created_at
                        yield self.run_response
                    if model_response_chunk.image is not None:
                        self.add_image(model_response_chunk.image)
                        yield self.run_response
                elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                    tool_calls_list = model_response_chunk.tool_calls
                    if tool_calls_list is not None:
                        if self.run_response.tools is None:
                            self.run_response.tools = tool_calls_list
                        else:
                            self.run_response.tools.extend(tool_calls_list)
                        self.run_response.formatted_tool_calls = format_tool_calls(self.run_response.tools)
                    if self.stream_intermediate_steps:
                        yield self.create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_started)
                elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                    tool_calls_list = model_response_chunk.tool_calls
                    if tool_calls_list is not None:
                        if self.run_response.tools:
                            tool_call_index_map = {tc['tool_call_id']: i
                                for i, tc in enumerate(self.run_response.tools)
                                if tc.get('tool_call_id') is not None}
                            for tool_call_dict in tool_calls_list:
                                tool_call_id = (tool_call_dict['tool_call_id'] if 'tool_call_id' in tool_call_dict else None)
                                index = tool_call_index_map.get(tool_call_id)
                                if index is not None:
                                    self.run_response.tools[index] = tool_call_dict
                        else:
                            self.run_response.tools = tool_calls_list
                    if self.stream_intermediate_steps:
                        yield self.create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_completed)
        else:
            model_response = await self.model.aresponse(messages=run_messages.messages)
            if model_response.tool_calls:
                self.run_response.formatted_tool_calls = format_tool_calls(model_response.tool_calls)
            if self.response_model is not None and model_response.parsed is not None:
                if self.model.structured_outputs:
                    self.run_response.content = model_response.parsed
                    self.run_response.content_type = self.response_model.__name__
            else:
                self.run_response.content = model_response.content
            if model_response.thinking is not None:
                self.run_response.thinking = model_response.thinking
            if model_response.redacted_thinking is not None:
                if self.run_response.thinking is None:
                    self.run_response.thinking = model_response.redacted_thinking
                else:
                    self.run_response.thinking += model_response.redacted_thinking
            if model_response.citations is not None:
                self.run_response.citations = model_response.citations
            if model_response.tool_calls is not None:
                if self.run_response.tools is None:
                    self.run_response.tools = model_response.tool_calls
                else:
                    self.run_response.tools.extend(model_response.tool_calls)
            if model_response.audio is not None:
                self.run_response.response_audio = model_response.audio
            if model_response.image is not None:
                self.add_image(model_response.image)
            self.run_response.messages = run_messages.messages
            self.run_response.created_at = model_response.created_at
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        self.run_response.messages = messages_for_run_response
        self.run_response.metrics = self.aggregate_metrics_from_messages(messages_for_run_response)
        if self.stream and model_response.audio is not None:
            self.run_response.response_audio = model_response.audio
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role=self.system_message_role)
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        if self.stream_intermediate_steps:
            yield self.create_run_response(content='Memory updated', event=RunEvent.updating_memory)
        agent_run = AgentRun(response=self.run_response)
        agent_run.message = run_messages.user_message
        if self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            await self.memory.aupdate_memory(input=run_messages.user_message.get_content_string())
        if messages is not None and len(messages) > 0:
            for _im in messages:
                mp = None
                if isinstance(_im, Message):
                    mp = _im
                elif isinstance(_im, dict):
                    try:
                        mp = Message(**_im)
                    except Exception as e:
                        print(f'验证消息失败: {e}')
                else:
                    print(f'不支持的消息类型: {type(_im)}')
                    continue
                if mp:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(mp)
                    if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        await self.memory.aupdate_memory(input=mp.get_content_string())
                else:
                    print('无法将消息添加到内存中')
        self.memory.add_run(agent_run)
        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            await self.memory.aupdate_summary()
        self.session_metrics = self.calculate_session_metrics(self.memory.messages)
        self.write_to_storage()
        self.save_run_response_to_file(message=message)
        if message is not None:
            if isinstance(message, str):
                self.run_input = message
            elif isinstance(message, Message):
                self.run_input = message.to_dict()
            else:
                self.run_input = message
        elif messages is not None:
            self.run_input = [m.to_dict() if isinstance(m, Message) else m for m in messages]
        await self._alog_agent_run()
        print(f'Agent Run End: {self.run_response.run_id}')
        if self.stream_intermediate_steps:
            yield self.create_run_response(content=self.run_response.content, event=RunEvent.run_completed)
        if not self.stream:
            yield self.run_response

    async def arun(self, message: Optional[Union[str, List, Dict, Message]] = None, *, stream: Optional[bool] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, stream_intermediate_steps: bool = False, retries: Optional[int] = None, **kwargs: Any) -> Any:
        if retries is None:
            retries = self.retries
        if stream is None:
            stream = False if self.stream is None else self.stream
        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            print(f'Attempt {attempt + 1}/{num_attempts}')
            try:
                if self.response_model is not None and self.parse_response:
                    print('Setting stream=False as response_model is set')
                    run_response = await self._arun(message=message, stream=False, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs).__anext__()
                    if isinstance(run_response.content, self.response_model):
                        return run_response
                    if isinstance(run_response.content, str):
                        try:
                            structured_output = parse_response_model_str(run_response.content, self.response_model)
                            if structured_output is not None:
                                run_response.content = structured_output
                                run_response.content_type = self.response_model.__name__
                                if self.run_response is not None:
                                    self.run_response.content = structured_output
                                    self.run_response.content_type = self.response_model.__name__
                            else:
                                print('无法将响应转换为response_model')
                        except Exception as e:
                            print(f'无法将响应转换为输出模型: {e}')
                    else:
                        print('出了点问题。运行响应内容不是字符串')
                    return run_response
                else:
                    if stream and self.is_streamable:
                        resp = self._arun(message=message, stream=True, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
                        return resp
                    else:
                        resp = self._arun(message=message, stream=False, audio=audio, images=images, videos=videos, files=files, messages=messages, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
                        return await resp.__anext__()
            except ModelProviderError as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
                last_exception = e
                if attempt < num_attempts - 1:
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time
                    time.sleep(delay)
            except KeyboardInterrupt:
                return RunResponse(run_id=self.run_id or str(uuid4()), session_id=self.session_id, agent_id=self.agent_id, content='Operation cancelled by user', event=RunEvent.run_cancelled)
        if last_exception is not None:
            print(f'Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})')
            raise last_exception
        else:
            raise Exception(f'Failed after {num_attempts} attempts.')

    def create_run_response(self, content: Optional[Any] = None, *, thinking: Optional[str] = None, redacted_thinking: Optional[str] = None, event: RunEvent = RunEvent.run_response, content_type: Optional[str] = None, created_at: Optional[int] = None, citations: Optional[Citations] = None) -> RunResponse:
        self.run_response = cast(RunResponse, self.run_response)
        thinking_combined = (thinking or '') + (redacted_thinking or '')
        rr = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id, content=content, thinking=thinking_combined if thinking_combined else None, tools=self.run_response.tools, audio=self.run_response.audio, images=self.run_response.images, videos=self.run_response.videos, citations=citations or self.run_response.citations, response_audio=self.run_response.response_audio, model=self.run_response.model, messages=self.run_response.messages, extra_data=self.run_response.extra_data, event=event.value)
        if content_type is not None:
            rr.content_type = content_type
        if created_at is not None:
            rr.created_at = created_at
        return rr

    def get_tools(self, async_mode: bool = False) -> Optional[List[Union[Toolkit, Callable, Function, Dict]]]:
        self.memory = cast(AgentMemory, self.memory)
        agent_tools: List[Union[Toolkit, Callable, Function, Dict]] = []
        if self.tools is not None:
            for tool in self.tools:
                agent_tools.append(tool)
        if self.read_chat_history:
            agent_tools.append(self.get_chat_history)
        if self.read_tool_call_history:
            agent_tools.append(self.get_tool_call_history)
        if self.memory and self.memory.create_user_memories:
            agent_tools.append(self.update_memory)
        if self.knowledge is not None or self.retriever is not None:
            if self.search_knowledge:
                if async_mode:
                    agent_tools.append(self.async_search_knowledge_base)
                else:
                    agent_tools.append(self.search_knowledge_base)
            if self.update_knowledge:
                agent_tools.append(self.add_to_knowledge)
        if self.has_team and self.team is not None:
            for agent_index, agent in enumerate(self.team):
                agent_tools.append(self.get_transfer_function(agent, agent_index))
        return agent_tools

    def add_tools_to_model(self, model: Model, async_mode: bool = False) -> None:
        if self._functions_for_model is None or self._tools_for_model is None:
            agent_tools = self.get_tools(async_mode=async_mode)
            if agent_tools is not None and len(agent_tools) > 0:
                print('Processing tools for model')
                strict = False
                if self.response_model is not None and (self.structured_outputs or (not self.use_json_mode)) and model.supports_native_structured_outputs:
                    strict = True
                self._tools_for_model = []
                self._functions_for_model = {}
                for tool in agent_tools:
                    if isinstance(tool, Dict):
                        self._tools_for_model.append(tool)
                        print(f'Included builtin tool {tool}')
                    elif isinstance(tool, Toolkit):
                        for name, func in tool.functions.items():
                            if name not in self._functions_for_model:
                                func._agent = self
                                func.process_entrypoint(strict=strict)
                                if strict:
                                    func.strict = True
                                self._functions_for_model[name] = func
                                self._tools_for_model.append({'type': 'function', 'function': func.to_dict()})
                                print(f'Included function {name} from {tool.name}')
                    elif isinstance(tool, Function):
                        if tool.name not in self._functions_for_model:
                            tool._agent = self
                            tool.process_entrypoint(strict=strict)
                            if strict and tool.strict is None:
                                tool.strict = True
                            self._functions_for_model[tool.name] = tool
                            self._tools_for_model.append({'type': 'function', 'function': tool.to_dict()})
                            print(f'Included function {tool.name}')
                    elif callable(tool):
                        try:
                            function_name = tool.__name__
                            if function_name not in self._functions_for_model:
                                func = Function.from_callable(tool, strict=strict)
                                func._agent = self
                                if strict:
                                    func.strict = True
                                self._functions_for_model[func.name] = func
                                self._tools_for_model.append({'type': 'function', 'function': func.to_dict()})
                                print(f'Included function {func.name}')
                        except Exception as e:
                            print(f'Could not add function {tool}: {e}')
                model.set_tools(tools=self._tools_for_model)
                model.set_functions(functions=self._functions_for_model)

    def update_model(self, async_mode: bool = False) -> None:
        if self.model is None:
            try:
                from agno.ollama import Ollama
            except ModuleNotFoundError as e:
                print(e)
                print('Agno代理使用“openai”作为默认模型提供程序。')
                exit(1)
            self.model = Ollama()
        if self.response_model is None:
            self.model.response_format = None
        else:
            json_response_format = {'type': 'json_object'}
            if self.model.supports_native_structured_outputs:
                if (not self.use_json_mode) or self.structured_outputs:
                    print('将Model.response_format设置为Agent.response_Model')
                    self.model.response_format = self.response_model
                    self.model.structured_outputs = True
                else:
                    print('模型支持本机结构化输出，但未启用。改用JSON模式。')
                    self.model.response_format = json_response_format
                    self.model.structured_outputs = False
            elif self.model.supports_json_schema_outputs:
                if self.use_json_mode or not self.structured_outputs:
                    print('将Model.response_format设置为JSON响应模式')
                    self.model.response_format = {'type': 'json_schema', 'json_schema': {'name': self.response_model.__name__, 'schema': self.response_model.model_json_schema()}}
                else:
                    self.model.response_format = None
                self.model.structured_outputs = False
            else:
                print('模型不支持结构化或JSON模式输出')
                self.model.response_format = (json_response_format if (self.use_json_mode or not self.structured_outputs) else None)
                self.model.structured_outputs = False
        self.add_tools_to_model(model=self.model, async_mode=async_mode)
        if self.show_tool_calls is not None:
            self.model.show_tool_calls = self.show_tool_calls
        if self.tool_choice is not None:
            self.model.tool_choice = self.tool_choice
        if self.tool_call_limit is not None:
            self.model.tool_call_limit = self.tool_call_limit

    def resolve_run_context(self) -> None:
        from inspect import signature
        print('Resolving context')
        if self.context is not None:
            if isinstance(self.context, dict):
                for ctx_key, ctx_value in self.context.items():
                    if callable(ctx_value):
                        try:
                            sig = signature(ctx_value)
                            if 'agent' in sig.parameters:
                                resolved_ctx_value = ctx_value(agent=self)
                            else:
                                resolved_ctx_value = ctx_value()
                            if resolved_ctx_value is not None:
                                self.context[ctx_key] = resolved_ctx_value
                        except Exception as e:
                            print(f'Failed to resolve context for {ctx_key}: {e}')
                    else:
                        self.context[ctx_key] = ctx_value
            else:
                print('Context is not a dict')

    def load_user_memories(self) -> None:
        self.memory = cast(AgentMemory, self.memory)
        if self.memory and self.memory.create_user_memories:
            if self.user_id is not None and self.memory.user_id is None:
                self.memory.user_id = self.user_id
            self.memory.load_user_memories()
            if self.user_id is not None:
                print(f'Memories loaded for user: {self.user_id}')
            else:
                print('Memories loaded')

    def get_agent_data(self) -> Dict[str, Any]:
        agent_data: Dict[str, Any] = {}
        if self.name is not None:
            agent_data['name'] = self.name
        if self.agent_id is not None:
            agent_data['agent_id'] = self.agent_id
        if self.model is not None:
            agent_data['model'] = self.model.to_dict()
        return agent_data

    def get_session_data(self) -> Dict[str, Any]:
        session_data: Dict[str, Any] = {}
        if self.session_name is not None:
            session_data['session_name'] = self.session_name
        if self.session_state is not None and len(self.session_state) > 0:
            session_data['session_state'] = self.session_state
        if self.session_metrics is not None:
            session_data['session_metrics'] = asdict(self.session_metrics) if self.session_metrics is not None else None
        if self.team_data is not None:
            session_data['team_data'] = self.team_data
        if self.images is not None:
            session_data['images'] = [img.model_dump() for img in self.images]
        if self.videos is not None:
            session_data['videos'] = [vid.model_dump() for vid in self.videos]
        if self.audio is not None:
            session_data['audio'] = [aud.model_dump() for aud in self.audio]
        return session_data

    def get_agent_session(self) -> AgentSession:
        from time import time
        self.memory = cast(AgentMemory, self.memory)
        self.session_id = cast(str, self.session_id)
        self.team_session_id = cast(str, self.team_session_id)
        self.agent_id = cast(str, self.agent_id)
        return AgentSession(session_id=self.session_id, agent_id=self.agent_id, user_id=self.user_id, team_session_id=self.team_session_id, memory=self.memory.to_dict() if self.memory is not None else None, agent_data=self.get_agent_data(), session_data=self.get_session_data(), extra_data=self.extra_data, created_at=int(time()))

    def load_agent_session(self, session: AgentSession):
        if self.agent_id is None and session.agent_id is not None:
            self.agent_id = session.agent_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if session.agent_data is not None:
            if self.name is None and 'name' in session.agent_data:
                self.name = session.agent_data.get('name')
        if session.session_data is not None:
            if self.session_name is None and 'session_name' in session.session_data:
                self.session_name = session.session_data.get('session_name')
            if 'session_state' in session.session_data:
                session_state_from_db = session.session_data.get('session_state')
                if session_state_from_db is not None and isinstance(session_state_from_db, dict) and len(session_state_from_db) > 0:
                    if self.session_state is not None and len(self.session_state) > 0:
                        merge_dictionaries(session_state_from_db, self.session_state)
                    self.session_state = session_state_from_db
            if 'session_metrics' in session.session_data:
                session_metrics_from_db = session.session_data.get('session_metrics')
                if session_metrics_from_db is not None and isinstance(session_metrics_from_db, dict):
                    self.session_metrics = SessionMetrics(**session_metrics_from_db)
            if 'images' in session.session_data:
                images_from_db = session.session_data.get('images')
                if images_from_db is not None and isinstance(images_from_db, list):
                    if self.images is None:
                        self.images = []
                    self.images.extend([ImageArtifact.model_validate(img) for img in images_from_db])
            if 'videos' in session.session_data:
                videos_from_db = session.session_data.get('videos')
                if videos_from_db is not None and isinstance(videos_from_db, list):
                    if self.videos is None:
                        self.videos = []
                    self.videos.extend([VideoArtifact.model_validate(vid) for vid in videos_from_db])
            if 'audio' in session.session_data:
                audio_from_db = session.session_data.get('audio')
                if audio_from_db is not None and isinstance(audio_from_db, list):
                    if self.audio is None:
                        self.audio = []
                    self.audio.extend([AudioArtifact.model_validate(aud) for aud in audio_from_db])
        if session.extra_data is not None:
            if self.extra_data is not None:
                merge_dictionaries(session.extra_data, self.extra_data)
            self.extra_data = session.extra_data
        if self.memory is None:
            self.memory = session.memory
        if not isinstance(self.memory, AgentMemory):
            if isinstance(self.memory, dict):
                self.memory = AgentMemory(**self.memory)
            else:
                raise TypeError(f'预期内存为dict或AgentMemory，但实际得到{type(self.memory)}')
        if session.memory is not None:
            try:
                if 'runs' in session.memory:
                    try:
                        self.memory.runs = [AgentRun.model_validate(m) for m in session.memory['runs']]
                    except Exception as e:
                        print(f'无法从内存中加载运行: {e}')
                if 'messages' in session.memory:
                    try:
                        self.memory.messages = [Message.model_validate(m) for m in session.memory['messages']]
                    except Exception as e:
                        print(f'无法从内存中加载消息: {e}')
                if 'summary' in session.memory:
                    try:
                        self.memory.summary = SessionSummary.model_validate(session.memory['summary'])
                    except Exception as e:
                        print(f'无法从内存中加载会话摘要: {e}')
                if 'memories' in session.memory:
                    try:
                        self.memory.memories = [Memory.model_validate(m) for m in session.memory['memories']]
                    except Exception as e:
                        print(f'加载用户内存失败: {e}')
            except Exception as e:
                print(f'未能加载AgentMemory: {e}')
        print(f'-*- AgentSession loaded: {session.session_id}')

    def read_from_storage(self) -> Optional[AgentSession]:
        if self.storage is not None and self.session_id is not None:
            self.agent_session = cast(AgentSession, self.storage.read(session_id=self.session_id))
            if self.agent_session is not None:
                self.load_agent_session(session=self.agent_session)
            self.load_user_memories()
        return self.agent_session

    def write_to_storage(self) -> Optional[AgentSession]:
        if self.storage is not None:
            self.agent_session = cast(AgentSession, self.storage.upsert(session=self.get_agent_session()))
        return self.agent_session

    def add_introduction(self, introduction: str) -> None:
        self.memory = cast(AgentMemory, self.memory)
        if introduction is not None:
            if len(self.memory.runs) == 0:
                self.memory.add_run(AgentRun(response=RunResponse(content=introduction, messages=[Message(role=self.model.assistant_message_role, content=introduction)])))

    def load_session(self, force: bool = False) -> Optional[str]:
        if self.agent_session is not None and not force:
            if self.session_id is not None and self.agent_session.session_id == self.session_id:
                return self.agent_session.session_id
        if self.storage is not None:
            print(f'Reading AgentSession: {self.session_id}')
            self.read_from_storage()
            if self.agent_session is None:
                print('-*- Creating new AgentSession')
                if self.agent_id is None or self.session_id is None:
                    self.initialize_agent()
                if self.introduction is not None:
                    self.add_introduction(self.introduction)
                self.write_to_storage()
                if self.agent_session is None:
                    raise Exception('在存储中创建新代理会话失败')
                print(f'-*- Created AgentSession: {self.agent_session.session_id}')
                self._log_agent_session()
        return self.session_id

    def new_session(self) -> None:
        self.agent_session = None
        if self.model is not None:
            self.model.clear()
        if self.memory is not None:
            self.memory.clear()
        self.session_id = str(uuid4())
        self.load_session(force=True)

    def get_json_output_prompt(self) -> str:
        import json
        json_output_prompt = '以包含以下字段的JSON格式提供输出:'
        if self.response_model is not None:
            if isinstance(self.response_model, str):
                json_output_prompt += '\n<json_fields>'
                json_output_prompt += f'\n{self.response_model}'
                json_output_prompt += '\n</json_fields>'
            elif isinstance(self.response_model, list):
                json_output_prompt += '\n<json_fields>'
                json_output_prompt += f'\n{json.dumps(self.response_model)}'
                json_output_prompt += '\n</json_fields>'
            elif issubclass(self.response_model, BaseModel):
                json_schema = self.response_model.model_json_schema()
                if json_schema is not None:
                    response_model_properties = {}
                    json_schema_properties = json_schema.get('properties')
                    if json_schema_properties is not None:
                        for field_name, field_properties in json_schema_properties.items():
                            formatted_field_properties = {prop_name: prop_value
                                for prop_name, prop_value in field_properties.items()
                                if prop_name != 'title'}
                            if 'allOf' in formatted_field_properties:
                                ref = formatted_field_properties['allOf'][0].get('$ref', '')
                                if ref.startswith('#/$defs/'):
                                    enum_name = ref.split('/')[-1]
                                    formatted_field_properties['enum_type'] = enum_name
                            response_model_properties[field_name] = formatted_field_properties
                    json_schema_defs = json_schema.get('$defs')
                    if json_schema_defs is not None:
                        response_model_properties['$defs'] = {}
                        for def_name, def_properties in json_schema_defs.items():
                            if 'enum' in def_properties:
                                response_model_properties['$defs'][def_name] = {'type': 'string', 'enum': def_properties['enum'], 'description': def_properties.get('description', '')}
                            else:
                                def_fields = def_properties.get('properties')
                                formatted_def_properties = {}
                                if def_fields is not None:
                                    for field_name, field_properties in def_fields.items():
                                        formatted_field_properties = {prop_name: prop_value
                                            for prop_name, prop_value in field_properties.items()
                                            if prop_name != 'title'}
                                        formatted_def_properties[field_name] = formatted_field_properties
                                if len(formatted_def_properties) > 0:
                                    response_model_properties['$defs'][def_name] = formatted_def_properties
                    if len(response_model_properties) > 0:
                        json_output_prompt += '\n<json_fields>'
                        json_output_prompt += (f'\n{json.dumps([key for key in response_model_properties.keys() if key != "$defs"])}')
                        json_output_prompt += '\n</json_fields>'
                        json_output_prompt += '\n\nHere are the properties for each field:'
                        json_output_prompt += '\n<json_field_properties>'
                        json_output_prompt += f'\n{json.dumps(response_model_properties, indent=2)}'
                        json_output_prompt += '\n</json_field_properties>'
            else:
                print(f'无法为构建json架构{self.response_model}')
        else:
            json_output_prompt += 'Provide the output as JSON.'
        json_output_prompt += '\nStart your response with `{` and end it with `}`.'
        json_output_prompt += '\nYour output will be passed to json.loads() to convert it to a Python object.'
        json_output_prompt += '\nMake sure it only contains valid JSON.'
        return json_output_prompt

    def format_message_with_state_variables(self, msg: Any) -> Any:
        if not isinstance(msg, str):
            return msg
        format_variables = ChainMap(self.session_state or {}, self.context or {}, self.extra_data or {}, {'user_id': self.user_id} if self.user_id is not None else {})
        return self._formatter.format(msg, **format_variables)

    def get_system_message(self) -> Optional[Message]:
        self.memory = cast(AgentMemory, self.memory)
        if self.system_message is not None:
            if isinstance(self.system_message, Message):
                return self.system_message
            sys_message_content: str = ''
            if isinstance(self.system_message, str):
                sys_message_content = self.system_message
            elif callable(self.system_message):
                sys_message_content = self.system_message(agent=self)
                if not isinstance(sys_message_content, str):
                    raise Exception('system_message must return a string')
            if self.add_state_in_messages:
                sys_message_content = self.format_message_with_state_variables(sys_message_content)
            if self.response_model is not None and self.model and (self.model.supports_native_structured_outputs and (self.use_json_mode or self.structured_outputs is False)):
                sys_message_content += f'\n{self.get_json_output_prompt()}'
            return Message(role=self.system_message_role, content=sys_message_content)
        if not self.create_default_system_message:
            return None
        if self.model is None:
            raise Exception('model not set')
        instructions: List[str] = []
        if self.instructions is not None:
            _instructions = self.instructions
            if callable(self.instructions):
                _instructions = self.instructions(agent=self)
            if isinstance(_instructions, str):
                instructions.append(_instructions)
            elif isinstance(_instructions, list):
                instructions.extend(_instructions)
        _model_instructions = self.model.get_instructions_for_model()
        if _model_instructions is not None:
            instructions.extend(_model_instructions)
        additional_information: List[str] = []
        if self.markdown and self.response_model is None:
            additional_information.append('Use markdown to format your answers.')
        if self.add_datetime_to_instructions:
            from datetime import datetime
            additional_information.append(f'The current time is {datetime.now()}')
        if self.name is not None and self.add_name_to_instructions:
            additional_information.append(f'Your name is: {self.name}.')
        system_message_content: str = ''
        if self.description is not None:
            system_message_content += f'{self.description}\n\n'
        if self.goal is not None:
            system_message_content += f'<your_goal>\n{self.goal}\n</your_goal>\n\n'
        if self.role is not None:
            system_message_content += f'<your_role>\n{self.role}\n</your_role>\n\n'
        if self.has_team and self.add_transfer_instructions:
            system_message_content += ('<agent_team>\n'
                '您是AI代理团队的负责人：\n-您可以直接响应，也可以将任务转移给团队中的其他代理，具体取决于他们可用的工具。\n-如果将任务转移给另一个代理，请确保包括：\n-task_description（str）：任务的清晰描述。\n-expected_output（str）：预期输出。\n-additional_information（str）：有助于代理完成任务的其他信息。\n-在响应用户之前，您必须始终验证其他代理的输出。\n如果你对结果不满意，可以重新分配任务。\n'
                '</agent_team>\n\n')
        if len(instructions) > 0:
            system_message_content += '<instructions>'
            if len(instructions) > 1:
                for _upi in instructions:
                    system_message_content += f'\n- {_upi}'
            else:
                system_message_content += '\n' + instructions[0]
            system_message_content += '\n</instructions>\n\n'
        if len(additional_information) > 0:
            system_message_content += '<additional_information>'
            for _ai in additional_information:
                system_message_content += f'\n- {_ai}'
            system_message_content += '\n</additional_information>\n\n'
        if self.add_state_in_messages:
            system_message_content = self.format_message_with_state_variables(system_message_content)
        system_message_from_model = self.model.get_system_message_for_model()
        if system_message_from_model is not None:
            system_message_content += system_message_from_model
        if self.expected_output is not None:
            system_message_content += f'<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n'
        if self.additional_context is not None:
            system_message_content += f'{self.additional_context.strip()}\n'
        if self.has_team and self.add_transfer_instructions:
            system_message_content += f'<transfer_instructions>\n{self.get_transfer_instructions().strip()}\n</transfer_instructions>\n\n'
        if self.memory:
            if self.memory.create_user_memories:
                if self.memory.memories and len(self.memory.memories) > 0:
                    system_message_content += '您可以访问以前与用户交互的记忆，以便使用:\n\n'
                    system_message_content += '<memories_from_previous_interactions>'
                    for _memory in self.memory.memories:
                        system_message_content += f'\n- {_memory.memory}'
                    system_message_content += '\n</memories_from_previous_interactions>\n\n'
                    system_message_content += '注意：此信息来自之前的互动，可能会在本次对话中更新\n你应该总是更喜欢这次谈话中的信息，而不是过去的记忆。\n\n'
                else:
                    system_message_content += '你有能力保留之前与用户互动的记忆但尚未与用户进行任何交互。\n如果用户询问以前的记忆，你可以让他们知道你对用户没有任何记忆，因为你还没有任何互动。\n\n'
                system_message_content += '您可以使用`update_memory`工具添加新内存。\n如果使用`update_memory`工具，请记住将响应传递给用户。\n\n'
            if self.memory.create_session_summary:
                if self.memory.summary is not None:
                    system_message_content += 'Here is a brief summary of your previous interactions if it helps:\n\n'
                    system_message_content += '<summary_of_previous_interactions>\n'
                    system_message_content += str(self.memory.summary)
                    system_message_content += '\n</summary_of_previous_interactions>\n\n'
                    system_message_content += '注意：此信息来自以前的交互，可能已经过时。你应该总是更喜欢这次谈话中的信息，而不是过去的总结。\n\n'
        if self.response_model is not None and not (self.model.supports_native_structured_outputs and (not self.use_json_mode or self.structured_outputs is True)):
            system_message_content += f'{self.get_json_output_prompt()}'
        return Message(role=self.system_message_role, content=system_message_content.strip()) if system_message_content else None

    def get_user_message(self, *, message: Optional[Union[str, List]], audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, **kwargs: Any) -> Optional[Message]:
        references = None
        self.run_response = cast(RunResponse, self.run_response)
        if self.add_references and message:
            message_str: str
            if isinstance(message, str):
                message_str = message
            elif callable(message):
                message_str = message(agent=self)
            else:
                raise Exception('当add_references为True时，消息必须是字符串或可调用的')
            retrieval_timer = Timer()
            retrieval_timer.start()
            docs_from_knowledge = self.get_relevant_docs_from_knowledge(query=message_str, **kwargs)
            if docs_from_knowledge is not None:
                references = MessageReferences(query=message_str, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4))
                if self.run_response.extra_data is None:
                    self.run_response.extra_data = RunResponseExtraData()
                if self.run_response.extra_data.references is None:
                    self.run_response.extra_data.references = []
                self.run_response.extra_data.references.append(references)
            retrieval_timer.stop()
            print(f'Time to get references: {retrieval_timer.elapsed:.4f}s')
        if self.user_message is not None:
            if isinstance(self.user_message, Message):
                return self.user_message
            user_message_content = self.user_message
            if callable(self.user_message):
                user_message_kwargs = {'agent': self, 'message': message, 'references': references}
                user_message_content = self.user_message(**user_message_kwargs)
                if not isinstance(user_message_content, str):
                    raise Exception('user_message must return a string')
            if self.add_state_in_messages:
                user_message_content = self.format_message_with_state_variables(user_message_content)
            return Message(role=self.user_message_role, content=user_message_content, audio=audio, images=images, videos=videos, files=files, **kwargs)
        if not self.create_default_user_message or isinstance(message, list):
            return Message(role=self.user_message_role, content=message, images=images, audio=audio, videos=videos, files=files, **kwargs)
        if message is None:
            return None
        user_msg_content = message
        if self.add_state_in_messages:
            user_msg_content = self.format_message_with_state_variables(message)
        if self.add_references and references is not None and references.references is not None and len(references.references) > 0:
            user_msg_content += '\n\n如果有帮助，请使用知识库中的以下参考文献:\n'
            user_msg_content += '<references>\n'
            user_msg_content += self.convert_documents_to_string(references.references) + '\n'
            user_msg_content += '</references>'
        if self.add_context and self.context is not None:
            user_msg_content += '\n\n<context>\n'
            user_msg_content += self.convert_context_to_string(self.context) + '\n'
            user_msg_content += '</context>'
        return Message(role=self.user_message_role, content=user_msg_content, audio=audio, images=images, videos=videos, files=files, **kwargs)

    def get_run_messages(self, *, message: Optional[Union[str, List, Dict, Message]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, messages: Optional[Sequence[Union[Dict, Message]]] = None, **kwargs: Any) -> RunMessages:
        """此函数返回具有以下属性的RunMessages对象：
        -system_message：此运行的系统消息
        -user_message：此运行的用户消息
        -messages：要发送到模型的消息列表
        要构建RunMessages对象，请执行以下操作：
        1.将系统消息添加到run_message
        2.在run_message中添加额外的消息（如果提供）
        3.向run_message添加历史记录
        4.将用户消息添加到run_message
        5.在run_message中添加消息（如果提供）
        返回：
        具有以下属性的RunMessages对象：
        -system_message：此运行的系统消息
        -user_message：此运行的用户消息
        -messages：要发送到模型的所有消息的列表
        典型用法：
        run_messages=self.get_run_messages（message=消息，audio=音频，image=图像，video=视频，file=文件，messages=消息，**kwargs）
        """
        run_messages = RunMessages()
        self.memory = cast(AgentMemory, self.memory)
        self.run_response = cast(RunResponse, self.run_response)
        system_message = self.get_system_message()
        if system_message is not None:
            run_messages.system_message = system_message
            run_messages.messages.append(system_message)
        if self.add_messages is not None:
            messages_to_add_to_run_response: List[Message] = []
            if run_messages.extra_messages is None:
                run_messages.extra_messages = []
            for _m in self.add_messages:
                if isinstance(_m, Message):
                    messages_to_add_to_run_response.append(_m)
                    run_messages.messages.append(_m)
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        _m_parsed = Message.model_validate(_m)
                        messages_to_add_to_run_response.append(_m_parsed)
                        run_messages.messages.append(_m_parsed)
                        run_messages.extra_messages.append(_m_parsed)
                    except Exception as e:
                        print(f'验证消息失败: {e}')
            if len(messages_to_add_to_run_response) > 0:
                print(f'Adding {len(messages_to_add_to_run_response)} extra messages')
                if self.run_response.extra_data is None:
                    self.run_response.extra_data = RunResponseExtraData(add_messages=messages_to_add_to_run_response)
                else:
                    if self.run_response.extra_data.add_messages is None:
                        self.run_response.extra_data.add_messages = messages_to_add_to_run_response
                    else:
                        self.run_response.extra_data.add_messages.extend(messages_to_add_to_run_response)
        if self.add_history_to_messages:
            from copy import deepcopy
            history: List[Message] = self.memory.get_messages_from_last_n_runs(last_n=self.num_history_runs, skip_role=self.system_message_role)
            if len(history) > 0:
                history_copy = [deepcopy(msg) for msg in history]
                for _msg in history_copy:
                    _msg.from_history = True
                print(f'Adding {len(history_copy)} messages from history')
                run_messages.messages += history_copy
        user_message: Optional[Message] = None
        if message is None or isinstance(message, str) or isinstance(message, list):
            user_message = self.get_user_message(message=message, audio=audio, images=images, videos=videos, files=files, **kwargs)
        elif isinstance(message, Message):
            user_message = message
        elif isinstance(message, dict):
            try:
                user_message = Message.model_validate(message)
            except Exception as e:
                print(f'验证消息失败: {e}')
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)
        if messages is not None and len(messages) > 0:
            for _m in messages:
                if isinstance(_m, Message):
                    run_messages.messages.append(_m)
                    if run_messages.extra_messages is None:
                        run_messages.extra_messages = []
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        run_messages.messages.append(Message.model_validate(_m))
                        if run_messages.extra_messages is None:
                            run_messages.extra_messages = []
                        run_messages.extra_messages.append(Message.model_validate(_m))
                    except Exception as e:
                        print(f'验证消息失败: {e}')
        return run_messages

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> 'Agent':
        from dataclasses import fields
        excluded_fields = ['agent_session', 'session_name']
        fields_for_new_agent: Dict[str, Any] = {}
        for f in fields(self):
            if f.name in excluded_fields:
                continue
            field_value = getattr(self, f.name)
            if field_value is not None:
                fields_for_new_agent[f.name] = self._deep_copy_field(f.name, field_value)
        if update:
            fields_for_new_agent.update(update)
        new_agent = self.__class__(**fields_for_new_agent)
        print(f'Created new {self.__class__.__name__}')
        return new_agent

    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        from copy import copy, deepcopy
        if field_name in ('memory', 'reasoning_agent'):
            return field_value.deep_copy()
        elif field_name in ('storage', 'model', 'reasoning_model'):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    print(f'Failed to copy field: {field_name} - {e}')
                    return field_value
        elif isinstance(field_value, (list, dict, set)):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    print(f'Failed to copy field: {field_name} - {e}')
                    return field_value
        elif isinstance(field_value, BaseModel):
            try:
                return field_value.model_copy(deep=True)
            except Exception:
                try:
                    return field_value.model_copy(deep=False)
                except Exception as e:
                    print(f'Failed to copy field: {field_name} - {e}')
                    return field_value
        try:
            from copy import copy
            return copy(field_value)
        except Exception:
            return field_value

    def get_transfer_function(self, member_agent: 'Agent', index: int) -> Function:
        def _transfer_task_to_agent(task_description: str, expected_output: str, additional_information: Optional[str] = None) -> Iterator[str]:
            if member_agent.team_data is None:
                member_agent.team_data = {}
            member_agent.team_data['leader_session_id'] = self.session_id
            member_agent.team_data['leader_agent_id'] = self.agent_id
            member_agent.team_data['leader_run_id'] = self.run_id
            member_agent_task = f'{task_description}\n\n<expected_output>\n{expected_output}\n</expected_output>'
            try:
                if additional_information is not None and additional_information.strip() != '':
                    member_agent_task += f'\n\n<additional_information>\n{additional_information}\n</additional_information>'
            except Exception as e:
                print(f'Failed to add additional information to the member agent: {e}')
            member_agent_session_id = member_agent.session_id
            member_agent_agent_id = member_agent.agent_id
            member_agent_info = {'session_id': member_agent_session_id, 'agent_id': member_agent_agent_id}
            if self.team_data is None:
                self.team_data = {}
            if 'members' not in self.team_data:
                self.team_data['members'] = [member_agent_info]
            else:
                if member_agent_info not in self.team_data['members']:
                    self.team_data['members'].append(member_agent_info)
            if self.stream and member_agent.is_streamable:
                member_agent_run_response_stream = member_agent.run(member_agent_task, stream=True)
                for member_agent_run_response_chunk in member_agent_run_response_stream:
                    yield member_agent_run_response_chunk.content
            else:
                member_agent_run_response: RunResponse = member_agent.run(member_agent_task, stream=False)
                if member_agent_run_response.content is None:
                    yield 'No response from the member agent.'
                elif isinstance(member_agent_run_response.content, str):
                    yield member_agent_run_response.content
                elif issubclass(type(member_agent_run_response.content), BaseModel):
                    try:
                        yield member_agent_run_response.content.model_dump_json(indent=2)
                    except Exception as e:
                        yield str(e)
                else:
                    try:
                        import json
                        yield json.dumps(member_agent_run_response.content, indent=2)
                    except Exception as e:
                        yield str(e)
            yield self.team_response_separator
        agent_name = member_agent.name if member_agent.name else f'agent_{index}'
        agent_name = ''.join(c for c in agent_name if c.isalnum() or c in '_- ').strip()
        agent_name = agent_name.lower().replace(' ', '_')
        if member_agent.name is None:
            member_agent.name = agent_name
        strict = True if (member_agent.response_model is not None and member_agent.model is not None) else False
        transfer_function = Function.from_callable(_transfer_task_to_agent, strict=strict)
        transfer_function.strict = strict
        transfer_function.name = f'transfer_task_to_{agent_name}'
        transfer_function.description = dedent(f'''使用此功能将任务转移到{agent_name},您必须清晰简洁地描述代理应该完成的任务和预期的输出。
            Args:
            task_description(str):对代理应该完成的任务的清晰简洁的描述。
            expected_output(str):代理的预期输出。
            additional_information(可选[str]):帮助代理完成任务的其他信息。
            return:
            str：委托任务的结果。''')
        if member_agent.respond_directly:
            transfer_function.show_result = True
            transfer_function.stop_after_tool_call = True
        return transfer_function

    def get_transfer_instructions(self) -> str:
        if self.team and len(self.team) > 0:
            transfer_instructions = '您可以将任务转移到团队中的以下代理:\n'
            for agent_index, agent in enumerate(self.team):
                transfer_instructions += f'\nAgent {agent_index + 1}:\n'
                if agent.name:
                    transfer_instructions += f'Name: {agent.name}\n'
                if agent.role:
                    transfer_instructions += f'Role: {agent.role}\n'
                if agent.tools is not None:
                    _tools = []
                    for _tool in agent.tools:
                        if isinstance(_tool, Toolkit):
                            _tools.extend(list(_tool.functions.keys()))
                        elif isinstance(_tool, Function):
                            _tools.append(_tool.name)
                        elif callable(_tool):
                            _tools.append(_tool.__name__)
                    transfer_instructions += f'Available tools: {", ".join(_tools)}\n'
            return transfer_instructions
        return ''

    def get_relevant_docs_from_knowledge(self, query: str, num_documents: Optional[int] = None, **kwargs) -> Optional[List[Dict[str, Any]]]:
        if self.retriever is not None and callable(self.retriever):
            from inspect import signature
            try:
                sig = signature(self.retriever)
                retriever_kwargs: Dict[str, Any] = {}
                if 'agent' in sig.parameters:
                    retriever_kwargs = {'agent': self}
                retriever_kwargs.update({'query': query, 'num_documents': num_documents, **kwargs})
                return self.retriever(**retriever_kwargs)
            except Exception as e:
                print(f'Retriever failed: {e}')
                return None
        if self.knowledge is None:
            return None
        relevant_docs: List[Document] = self.knowledge.search(query=query, num_documents=num_documents, **kwargs)
        if len(relevant_docs) == 0:
            return None
        return [doc.to_dict() for doc in relevant_docs]
    
    async def aget_relevant_docs_from_knowledge(self, query: str, num_documents: Optional[int] = None, **kwargs) -> Optional[List[Dict[str, Any]]]:
        if self.retriever is not None and callable(self.retriever):
            from inspect import signature
            try:
                sig = signature(self.retriever)
                retriever_kwargs: Dict[str, Any] = {}
                if 'agent' in sig.parameters:
                    retriever_kwargs = {'agent': self}
                retriever_kwargs.update({'query': query, 'num_documents': num_documents, **kwargs})
                return self.retriever(**retriever_kwargs)
            except Exception as e:
                print(f'Retriever failed: {e}')
                return None
        if self.knowledge is None or self.knowledge.vector_db is None:
            return None
        relevant_docs: List[Document] = await self.knowledge.async_search(query=query, num_documents=num_documents, **kwargs)
        if len(relevant_docs) == 0:
            return None
        return [doc.to_dict() for doc in relevant_docs]

    def convert_documents_to_string(self, docs: List[Dict[str, Any]]) -> str:
        if docs is None or len(docs) == 0:
            return ''
        if self.references_format == 'yaml':
            import yaml
            return yaml.dump(docs)
        import json
        return json.dumps(docs, indent=2)

    def convert_context_to_string(self, context: Dict[str, Any]) -> str:
        if context is None:
            return ''
        import json
        try:
            return json.dumps(context, indent=2, default=str)
        except (TypeError, ValueError, OverflowError) as e:
            print(f'Failed to convert context to JSON: {e}')
            sanitized_context = {}
            for key, value in context.items():
                try:
                    json.dumps({key: value}, default=str)
                    sanitized_context[key] = value
                except Exception:
                    sanitized_context[key] = str(value)
            try:
                return json.dumps(sanitized_context, indent=2)
            except Exception as e:
                print(f'未能将经过净化的上下文转换为JSON: {e}')
                return str(context)

    def save_run_response_to_file(self, message: Optional[Union[str, List, Dict, Message]] = None) -> None:
        if self.save_response_to_file is not None and self.run_response is not None:
            message_str = None
            if message is not None:
                if isinstance(message, str):
                    message_str = message
                else:
                    print('输出文件名中未使用消息：消息不是字符串')
            try:
                from pathlib import Path
                fn = self.save_response_to_file.format(name=self.name, session_id=self.session_id, user_id=self.user_id, message=message_str, run_id=self.run_id)
                fn_path = Path(fn)
                if not fn_path.parent.exists():
                    fn_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(self.run_response.content, str):
                    fn_path.write_text(self.run_response.content)
                else:
                    import json
                    fn_path.write_text(json.dumps(self.run_response.content, indent=2))
            except Exception as e:
                print(f'未能将输出保存到文件: {e}')

    def update_run_response_with_reasoning(self, reasoning_steps: List[ReasoningStep], reasoning_agent_messages: List[Message]) -> None:
        self.run_response = cast(RunResponse, self.run_response)
        if self.run_response.extra_data is None:
            self.run_response.extra_data = RunResponseExtraData()
        extra_data = self.run_response.extra_data
        if extra_data.reasoning_steps is None:
            extra_data.reasoning_steps = reasoning_steps
        else:
            extra_data.reasoning_steps.extend(reasoning_steps)
        if extra_data.reasoning_messages is None:
            extra_data.reasoning_messages = reasoning_agent_messages
        else:
            extra_data.reasoning_messages.extend(reasoning_agent_messages)

    def aggregate_metrics_from_messages(self, messages: List[Message]) -> Dict[str, Any]:
        aggregated_metrics: Dict[str, Any] = defaultdict(list)
        assistant_message_role = self.model.assistant_message_role if self.model is not None else 'assistant'
        for m in messages:
            if m.role == assistant_message_role and m.metrics is not None:
                for k, v in asdict(m.metrics).items():
                    if k == 'timer':
                        continue
                    if v is not None:
                        aggregated_metrics[k].append(v)
        if aggregated_metrics is not None:
            aggregated_metrics = dict(aggregated_metrics)
        return aggregated_metrics

    def calculate_session_metrics(self, messages: List[Message]) -> SessionMetrics:
        session_metrics = SessionMetrics()
        assistant_message_role = self.model.assistant_message_role if self.model is not None else 'assistant'
        for m in messages:
            if m.role == assistant_message_role and m.metrics is not None:
                session_metrics += m.metrics
        return session_metrics

    def rename(self, name: str) -> None:
        self.read_from_storage()
        self.name = name
        self.write_to_storage()
        self._log_agent_session()

    def rename_session(self, session_name: str) -> None:
        self.read_from_storage()
        self.session_name = session_name
        self.write_to_storage()
        self._log_agent_session()

    def generate_session_name(self) -> str:
        if self.model is None:
            raise Exception('Model not set')
        gen_session_name_prompt = 'Conversation\n'
        messages_for_generating_session_name = []
        self.memory = cast(AgentMemory, self.memory)
        try:
            message_pairs = self.memory.get_message_pairs()
            for message_pair in message_pairs[:3]:
                messages_for_generating_session_name.append(message_pair[0])
                messages_for_generating_session_name.append(message_pair[1])
        except Exception as e:
            print(f'Failed to generate name: {e}')
        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f'{message.role.upper()}: {message.content}\n'
        gen_session_name_prompt += '\n\nConversation Name: '
        system_message = Message(role=self.system_message_role, content='请为这次对话提供一个合适的名称，最多5个字。')
        user_message = Message(role=self.user_message_role, content=gen_session_name_prompt)
        generate_name_messages = [system_message, user_message]
        generated_name = self.model.response(messages=generate_name_messages)
        content = generated_name.content
        if content is None:
            print('生成的名称为空。再试一次.')
            return self.generate_session_name()
        if len(content.split()) > 15:
            print('生成的名称太长。再试一次.')
            return self.generate_session_name()
        return content.replace('"', '').strip()

    def auto_rename_session(self) -> None:
        self.read_from_storage()
        generated_session_name = self.generate_session_name()
        print(f'Generated Session Name: {generated_session_name}')
        self.session_name = generated_session_name
        self.write_to_storage()
        self._log_agent_session()

    def delete_session(self, session_id: str):
        if self.storage is None:
            return
        self.storage.delete_session(session_id=session_id)

    def add_image(self, image: ImageArtifact) -> None:
        if self.images is None:
            self.images = []
        self.images.append(image)
        if self.run_response is not None:
            if self.run_response.images is None:
                self.run_response.images = []
            self.run_response.images.append(image)

    def add_video(self, video: VideoArtifact) -> None:
        if self.videos is None:
            self.videos = []
        self.videos.append(video)
        if self.run_response is not None:
            if self.run_response.videos is None:
                self.run_response.videos = []
            self.run_response.videos.append(video)

    def add_audio(self, audio: AudioArtifact) -> None:
        if self.audio is None:
            self.audio = []
        self.audio.append(audio)
        if self.run_response is not None:
            if self.run_response.audio is None:
                self.run_response.audio = []
            self.run_response.audio.append(audio)

    def get_images(self) -> Optional[List[ImageArtifact]]:
        return self.images

    def get_videos(self) -> Optional[List[VideoArtifact]]:
        return self.videos

    def get_audio(self) -> Optional[List[AudioArtifact]]:
        return self.audio

    def reason(self, run_messages: RunMessages) -> Iterator[RunResponse]:
        if self.stream_intermediate_steps:
            yield self.create_run_response(content='Reasoning started', event=RunEvent.reasoning_started)
        use_default_reasoning = False
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            reasoning_model = self.model.__class__(id=self.model.id)
        if reasoning_model is None:
            print('推理错误。推理模型为无，继续常规会话...')
            return
        if reasoning_model_provided:
            if reasoning_model.__class__.__name__ == 'DeepSeek' and reasoning_model.id.lower() == 'deepseek-reasoner':
                ds_reasoning_agent = self.reasoning_agent or get_deepseek_reasoning_agent(reasoning_model=reasoning_model, monitoring=self.monitoring)
                print('Starting DeepSeek Reasoning')
                ds_reasoning_message: Optional[Message] = get_deepseek_reasoning(reasoning_agent=ds_reasoning_agent, messages=run_messages.get_input_messages())
                if ds_reasoning_message is None:
                    print('推理错误。推理反应为无，继续常规会话...')
                    return
                run_messages.messages.append(ds_reasoning_message)
                self.update_run_response_with_reasoning(reasoning_steps=[ReasoningStep(result=ds_reasoning_message.content)], reasoning_agent_messages=[ds_reasoning_message])
                if self.stream_intermediate_steps:
                    yield self.create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=ds_reasoning_message.content)]), event=RunEvent.reasoning_completed)
            elif 'deepseek-r1' in reasoning_model.id.lower():
                openai_reasoning_agent = self.reasoning_agent or get_openai_reasoning_agent(reasoning_model=reasoning_model, monitoring=self.monitoring)
                print('Starting OpenAI Reasoning')
                openai_reasoning_message: Optional[Message] = get_openai_reasoning(reasoning_agent=openai_reasoning_agent, messages=run_messages.get_input_messages())
                if openai_reasoning_message is None:
                    print('推理错误。推理反应为无，继续常规会话...')
                    return
                run_messages.messages.append(openai_reasoning_message)
                self.update_run_response_with_reasoning(reasoning_steps=[ReasoningStep(result=openai_reasoning_message.content)], reasoning_agent_messages=[openai_reasoning_message])
                if self.stream_intermediate_steps:
                    yield self.create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=openai_reasoning_message.content)]), event=RunEvent.reasoning_completed)
            else:
                print(f'Reasoning model: {reasoning_model.__class__.__name__} 不是本地推理模型，默认为手动思维链推理')
                use_default_reasoning = True
        else:
            use_default_reasoning = True
        if use_default_reasoning:
            reasoning_agent: Optional[Agent] = self.reasoning_agent
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(reasoning_model=reasoning_model, min_steps=self.reasoning_min_steps, max_steps=self.reasoning_max_steps, tools=self.tools, use_json_mode=self.use_json_mode, monitoring=self.monitoring, telemetry=self.telemetry, debug_mode=self.debug_mode)
            if reasoning_agent is None:
                print('推理错误。推理代理为无，继续常规会话...')
                return
            if reasoning_agent.response_model is not None and not isinstance(reasoning_agent.response_model, type):
                if not issubclass(reasoning_agent.response_model, ReasoningSteps):
                    print('推理代理响应模型应为“推理步骤”，继续定期会话...')
                return
            reasoning_agent.show_tool_calls = False
            reasoning_agent.model.show_tool_calls = False
            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            print('Starting Reasoning')
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                print(f'Step {step_count}')
                step_count += 1
                try:
                    reasoning_agent_response: RunResponse = reasoning_agent.run(messages=run_messages.get_input_messages())
                    if reasoning_agent_response.content is None or reasoning_agent_response.messages is None:
                        print('推理错误。推理响应为空，继续常规会话...')
                        break
                    if reasoning_agent_response.content.reasoning_steps is None:
                        print('推理错误。推理步骤为空，继续常规会话...')
                        break
                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    if self.stream_intermediate_steps:
                        for reasoning_step in reasoning_steps:
                            yield self.create_run_response(content=reasoning_step, content_type=reasoning_step.__class__.__name__, event=RunEvent.reasoning_step)
                    first_assistant_index = next((i for i, m in enumerate(reasoning_agent_response.messages) if m.role == 'assistant'), len(reasoning_agent_response.messages))
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]
                    self.update_run_response_with_reasoning(reasoning_steps=reasoning_steps, reasoning_agent_messages=reasoning_agent_response.messages)
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    print(f'Reasoning error: {e}')
                    break
            print(f'Total Reasoning steps: {len(all_reasoning_steps)}')
            print('Reasoning finished')
            update_messages_with_reasoning(run_messages=run_messages, reasoning_messages=reasoning_messages)
            if self.stream_intermediate_steps:
                yield self.create_run_response(content=ReasoningSteps(reasoning_steps=all_reasoning_steps), content_type=ReasoningSteps.__class__.__name__, event=RunEvent.reasoning_completed)
                
    async def areason(self, run_messages: RunMessages) -> Any:
        if self.stream_intermediate_steps:
            yield self.create_run_response(content='Reasoning started', event=RunEvent.reasoning_started)
        use_default_reasoning = False
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            reasoning_model = self.model.__class__(id=self.model.id)
        if reasoning_model is None:
            print('推理错误。推理模型为无，继续常规会话...')
            return
        if reasoning_model_provided:
            if reasoning_model.__class__.__name__ == 'DeepSeek' and reasoning_model.id == 'deepseek-reasoner':
                ds_reasoning_agent = self.reasoning_agent or get_deepseek_reasoning_agent(reasoning_model=reasoning_model, monitoring=self.monitoring)
                print('Starting DeepSeek Reasoning')
                ds_reasoning_message: Optional[Message] = await aget_deepseek_reasoning(reasoning_agent=ds_reasoning_agent, messages=run_messages.get_input_messages())
                if ds_reasoning_message is None:
                    print('推理错误。推理反应为无，继续常规会话...')
                    return
                run_messages.messages.append(ds_reasoning_message)
                self.update_run_response_with_reasoning(reasoning_steps=[ReasoningStep(result=ds_reasoning_message.content)], reasoning_agent_messages=[ds_reasoning_message])
                if self.stream_intermediate_steps:
                    yield self.create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=ds_reasoning_message.content)]), event=RunEvent.reasoning_completed)
            elif 'deepseek' in reasoning_model.id.lower():
                openai_reasoning_agent = self.reasoning_agent or get_openai_reasoning_agent(reasoning_model=reasoning_model, monitoring=self.monitoring)
                print('Starting OpenAI Reasoning')
                openai_reasoning_message: Optional[Message] = await aget_openai_reasoning(reasoning_agent=openai_reasoning_agent, messages=run_messages.get_input_messages())
                if openai_reasoning_message is None:
                    print('推理错误。推理反应为无，继续常规会话...')
                    return
                run_messages.messages.append(openai_reasoning_message)
                self.update_run_response_with_reasoning(reasoning_steps=[ReasoningStep(result=openai_reasoning_message.content)], reasoning_agent_messages=[openai_reasoning_message])
                if self.stream_intermediate_steps:
                    yield self.create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=openai_reasoning_message.content)]), event=RunEvent.reasoning_completed)
            else:
                print(f'Reasoning model: {reasoning_model.__class__.__name__} 不是本地推理模型，默认为手动思维链推理')
                use_default_reasoning = True
        else:
            use_default_reasoning = True
        if use_default_reasoning:
            reasoning_agent: Optional[Agent] = self.reasoning_agent
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(reasoning_model=reasoning_model, min_steps=self.reasoning_min_steps, max_steps=self.reasoning_max_steps, tools=self.tools, use_json_mode=self.use_json_mode, monitoring=self.monitoring, telemetry=self.telemetry, debug_mode=self.debug_mode)
            if reasoning_agent is None:
                print('推理错误。推理代理为无，继续常规会话...')
                return
            if reasoning_agent.response_model is not None and not isinstance(reasoning_agent.response_model, type):
                if not issubclass(reasoning_agent.response_model, ReasoningSteps):
                    print('推理代理响应模型应为“ReasoningSteps”，继续定期会话...')
                return
            reasoning_agent.show_tool_calls = False
            reasoning_agent.model.show_tool_calls = False
            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            print('Starting Reasoning')
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                print(f'Step {step_count}')
                step_count += 1
                try:
                    reasoning_agent_response: RunResponse = await reasoning_agent.arun(messages=run_messages.get_input_messages())
                    if reasoning_agent_response.content is None or reasoning_agent_response.messages is None:
                        print('推理错误。推理响应为空，继续常规会话...')
                        break
                    if reasoning_agent_response.content.reasoning_steps is None:
                        print('推理错误。推理步骤为空，继续常规会话n...')
                        break
                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    if self.stream_intermediate_steps:
                        for reasoning_step in reasoning_steps:
                            yield self.create_run_response(content=reasoning_step, content_type=reasoning_step.__class__.__name__, event=RunEvent.reasoning_step)
                    first_assistant_index = next((i for i, m in enumerate(reasoning_agent_response.messages) if m.role == 'assistant'), len(reasoning_agent_response.messages))
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]
                    self.update_run_response_with_reasoning(reasoning_steps=reasoning_steps, reasoning_agent_messages=reasoning_agent_response.messages)
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    print(f'Reasoning error: {e}')
                    break
            print(f'Total Reasoning steps: {len(all_reasoning_steps)}')
            print('Reasoning finished')
            update_messages_with_reasoning(run_messages=run_messages, reasoning_messages=reasoning_messages)
            if self.stream_intermediate_steps:
                yield self.create_run_response(content=ReasoningSteps(reasoning_steps=all_reasoning_steps), content_type=ReasoningSteps.__class__.__name__, event=RunEvent.reasoning_completed)

    def get_chat_history(self, num_chats: Optional[int] = None) -> str:
        import json
        history: List[Dict[str, Any]] = []
        self.memory = cast(AgentMemory, self.memory)
        all_chats = self.memory.get_message_pairs()
        if len(all_chats) == 0:
            return ''
        chats_added = 0
        for chat in all_chats[::-1]:
            history.insert(0, chat[1].to_dict())
            history.insert(0, chat[0].to_dict())
            chats_added += 1
            if num_chats is not None and chats_added >= num_chats:
                break
        return json.dumps(history)

    def get_tool_call_history(self, num_calls: int = 3) -> str:
        import json
        self.memory = cast(AgentMemory, self.memory)
        tool_calls = self.memory.get_tool_calls(num_calls)
        if len(tool_calls) == 0:
            return ''
        print(f'tool_calls: {tool_calls}')
        return json.dumps(tool_calls)

    def search_knowledge_base(self, query: str) -> str:
        self.run_response = cast(RunResponse, self.run_response)
        retrieval_timer = Timer()
        retrieval_timer.start()
        docs_from_knowledge = self.get_relevant_docs_from_knowledge(query=query)
        if docs_from_knowledge is not None:
            references = MessageReferences(query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4))
            if self.run_response.extra_data is None:
                self.run_response.extra_data = RunResponseExtraData()
            if self.run_response.extra_data.references is None:
                self.run_response.extra_data.references = []
            self.run_response.extra_data.references.append(references)
        retrieval_timer.stop()
        print(f'Time to get references: {retrieval_timer.elapsed:.4f}s')
        if docs_from_knowledge is None:
            return 'No documents found'
        return self.convert_documents_to_string(docs_from_knowledge)
    
    async def async_search_knowledge_base(self, query: str) -> str:
        self.run_response = cast(RunResponse, self.run_response)
        retrieval_timer = Timer()
        retrieval_timer.start()
        docs_from_knowledge = await self.aget_relevant_docs_from_knowledge(query=query)
        if docs_from_knowledge is not None:
            references = MessageReferences(query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4))
            if self.run_response.extra_data is None:
                self.run_response.extra_data = RunResponseExtraData()
            if self.run_response.extra_data.references is None:
                self.run_response.extra_data.references = []
            self.run_response.extra_data.references.append(references)
        retrieval_timer.stop()
        print(f'Time to get references: {retrieval_timer.elapsed:.4f}s')
        if docs_from_knowledge is None:
            return 'No documents found'
        return self.convert_documents_to_string(docs_from_knowledge)

    def add_to_knowledge(self, query: str, result: str) -> str:
        import json
        if self.knowledge is None:
            return 'Knowledge base not available'
        document_name = self.name
        if document_name is None:
            document_name = query.replace(' ', '_').replace('?', '').replace('!', '').replace('.', '')
        document_content = json.dumps({'query': query, 'result': result})
        print(f'将文档添加到知识库: {document_name}: {document_content}')
        self.knowledge.load_document(document=Document(name=document_name, content=document_content))
        return '已成功添加到知识库'

    def update_memory(self, task: str) -> str:
        self.memory = cast(AgentMemory, self.memory)
        try:
            return self.memory.update_memory(input=task, force=True) or '内存更新成功'
        except Exception as e:
            return f'内存更新失败: {e}'

    def _log_agent_session(self):
        if not (self.telemetry or self.monitoring):
            return
        try:
            agent_session: AgentSession = self.agent_session or self.get_agent_session()
        except Exception as e:
            print(f'Could not create agent monitor: {e}')

    def _create_run_data(self) -> Dict[str, Any]:
        run_response_format = 'text'
        self.run_response = cast(RunResponse, self.run_response)
        if self.response_model is not None:
            run_response_format = 'json'
        elif self.markdown:
            run_response_format = 'markdown'
        functions = {}
        if self.model is not None and self.model._functions is not None:
            functions = {f_name: func.to_dict() for f_name, func in self.model._functions.items() if isinstance(func, Function)}
        run_data: Dict[str, Any] = {'functions': functions, 'metrics': self.run_response.metrics}
        if self.monitoring:
            run_data.update({'run_input': self.run_input, 'run_response': self.run_response.to_dict(), 'run_response_format': run_response_format})
        return run_data

    def _log_agent_run(self) -> None:
        self.set_monitoring()
        if not self.telemetry and not self.monitoring:
            return
        try:
            run_data = self._create_run_data()
            agent_session: AgentSession = self.agent_session or self.get_agent_session()
        except Exception as e:
            print(f'Could not create agent event: {e}')

    async def _alog_agent_run(self) -> None:
        self.set_monitoring()
        if not self.telemetry and not self.monitoring:
            return
        try:
            run_data = self._create_run_data()
            agent_session: AgentSession = self.agent_session or self.get_agent_session()
        except Exception as e:
            print(f'Could not create agent event: {e}')

    def print_response(self, message: Optional[Union[List, Dict, str, Message]] = None, *, messages: Optional[List[Union[Dict, Message]]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, stream: bool = False, markdown: bool = False, show_message: bool = True, show_reasoning: bool = True, show_full_reasoning: bool = False, console: Optional[Any] = None, tags_to_include_in_markdown: Set[str] = {'think', 'thinking'}, **kwargs: Any) -> None:
        import json
        from rich.console import Group
        from rich.json import JSON
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text
        if markdown:
            self.markdown = True
        if self.response_model is not None:
            self.markdown = False
            stream = False
        if stream:
            _response_content: str = ''
            _response_thinking: str = ''
            reasoning_steps: List[ReasoningStep] = []
            with Live(console=console) as live_log:
                status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
                live_log.update(status)
                response_timer = Timer()
                response_timer.start()
                render = False
                panels = [status]
                if message and show_message:
                    render = True
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                if render:
                    live_log.update(Group(*panels))
                for resp in self.run(message=message, messages=messages, audio=audio, images=images, videos=videos, files=files, stream=True, **kwargs):
                    if isinstance(resp, RunResponse):
                        if resp.event == RunEvent.run_response:
                            if isinstance(resp.content, str):
                                _response_content += resp.content
                            if resp.thinking is not None:
                                _response_thinking += resp.thinking
                        if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                            reasoning_steps = resp.extra_data.reasoning_steps
                    response_content_stream: Union[str, Markdown] = _response_content
                    if self.markdown:
                        escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                        response_content_stream = Markdown(escaped_content)
                    panels = [status]
                    if message and show_message:
                        render = True
                        message_content = get_text_from_message(message)
                        message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                        panels.append(message_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if len(reasoning_steps) > 0 and show_reasoning:
                        render = True
                        for i, step in enumerate(reasoning_steps, 1):
                            step_content = Text.assemble()
                            if step.title is not None:
                                step_content.append(f'{step.title}\n', 'bold')
                            if step.action is not None:
                                step_content.append(f'{step.action}\n', 'dim')
                            if step.result is not None:
                                step_content.append(Text.from_markup(step.result, style='dim'))
                            if show_full_reasoning:
                                if step.reasoning is not None:
                                    step_content.append(Text.from_markup(f'\n[bold]Reasoning:[/bold] {step.reasoning}', style='dim'))
                                if step.confidence is not None:
                                    step_content.append(Text.from_markup(f'\n[bold]Confidence:[/bold] {step.confidence}', style='dim'))
                            reasoning_panel = create_panel(content=step_content, title=f'Reasoning step {i}', border_style='green')
                            panels.append(reasoning_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if len(_response_thinking) > 0:
                        render = True
                        thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                        panels.append(thinking_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if self.show_tool_calls and self.run_response is not None and self.run_response.formatted_tool_calls:
                        render = True
                        tool_calls_content = Text()
                        for tool_call in self.run_response.formatted_tool_calls:
                            tool_calls_content.append(f'• {tool_call}\n')
                        tool_calls_panel = create_panel(content=tool_calls_content.plain.rstrip(), title='Tool Calls', border_style='yellow')
                        panels.append(tool_calls_panel)
                    if len(_response_content) > 0:
                        render = True
                        response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                        panels.append(response_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if isinstance(resp, RunResponse) and resp.citations is not None and resp.citations.urls is not None:
                        md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})'
                            for i, citation in enumerate(resp.citations.urls)
                            if citation.url)
                        if md_content:
                            citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                            panels.append(citations_panel)
                            live_log.update(Group(*panels))
                response_timer.stop()
                panels = [p for p in panels if not isinstance(p, Status)]
                live_log.update(Group(*panels))
        else:
            with Live(console=console) as live_log:
                status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
                live_log.update(status)
                response_timer = Timer()
                response_timer.start()
                panels = [status]
                if message and show_message:
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                    live_log.update(Group(*panels))
                run_response = self.run(message=message, messages=messages, audio=audio, images=images, videos=videos, files=files, stream=False, **kwargs)
                response_timer.stop()
                reasoning_steps = []
                if isinstance(run_response, RunResponse) and run_response.extra_data is not None and run_response.extra_data.reasoning_steps is not None:
                    reasoning_steps = run_response.extra_data.reasoning_steps
                if len(reasoning_steps) > 0 and show_reasoning:
                    for i, step in enumerate(reasoning_steps, 1):
                        step_content = Text.assemble()
                        if step.title is not None:
                            step_content.append(f'{step.title}\n', 'bold')
                        if step.action is not None:
                            step_content.append(f'{step.action}\n', 'dim')
                        if step.result is not None:
                            step_content.append(Text.from_markup(step.result, style='dim'))
                        if show_full_reasoning:
                            if step.reasoning is not None:
                                step_content.append(Text.from_markup(f'\n[bold]Reasoning:[/bold] {step.reasoning}', style='dim'))
                            if step.confidence is not None:
                                step_content.append(Text.from_markup(f'\n[bold]Confidence:[/bold] {step.confidence}', style='dim'))
                        reasoning_panel = create_panel(content=step_content, title=f'Reasoning step {i}', border_style='green')
                        panels.append(reasoning_panel)
                    live_log.update(Group(*panels))
                if isinstance(run_response, RunResponse) and run_response.thinking is not None:
                    thinking_panel = create_panel(content=Text(run_response.thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                    panels.append(thinking_panel)
                    live_log.update(Group(*panels))
                if self.show_tool_calls and isinstance(run_response, RunResponse) and run_response.formatted_tool_calls:
                    tool_calls_content = Text()
                    for tool_call in run_response.formatted_tool_calls:
                        tool_calls_content.append(f'• {tool_call}\n')
                    tool_calls_panel = create_panel(content=tool_calls_content.plain.rstrip(), title='Tool Calls', border_style='yellow')
                    panels.append(tool_calls_panel)
                    live_log.update(Group(*panels))
                response_content_batch: Union[str, JSON, Markdown] = ''
                if isinstance(run_response, RunResponse):
                    if isinstance(run_response.content, str):
                        if self.markdown:
                            escaped_content = escape_markdown_tags(run_response.content, tags_to_include_in_markdown)
                            response_content_batch = Markdown(escaped_content)
                        else:
                            response_content_batch = run_response.get_content_as_string(indent=4)
                    elif self.response_model is not None and isinstance(run_response.content, BaseModel):
                        try:
                            response_content_batch = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
                        except Exception as e:
                            print(f'Failed to convert response to JSON: {e}')
                    else:
                        try:
                            response_content_batch = JSON(json.dumps(run_response.content), indent=4)
                        except Exception as e:
                            print(f'Failed to convert response to JSON: {e}')
                response_panel = create_panel(content=response_content_batch, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                panels.append(response_panel)
                if isinstance(run_response, RunResponse) and run_response.citations is not None and run_response.citations.urls is not None:
                    md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})'
                        for i, citation in enumerate(run_response.citations.urls)
                        if citation.url)
                    if md_content:
                        citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                        panels.append(citations_panel)
                        live_log.update(Group(*panels))
                panels = [p for p in panels if not isinstance(p, Status)]
                live_log.update(Group(*panels))

    async def aprint_response(self, message: Optional[Union[List, Dict, str, Message]] = None, *, messages: Optional[List[Union[Dict, Message]]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, stream: bool = False, markdown: bool = False, show_message: bool = True, show_reasoning: bool = True, show_full_reasoning: bool = False, console: Optional[Any] = None, tags_to_include_in_markdown: Set[str] = {'think', 'thinking'}, **kwargs: Any) -> None:
        import json
        from rich.console import Group
        from rich.json import JSON
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text
        if markdown:
            self.markdown = True
        if self.response_model is not None:
            self.markdown = False
            stream = False
        if stream:
            _response_content: str = ''
            _response_thinking: str = ''
            reasoning_steps: List[ReasoningStep] = []
            with Live(console=console) as live_log:
                status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
                live_log.update(status)
                response_timer = Timer()
                response_timer.start()
                render = False
                panels = [status]
                if message and show_message:
                    render = True
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                if render:
                    live_log.update(Group(*panels))
                async for resp in await self.arun(message=message, messages=messages, audio=audio, images=images, videos=videos, files=files, stream=True, **kwargs):
                    if isinstance(resp, RunResponse):
                        if resp.event == RunEvent.run_response:
                            if isinstance(resp.content, str):
                                _response_content += resp.content
                            if resp.thinking is not None:
                                _response_thinking += resp.thinking
                        if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                            reasoning_steps = resp.extra_data.reasoning_steps
                    response_content_stream: Union[str, Markdown] = _response_content
                    if self.markdown:
                        escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                        response_content_stream = Markdown(escaped_content)
                    panels = [status]
                    if message and show_message:
                        render = True
                        message_content = get_text_from_message(message)
                        message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                        panels.append(message_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if len(reasoning_steps) > 0 and (show_reasoning or show_full_reasoning):
                        render = True
                        for i, step in enumerate(reasoning_steps, 1):
                            step_content = Text.assemble()
                            if step.title is not None:
                                step_content.append(f'{step.title}\n', 'bold')
                            if step.action is not None:
                                step_content.append(f'{step.action}\n', 'dim')
                            if step.result is not None:
                                step_content.append(Text.from_markup(step.result, style='dim'))
                            if show_full_reasoning:
                                if step.reasoning is not None:
                                    step_content.append(Text.from_markup(f'\n[bold]Reasoning:[/bold] {step.reasoning}', style='dim'))
                                if step.confidence is not None:
                                    step_content.append(Text.from_markup(f'\n[bold]Confidence:[/bold] {step.confidence}', style='dim'))
                            reasoning_panel = create_panel(content=step_content, title=f'Reasoning step {i}', border_style='green')
                            panels.append(reasoning_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if len(_response_thinking) > 0:
                        render = True
                        thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                        panels.append(thinking_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if self.show_tool_calls and self.run_response is not None and self.run_response.formatted_tool_calls:
                        render = True
                        tool_calls_content = Text()
                        for tool_call in self.run_response.formatted_tool_calls:
                            tool_calls_content.append(f'• {tool_call}\n')
                        tool_calls_panel = create_panel(content=tool_calls_content.plain.rstrip(), title='Tool Calls', border_style='yellow')
                        panels.append(tool_calls_panel)
                    if len(_response_content) > 0:
                        render = True
                        response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                        panels.append(response_panel)
                    if render:
                        live_log.update(Group(*panels))
                    if isinstance(resp, RunResponse) and resp.citations is not None and resp.citations.urls is not None:
                        md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})'
                            for i, citation in enumerate(resp.citations.urls)
                            if citation.url)
                        if md_content:
                            citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                            panels.append(citations_panel)
                            live_log.update(Group(*panels))
                response_timer.stop()
                panels = [p for p in panels if not isinstance(p, Status)]
                live_log.update(Group(*panels))
        else:
            with Live(console=console) as live_log:
                status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
                live_log.update(status)
                response_timer = Timer()
                response_timer.start()
                panels = [status]
                if message and show_message:
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                    live_log.update(Group(*panels))
                run_response = await self.arun(message=message, messages=messages, audio=audio, images=images, videos=videos, files=files, stream=False, **kwargs)
                response_timer.stop()
                reasoning_steps = []
                if isinstance(run_response, RunResponse) and run_response.extra_data is not None and run_response.extra_data.reasoning_steps is not None:
                    reasoning_steps = run_response.extra_data.reasoning_steps
                if len(reasoning_steps) > 0 and show_reasoning:
                    for i, step in enumerate(reasoning_steps, 1):
                        step_content = Text.assemble()
                        if step.title is not None:
                            step_content.append(f'{step.title}\n', 'bold')
                        if step.action is not None:
                            step_content.append(f'{step.action}\n', 'dim')
                        if step.result is not None:
                            step_content.append(Text.from_markup(step.result, style='dim'))
                        if show_full_reasoning:
                            if step.reasoning is not None:
                                step_content.append(Text.from_markup(f'\n[bold]Reasoning:[/bold] {step.reasoning}', style='dim'))
                            if step.confidence is not None:
                                step_content.append(Text.from_markup(f'\n[bold]Confidence:[/bold] {step.confidence}', style='dim'))
                        reasoning_panel = create_panel(content=step_content, title=f'Reasoning step {i}', border_style='green')
                        panels.append(reasoning_panel)
                    live_log.update(Group(*panels))
                if isinstance(run_response, RunResponse) and run_response.thinking is not None:
                    thinking_panel = create_panel(content=Text(run_response.thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                    panels.append(thinking_panel)
                    live_log.update(Group(*panels))
                if self.show_tool_calls and isinstance(run_response, RunResponse) and run_response.formatted_tool_calls:
                    tool_calls_content = Text()
                    for tool_call in run_response.formatted_tool_calls:
                        tool_calls_content.append(f'• {tool_call}\n')
                    tool_calls_panel = create_panel(content=tool_calls_content.plain.rstrip(), title='Tool Calls', border_style='yellow')
                    panels.append(tool_calls_panel)
                    live_log.update(Group(*panels))
                response_content_batch: Union[str, JSON, Markdown] = ''
                if isinstance(run_response, RunResponse):
                    if isinstance(run_response.content, str):
                        if self.markdown:
                            escaped_content = escape_markdown_tags(run_response.content, tags_to_include_in_markdown)
                            response_content_batch = Markdown(escaped_content)
                        else:
                            response_content_batch = run_response.get_content_as_string(indent=4)
                    elif self.response_model is not None and isinstance(run_response.content, BaseModel):
                        try:
                            response_content_batch = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
                        except Exception as e:
                            print(f'Failed to convert response to JSON: {e}')
                    else:
                        try:
                            response_content_batch = JSON(json.dumps(run_response.content), indent=4)
                        except Exception as e:
                            print(f'Failed to convert response to JSON: {e}')
                response_panel = create_panel(content=response_content_batch, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                panels.append(response_panel)
                if isinstance(run_response, RunResponse) and run_response.citations is not None and run_response.citations.urls is not None:
                    md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})'
                        for i, citation in enumerate(run_response.citations.urls)
                        if citation.url)
                    if md_content:
                        citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                        panels.append(citations_panel)
                        live_log.update(Group(*panels))
                panels = [p for p in panels if not isinstance(p, Status)]
                live_log.update(Group(*panels))

    def cli_app(self, message: Optional[str] = None, user: str = 'User', emoji: str = ':sunglasses:', stream: bool = False, markdown: bool = False, exit_on: Optional[List[str]] = None, **kwargs: Any) -> None:
        from rich.prompt import Prompt
        if message:
            self.print_response(message=message, stream=stream, markdown=markdown, **kwargs)
        _exit_on = exit_on or ['exit', 'quit', 'bye']
        while True:
            message = Prompt.ask(f'[bold] {emoji} {user} [/bold]')
            if message in _exit_on:
                break
            self.print_response(message=message, stream=stream, markdown=markdown, **kwargs)


class RunCancelledException(Exception):
    def __init__(self, message: str = 'Operation cancelled by user'):
        super().__init__(message)


class ModelProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502, model_name: Optional[str] = None, model_id: Optional[str] = None):
        self.model_name = model_name
        self.model_id = model_id
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return str(self.message)


def merge_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dictionaries(a[key], b[key])
        else:
            a[key] = b[key]


def create_panel(content, title, border_style='blue'):
    from rich.box import HEAVY
    from rich.panel import Panel
    return Panel(content, title=title, title_align='left', border_style=border_style, box=HEAVY, expand=True, padding=(1, 1))


def escape_markdown_tags(content: str, tags: Set[str]) -> str:
    escaped_content = content
    for tag in tags:
        escaped_content = escaped_content.replace(f'<{tag}>', f'&lt;{tag}&gt;')
        escaped_content = escaped_content.replace(f'</{tag}>', f'&lt;/{tag}&gt;')
    return escaped_content


def check_if_run_cancelled(run_response: Union[RunResponse, TeamRunResponse]):
    if run_response.event == RunEvent.run_cancelled:
        raise RunCancelledException()


def update_run_response_with_reasoning(run_response: Union[RunResponse, TeamRunResponse], reasoning_steps: List[ReasoningStep], reasoning_agent_messages: List[Message]) -> None:
    if run_response.extra_data is None:
        run_response.extra_data = RunResponseExtraData()
    if run_response.extra_data.reasoning_steps is None:
        run_response.extra_data.reasoning_steps = reasoning_steps
    else:
        run_response.extra_data.reasoning_steps.extend(reasoning_steps)
    if run_response.extra_data.reasoning_messages is None:
        run_response.extra_data.reasoning_messages = reasoning_agent_messages
    else:
        run_response.extra_data.reasoning_messages.extend(reasoning_agent_messages)


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    formatted_tool_calls = []
    for tool_call in tool_calls:
        if 'tool_name' in tool_call and 'tool_args' in tool_call:
            tool_name = tool_call['tool_name']
            args_str = ''
            if 'tool_args' in tool_call and tool_call['tool_args'] is not None:
                args_str = ', '.join(f'{k}={v}' for k, v in tool_call['tool_args'].items())
            formatted_tool_calls.append(f'{tool_name}({args_str})')
    return formatted_tool_calls


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if key not in kwargs:
            return f'{key}'
        return kwargs[key]

    def format_field(self, value, format_spec):
        if not format_spec:
            return super().format_field(value, format_spec)
        try:
            return super().format_field(value, format_spec)
        except ValueError:
            return f'{{{value}:{format_spec}}}'


def parse_response_model_str(content: str, response_model: Type[BaseModel]) -> Optional[BaseModel]:
    structured_output = None
    try:
        structured_output = response_model.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError):
        content = content
        if '```json' in content:
            content = content.split('```json')[-1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].strip()
        content = re.sub(r'[*_`#]', '', content)
        content = content.replace('\n', ' ').replace('\r', '')
        content = re.sub(r'[\x00-\x1F\x7F]', '', content)
        def escape_quotes_in_values(match):
            key = match.group(1)
            value = match.group(2)
            escaped_value = value.replace('"', '\\"')
            return f'"{key}": {escaped_value}'
        content = re.sub(r'"(?P<key>[^"]+)"\s*:\s*"(?P<value>.*?)(?="\s*(?:,|\}))', escape_quotes_in_values, content)
        try:
            structured_output = response_model.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f'Failed to parse cleaned JSON: {e}')
            try:
                data = json.loads(content)
                structured_output = response_model.model_validate(data)
            except (ValidationError, json.JSONDecodeError) as e:
                print(f'Failed to parse as Python dict: {e}')
    return structured_output


def get_text_from_message(message: Union[List, Dict, str, Message]) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_messages = []
        if len(message) == 0:
            return ''
        if 'type' in message[0]:
            for m in message:
                m_type = m.get('type')
                if m_type is not None and isinstance(m_type, str):
                    m_value = m.get(m_type)
                    if m_value is not None and isinstance(m_value, str):
                        if m_type == 'text':
                            text_messages.append(m_value)
                        if m_type == 'image_url':
                            text_messages.append(f'Image: {m_value}')
                        else:
                            text_messages.append(f'{m_type}: {m_value}')
        elif 'role' in message[0]:
            for m in message:
                m_role = m.get('role')
                if m_role is not None and isinstance(m_role, str):
                    m_content = m.get('content')
                    if m_content is not None and isinstance(m_content, str):
                        if m_role == 'user':
                            text_messages.append(m_content)
        if len(text_messages) > 0:
            return '\n'.join(text_messages)
    if isinstance(message, dict):
        if 'content' in message:
            return get_text_from_message(message['content'])
    if isinstance(message, Message) and message.content is not None:
        return get_text_from_message(message.content)
    return ''
