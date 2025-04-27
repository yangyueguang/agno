import asyncio
import textwrap
from inspect import getdoc
from agno.agent import *
from agno.media import Audio, AudioArtifact, AudioResponse, File, Image, ImageArtifact, Video, VideoArtifact
from agno.memory import Memory, TeamMemory, TeamRun
from agno.models import Model, Citations, Message, ModelResponse, ModelResponseEvent, Timer
from agno.run import NextAction, ReasoningStep, ReasoningSteps, RunMessages, RunEvent, RunResponse, TeamRunResponse, get_deepseek_reasoning, aget_deepseek_reasoning, get_next_action, update_messages_with_reasoning
from agno.storage import Storage, TeamSession
from agno.tools import Function, Toolkit
from functools import partial
from docstring_parser import parse


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


def get_entrypoint_docstring(entrypoint: Callable) -> str:
    if isinstance(entrypoint, partial):
        return str(entrypoint)
    doc = getdoc(entrypoint)
    if not doc:
        return ''
    parsed = parse(doc)
    lines = []
    if parsed.short_description:
        lines.append(parsed.short_description)
    if parsed.long_description:
        lines.extend(parsed.long_description.split('\n'))
    return '\n'.join(lines)


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


class Team:
    members: List[Union[Agent, 'Team']]
    mode: Literal['route', 'coordinate', 'collaborate'] = 'coordinate'
    model: Optional[Model] = None
    name: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    team_session_id: Optional[str] = None
    session_name: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    add_state_in_messages: bool = False
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    expected_output: Optional[str] = None
    markdown: bool = False
    add_datetime_to_instructions: bool = False
    success_criteria: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    add_context: bool = False
    enable_agentic_context: bool = False
    share_member_interactions: bool = False
    read_team_history: bool = False
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None
    show_tool_calls: bool = True
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tool_call_limit: Optional[int] = None
    response_model: Optional[Type[BaseModel]] = None
    use_json_mode: bool = False
    parse_response: bool = True
    memory: Optional[TeamMemory] = None
    enable_team_history: bool = False
    num_of_interactions_from_history: int = 3
    storage: Optional[Storage] = None
    extra_data: Optional[Dict[str, Any]] = None
    reasoning: bool = False
    reasoning_model: Optional[Model] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10
    debug_mode: bool = False
    show_members_responses: bool = False
    monitoring: bool = False
    telemetry: bool = True

    def __init__(self, members: List[Union[Agent, 'Team']], mode: Literal['route', 'coordinate', 'collaborate'] = 'coordinate',
                 model: Optional[Model] = None, name: Optional[str] = None, team_id: Optional[str] = None, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, session_name: Optional[str] = None, session_state: Optional[Dict[str, Any]] = None,
                 add_state_in_messages: bool = False, description: Optional[str] = None, instructions: Optional[Union[str, List[str], Callable]] = None,
                 expected_output: Optional[str] = None, success_criteria: Optional[str] = None, markdown: bool = False, add_datetime_to_instructions: bool = False,
                 context: Optional[Dict[str, Any]] = None, add_context: bool = False, enable_agentic_context: bool = False, share_member_interactions: bool = False,
                 read_team_history: bool = False, tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None, show_tool_calls: bool = True,
                 tool_call_limit: Optional[int] = None, tool_choice: Optional[Union[str, Dict[str, Any]]] = None, response_model: Optional[Type[BaseModel]] = None,
                 use_json_mode: bool = False, parse_response: bool = True, memory: Optional[TeamMemory] = None, enable_team_history: bool = False,
                 num_of_interactions_from_history: int = 3, storage: Optional[Storage] = None, extra_data: Optional[Dict[str, Any]] = None, reasoning: bool = False,
                 reasoning_model: Optional[Model] = None, reasoning_min_steps: int = 1, reasoning_max_steps: int = 10, debug_mode: bool = False,
                 show_members_responses: bool = False, monitoring: bool = False, telemetry: bool = True):
        self.members = members
        self.mode = mode
        self.model = model
        self.name = name
        self.team_id = team_id
        self.user_id = user_id
        self.session_id = session_id
        self.session_name = session_name
        self.session_state = session_state
        self.add_state_in_messages = add_state_in_messages
        self.description = description
        self.instructions = instructions
        self.expected_output = expected_output
        self.markdown = markdown
        self.add_datetime_to_instructions = add_datetime_to_instructions
        self.success_criteria = success_criteria
        self.context = context
        self.add_context = add_context
        self.enable_agentic_context = enable_agentic_context
        self.share_member_interactions = share_member_interactions
        self.read_team_history = read_team_history
        self.tools = tools
        self.show_tool_calls = show_tool_calls
        self.tool_choice = tool_choice
        self.tool_call_limit = tool_call_limit
        self.response_model = response_model
        self.use_json_mode = use_json_mode
        self.parse_response = parse_response
        self.memory = memory
        self.enable_team_history = enable_team_history
        self.num_of_interactions_from_history = num_of_interactions_from_history
        self.storage = storage
        self.extra_data = extra_data
        self.reasoning = reasoning
        self.reasoning_model = reasoning_model
        self.reasoning_min_steps = reasoning_min_steps
        self.reasoning_max_steps = reasoning_max_steps
        self.debug_mode = debug_mode
        self.show_members_responses = show_members_responses
        self.monitoring = monitoring
        self.telemetry = telemetry
        self.session_metrics: Optional[SessionMetrics] = None
        self.full_team_session_metrics: Optional[SessionMetrics] = None
        self.run_id: Optional[str] = None
        self.run_input: Optional[Union[str, List, Dict]] = None
        self.run_messages: Optional[RunMessages] = None
        self.run_response: Optional[TeamRunResponse] = None
        self.images: Optional[List[ImageArtifact]] = None
        self.audio: Optional[List[AudioArtifact]] = None
        self.videos: Optional[List[VideoArtifact]] = None
        self.team_session: Optional[TeamSession] = None
        self._formatter: Optional[SafeFormatter] = None
        self._tools_for_model: Optional[List[Dict]] = None
        self._functions_for_model: Optional[Dict[str, Function]] = None
        self._member_response_model: Optional[Type[BaseModel]] = None

    def _set_team_id(self) -> str:
        if self.team_id is None:
            self.team_id = str(uuid4())
        return self.team_id

    def _set_session_id(self) -> str:
        if self.session_id is None or self.session_id == '':
            self.session_id = str(uuid4())
        return self.session_id

    def _set_debug(self) -> None:
        if self.debug_mode or os.getenv('AGNO_DEBUG', 'false').lower() == 'true':
            self.debug_mode = True

    def _set_storage_mode(self):
        if self.storage is not None:
            self.storage.mode = 'team'

    def _set_monitoring(self) -> None:
        monitor_env = os.getenv('AGNO_MONITOR')
        if monitor_env is not None:
            self.monitoring = monitor_env.lower() == 'true'
        telemetry_env = os.getenv('AGNO_TELEMETRY')
        if telemetry_env is not None:
            self.telemetry = telemetry_env.lower() == 'true'

    def _initialize_member(self, member: Union['Team', Agent]):
        if self.debug_mode:
            member.debug_mode = True
        if self.show_tool_calls:
            member.show_tool_calls = True
        if self.markdown:
            member.markdown = True
        member.team_session_id = self.session_id
        member.team_id = self.team_id
        if member.name is None and member.role is None:
            print('Team member name and role is undefined.')

    def _initialize_team(self) -> None:
        self._set_storage_mode()
        self._set_debug()
        self._set_monitoring()
        self._set_team_id()
        self._set_session_id()
        print(f'Team ID: {self.team_id}\nSession ID: {self.session_id}')
        if self._formatter is None:
            self._formatter = SafeFormatter()
        for member in self.members:
            self._initialize_member(member)

    def run(self, message: Union[str, List, Dict, Message], *, stream: bool = False,          stream_intermediate_steps: bool = False, retries: Optional[int] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, **kwargs: Any) -> Union[TeamRunResponse, Iterator[TeamRunResponse]]:
        self._initialize_team()
        retries = retries or 3
        if retries < 1:
            raise ValueError('Retries must be at least 1')
        show_tool_calls = self.show_tool_calls
        self.read_from_storage()
        if self.memory is None:
            self.memory = TeamMemory()
        if self.context is not None:
            self._resolve_run_context()
        if self.response_model is not None and self.parse_response:
            stream = False
            print('Disabling stream as response_model is set')
        self._configure_model(show_tool_calls=show_tool_calls)
        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            run_id = str(uuid4())
            self.run_id = run_id
            print(f'Team Run Start: {self.run_id}\nMode: "{self.mode}"')
            if message is not None:
                if isinstance(message, str):
                    self.run_input = message
                elif isinstance(message, Message):
                    self.run_input = message.to_dict()
                else:
                    self.run_input = message
            _tools: List[Union[Toolkit, Callable, Function, Dict]] = []
            if self.tools is not None:
                for tool in self.tools:
                    _tools.append(tool)
            if self.read_team_history:
                _tools.append(self.get_team_history)
            if self.mode == 'route':
                user_message = self._get_user_message(message, audio=audio, images=images, videos=videos, files=files)
                forward_task_func: Function = self.get_forward_task_function(message=user_message, stream=stream, async_mode=False, images=images, videos=videos, audio=audio, files=files)
                _tools.append(forward_task_func)
            elif self.mode == 'coordinate':
                _tools.append(self.get_transfer_task_function(stream=stream, async_mode=False, images=images, videos=videos, audio=audio, files=files))
                if self.enable_agentic_context:
                    _tools.append(self.set_team_context)
            elif self.mode == 'collaborate':
                run_member_agents_func = self.get_run_member_agents_function(stream=stream, async_mode=False, images=images, videos=videos, audio=audio, files=files)
                _tools.append(run_member_agents_func)
                if self.enable_agentic_context:
                    _tools.append(self.set_team_context)
            self._add_tools_to_model(self.model, tools=_tools)
            try:
                self.run_response = TeamRunResponse(run_id=self.run_id, session_id=self.session_id, team_id=self.team_id)
                self.run_response.model = self.model.id if self.model is not None else None
                if self.mode == 'route':
                    run_messages: RunMessages = self.get_run_messages(run_response=self.run_response, message=message, audio=audio, images=images, videos=videos, files=files, **kwargs)
                else:
                    run_messages = self.get_run_messages(run_response=self.run_response, message=message, audio=audio, images=images, videos=videos, files=files, **kwargs)
                if stream:
                    resp = self._run_stream(run_response=self.run_response, run_messages=run_messages, stream_intermediate_steps=stream_intermediate_steps)
                    return resp
                else:
                    self._run(run_response=self.run_response, run_messages=run_messages)
                    return self.run_response
            except ModelProviderError as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
                last_exception = e
                if attempt < num_attempts - 1:
                    time.sleep(2**attempt)
            except (KeyboardInterrupt, RunCancelledException):
                return TeamRunResponse(run_id=self.run_id or str(uuid4()), session_id=self.session_id, team_id=self.team_id, content='Operation cancelled by user', event=RunEvent.run_cancelled)
        if last_exception is not None:
            print(f'Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})')
            raise last_exception
        else:
            raise Exception(f'Failed after {num_attempts} attempts.')

    def _run(self, run_response: TeamRunResponse, run_messages: RunMessages) -> None:
        """运行团队并返回响应。
        步骤：
        1.如果启用推理，则说明任务的原因
        2.从模型中获取响应
        3.更新run_response
        4.更新团队记忆
        5.计算会话度量
        6.将会话保存到存储
        7.解析任何结构化输出
        8.记录团队跑步记录
        """
        self.memory = cast(TeamMemory, self.memory)
        self.model = cast(Model, self.model)
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(run_response=run_response, run_messages=run_messages)
            deque(reasoning_generator, maxlen=0)
        self.run_messages = run_messages
        index_of_last_user_message = len(run_messages.messages)
        model_response = self.model.response(messages=run_messages.messages)
        if (self.response_model is not None) and not self.use_json_mode and (model_response.parsed is not None):
            run_response.content = model_response.parsed
            run_response.content_type = self.response_model.__name__
        else:
            if not run_response.content:
                run_response.content = model_response.content
            else:
                run_response.content += model_response.content
        if model_response.thinking is not None:
            if not run_response.thinking:
                run_response.thinking = model_response.thinking
            else:
                run_response.thinking += model_response.thinking
        if model_response.citations is not None:
            run_response.citations = model_response.citations
        if model_response.tool_calls is not None:
            if run_response.tools is None:
                run_response.tools = model_response.tool_calls
            else:
                run_response.tools.extend(model_response.tool_calls)
        run_response.formatted_tool_calls = format_tool_calls(run_response.tools)
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio
        run_response.created_at = model_response.created_at
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        run_response.messages = messages_for_run_response
        run_response.metrics = self._aggregate_metrics_from_messages(messages_for_run_response)
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role='system')
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        team_run = TeamRun(response=run_response)
        team_run.message = run_messages.user_message
        if self.memory is not None and self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            self.memory.update_memory(input=run_messages.user_message.get_content_string())
        self.memory.add_team_run(team_run)
        self.session_metrics = self._calculate_session_metrics()
        self.full_team_session_metrics = self._calculate_full_team_session_metrics()
        self.write_to_storage()
        if self.response_model is not None:
            if isinstance(run_response.content, str) and self.parse_response:
                try:
                    parsed_response_content = parse_response_model_str(run_response.content, self.response_model)
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        run_response.content_type = self.response_model.__name__
                    else:
                        print('Failed to convert response to response_model')
                except Exception as e:
                    print(f'Failed to convert response to output model: {e}')
            else:
                print('Something went wrong. Run response content is not a string')
        elif self._member_response_model is not None:
            if isinstance(run_response.content, str):
                try:
                    parsed_response_content = parse_response_model_str(run_response.content, self._member_response_model)
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        run_response.content_type = self._member_response_model.__name__
                    else:
                        print('Failed to convert response to response_model')
                except Exception as e:
                    print(f'Failed to convert response to output model: {e}')
            else:
                print('Something went wrong. Run response content is not a string')
        self._log_team_run()
        print(f'Team Run End: {self.run_id}')

    def _run_stream(self, run_response: TeamRunResponse, run_messages: RunMessages, stream_intermediate_steps: bool = False) -> Iterator[TeamRunResponse]:
        """运行Team并返回响应迭代器。
        步骤：
        1.如果启用推理，则说明任务的原因
        2.从模型中获取响应
        3.更新run_response
        4.更新团队记忆
        5.计算会话度量
        6.将会话保存到存储
        7.记录团队运行日志
        """
        self.memory = cast(TeamMemory, self.memory)
        self.model = cast(Model, self.model)
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(run_response=run_response, run_messages=run_messages)
            yield from reasoning_generator
        self.run_messages = run_messages
        index_of_last_user_message = len(run_messages.messages)
        if stream_intermediate_steps:
            yield self._create_run_response(content='Run started', event=RunEvent.run_started)
        full_model_response = ModelResponse()
        model_stream = self.model.response_stream(messages=run_messages.messages)
        for model_response_chunk in model_stream:
            if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                should_yield = False
                if model_response_chunk.content is not None:
                    if not full_model_response.content:
                        full_model_response.content = model_response_chunk.content
                    else:
                        full_model_response.content += model_response_chunk.content
                    should_yield = True
                if model_response_chunk.thinking is not None:
                    if not full_model_response.thinking:
                        full_model_response.thinking = model_response_chunk.thinking
                    else:
                        full_model_response.thinking += model_response_chunk.thinking
                    should_yield = True
                if model_response_chunk.citations is not None:
                    full_model_response.citations = model_response_chunk.citations
                    should_yield = True
                if model_response_chunk.audio is not None:
                    if full_model_response.audio is None:
                        full_model_response.audio = AudioResponse(id=str(uuid4()), content='', transcript='')
                    if model_response_chunk.audio.id is not None:
                        full_model_response.audio.id = model_response_chunk.audio.id
                    if model_response_chunk.audio.content is not None:
                        full_model_response.audio.content += model_response_chunk.audio.content
                    if model_response_chunk.audio.transcript is not None:
                        full_model_response.audio.transcript += model_response_chunk.audio.transcript
                    if model_response_chunk.audio.expires_at is not None:
                        full_model_response.audio.expires_at = model_response_chunk.audio.expires_at
                    if model_response_chunk.audio.mime_type is not None:
                        full_model_response.audio.mime_type = model_response_chunk.audio.mime_type
                    if model_response_chunk.audio.sample_rate is not None:
                        full_model_response.audio.sample_rate = model_response_chunk.audio.sample_rate
                    if model_response_chunk.audio.channels is not None:
                        full_model_response.audio.channels = model_response_chunk.audio.channels
                    should_yield = True
                if should_yield:
                    yield self._create_run_response(content=model_response_chunk.content, thinking=model_response_chunk.thinking, response_audio=model_response_chunk.audio, citations=model_response_chunk.citations, created_at=model_response_chunk.created_at)
            elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                tool_calls_list = model_response_chunk.tool_calls
                if tool_calls_list is not None:
                    if run_response.tools is None:
                        run_response.tools = tool_calls_list
                    else:
                        run_response.tools.extend(tool_calls_list)
                run_response.formatted_tool_calls = format_tool_calls(run_response.tools)
                if stream_intermediate_steps:
                    yield self._create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_started, from_run_response=run_response)
            elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                tool_calls_list = model_response_chunk.tool_calls
                if tool_calls_list is not None:
                    if run_response.tools:
                        tool_call_index_map = {tc['tool_call_id']: i
                            for i, tc in enumerate(run_response.tools)
                            if tc.get('tool_call_id') is not None}
                        for tool_call_dict in tool_calls_list:
                            tool_call_id = tool_call_dict.get('tool_call_id')
                            index = tool_call_index_map.get(tool_call_id)
                            if index is not None:
                                run_response.tools[index] = tool_call_dict
                    else:
                        run_response.tools = tool_calls_list
                    if stream_intermediate_steps:
                        yield self._create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_completed, from_run_response=run_response)
        run_response.created_at = full_model_response.created_at
        if full_model_response.content is not None:
            run_response.content = full_model_response.content
        if full_model_response.thinking is not None:
            run_response.thinking = full_model_response.thinking
        if full_model_response.audio is not None:
            run_response.response_audio = full_model_response.audio
        if full_model_response.citations is not None:
            run_response.citations = full_model_response.citations
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        run_response.messages = messages_for_run_response
        run_response.metrics = self._aggregate_metrics_from_messages(messages_for_run_response)
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role='system')
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        team_run = TeamRun(response=run_response)
        team_run.message = run_messages.user_message
        if self.memory is not None and self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            self.memory.update_memory(input=run_messages.user_message.get_content_string())
        self.memory.add_team_run(team_run)
        self.session_metrics = self._calculate_session_metrics()
        self.full_team_session_metrics = self._calculate_full_team_session_metrics()
        self.write_to_storage()
        self._log_team_run()
        if stream_intermediate_steps:
            yield self._create_run_response(from_run_response=run_response, event=RunEvent.run_completed)
        print(f'Team Run End: {self.run_id}')

    async def arun(self, message: Union[str, List, Dict, Message], *, stream: bool = False, stream_intermediate_steps: bool = False, retries: Optional[int] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, **kwargs: Any) -> Union[TeamRunResponse, AsyncIterator[TeamRunResponse]]:
        self._initialize_team()
        retries = retries or 3
        if retries < 1:
            raise ValueError('Retries must be at least 1')
        show_tool_calls = self.show_tool_calls
        self.read_from_storage()
        if self.memory is None:
            self.memory = TeamMemory()
        if self.context is not None:
            self._resolve_run_context()
        if self.response_model is not None and self.parse_response:
            stream = False
            print('Disabling stream as response_model is set')
        self._configure_model(show_tool_calls=show_tool_calls)
        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            run_id = str(uuid4())
            self.run_id = run_id
            print(f'Team Run Start: {self.run_id}')
            print(f'Mode: "{self.mode}"')
            if message is not None:
                if isinstance(message, str):
                    self.run_input = message
                elif isinstance(message, Message):
                    self.run_input = message.to_dict()
                else:
                    self.run_input = message
            _tools: List[Union[Function, Callable, Toolkit, Dict]] = []
            if self.tools is not None:
                for tool in self.tools:
                    _tools.append(tool)
            if self.read_team_history:
                _tools.append(self.get_team_history)
            if self.mode == 'route':
                user_message = self._get_user_message(message, audio=audio, images=images, videos=videos, files=files)
                forward_task_func: Function = self.get_forward_task_function(message=user_message, stream=stream, async_mode=True, images=images, videos=videos, audio=audio, files=files)
                _tools.append(forward_task_func)
                self.model.tool_choice = 'required'
            elif self.mode == 'coordinate':
                _tools.append(self.get_transfer_task_function(stream=stream, async_mode=True, images=images, videos=videos, audio=audio, files=files))
                self.model.tool_choice = 'auto'
                if self.enable_agentic_context:
                    _tools.append(self.set_team_context)
            elif self.mode == 'collaborate':
                run_member_agents_func = self.get_run_member_agents_function(stream=stream, async_mode=True, images=images, videos=videos, audio=audio, files=files)
                _tools.append(run_member_agents_func)
                self.model.tool_choice = 'auto'
                if self.enable_agentic_context:
                    _tools.append(self.set_team_context)
            self._add_tools_to_model(self.model, tools=_tools)
            try:
                self.run_response = TeamRunResponse(run_id=self.run_id, session_id=self.session_id, team_id=self.team_id)
                self.run_response.model = self.model.id if self.model is not None else None
                if self.mode == 'route':
                    run_messages: RunMessages = self.get_run_messages(run_response=self.run_response, message=message, audio=audio, images=images, videos=videos, files=files, **kwargs)
                else:
                    run_messages = self.get_run_messages(run_response=self.run_response, message=message, audio=audio, images=images, videos=videos, files=files, **kwargs)
                if stream:
                    resp = self._arun_stream(run_response=self.run_response, run_messages=run_messages, stream_intermediate_steps=stream_intermediate_steps)
                    return resp
                else:
                    await self._arun(run_response=self.run_response, run_messages=run_messages)
                    return self.run_response
            except ModelProviderError as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
                last_exception = e
                if attempt < num_attempts - 1:
                    await asyncio.sleep(2**attempt)
            except (KeyboardInterrupt, RunCancelledException):
                return TeamRunResponse(run_id=self.run_id or str(uuid4()), session_id=self.session_id, team_id=self.team_id, content='Operation cancelled by user', event=RunEvent.run_cancelled)
        if last_exception is not None:
            print(f'Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})')
            raise last_exception
        else:
            raise Exception(f'Failed after {num_attempts} attempts.')
        
    async def _arun(self, run_response: TeamRunResponse, run_messages: RunMessages) -> None:
        """运行团队并返回响应。
        步骤：
        1.如果启用推理，则说明任务的原因
        2.从模型中获取响应
        3.更新run_response
        4.更新团队记忆
        5.计算会话度量
        6.将会话保存到存储
        7.解析任何结构化输出
        8.记录团队运行记录
        """
        self.memory = cast(TeamMemory, self.memory)
        self.model = cast(Model, self.model)
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._areason(run_response=run_response, run_messages=run_messages)
            async for _ in reasoning_generator:
                pass
        self.run_messages = run_messages
        index_of_last_user_message = len(run_messages.messages)
        model_response = await self.model.aresponse(messages=run_messages.messages)
        if (self.response_model is not None) and not self.use_json_mode and (model_response.parsed is not None):
            run_response.content = model_response.parsed
            run_response.content_type = self.response_model.__name__
        else:
            if not run_response.content:
                run_response.content = model_response.content
            else:
                run_response.content += model_response.content
        if model_response.thinking is not None:
            if not run_response.thinking:
                run_response.thinking = model_response.thinking
            else:
                run_response.thinking += model_response.thinking
        if model_response.citations is not None:
            run_response.citations = model_response.citations
        if model_response.tool_calls is not None:
            if run_response.tools is None:
                run_response.tools = model_response.tool_calls
            else:
                run_response.tools.extend(model_response.tool_calls)
        run_response.formatted_tool_calls = format_tool_calls(run_response.tools)
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio
        run_response.created_at = model_response.created_at
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        run_response.messages = messages_for_run_response
        run_response.metrics = self._aggregate_metrics_from_messages(messages_for_run_response)
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role='system')
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        team_run = TeamRun(response=run_response)
        team_run.message = run_messages.user_message
        if self.memory is not None and self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            await self.memory.aupdate_memory(input=run_messages.user_message.get_content_string())
        self.memory.add_team_run(team_run)
        self.session_metrics = self._calculate_session_metrics()
        self.full_team_session_metrics = self._calculate_full_team_session_metrics()
        self.write_to_storage()
        if self.response_model is not None:
            if isinstance(run_response.content, str) and self.parse_response:
                try:
                    parsed_response_content = parse_response_model_str(run_response.content, self.response_model)
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        run_response.content_type = self.response_model.__name__
                    else:
                        print('Failed to convert response to response_model')
                except Exception as e:
                    print(f'Failed to convert response to output model: {e}')
            else:
                print('Something went wrong. Run response content is not a string')
        elif self._member_response_model is not None:
            if isinstance(run_response.content, str):
                try:
                    parsed_response_content = parse_response_model_str(run_response.content, self._member_response_model)
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        run_response.content_type = self._member_response_model.__name__
                    else:
                        print('Failed to convert response to response_model')
                except Exception as e:
                    print(f'Failed to convert response to output model: {e}')
            else:
                print('Something went wrong. Run response content is not a string')
        await self._alog_team_run()
        print(f'Team Run End: {self.run_id}')
        
    async def _arun_stream(self, run_response: TeamRunResponse, run_messages: RunMessages, stream_intermediate_steps: bool = False) -> AsyncIterator[TeamRunResponse]:
        """运行团队并返回响应。
        步骤：
        1.如果启用推理，则说明任务的原因
        2.从模型中获取响应
        3.更新run_response
        4.更新团队记忆
        5.计算会话度量
        6.将会话保存到存储
        7.记录团队运行日志
        """
        self.memory = cast(TeamMemory, self.memory)
        self.model = cast(Model, self.model)
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._areason(run_response=run_response, run_messages=run_messages)
            async for reasoning_response in reasoning_generator:
                yield reasoning_response
        self.run_messages = run_messages
        index_of_last_user_message = len(run_messages.messages)
        if stream_intermediate_steps:
            yield self._create_run_response(content='Run started', event=RunEvent.run_started)
        full_model_response = ModelResponse()
        model_stream = self.model.aresponse_stream(messages=run_messages.messages)
        async for model_response_chunk in model_stream:
            if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                should_yield = False
                if model_response_chunk.content is not None:
                    if not full_model_response.content:
                        full_model_response.content = model_response_chunk.content
                    else:
                        full_model_response.content += model_response_chunk.content
                    should_yield = True
                if model_response_chunk.thinking is not None:
                    if not full_model_response.thinking:
                        full_model_response.thinking = model_response_chunk.thinking
                    else:
                        full_model_response.thinking += model_response_chunk.thinking
                    should_yield = True
                if model_response_chunk.citations is not None:
                    full_model_response.citations = model_response_chunk.citations
                    should_yield = True
                if model_response_chunk.audio is not None:
                    if full_model_response.audio is None:
                        full_model_response.audio = AudioResponse(id=str(uuid4()), content='', transcript='')
                    if model_response_chunk.audio.id is not None:
                        full_model_response.audio.id = model_response_chunk.audio.id
                    if model_response_chunk.audio.content is not None:
                        full_model_response.audio.content += model_response_chunk.audio.content
                    if model_response_chunk.audio.transcript is not None:
                        full_model_response.audio.transcript += model_response_chunk.audio.transcript
                    if model_response_chunk.audio.expires_at is not None:
                        full_model_response.audio.expires_at = model_response_chunk.audio.expires_at
                    if model_response_chunk.audio.mime_type is not None:
                        full_model_response.audio.mime_type = model_response_chunk.audio.mime_type
                    if model_response_chunk.audio.sample_rate is not None:
                        full_model_response.audio.sample_rate = model_response_chunk.audio.sample_rate
                    if model_response_chunk.audio.channels is not None:
                        full_model_response.audio.channels = model_response_chunk.audio.channels
                    should_yield = True
                if should_yield:
                    yield self._create_run_response(content=model_response_chunk.content, thinking=model_response_chunk.thinking, response_audio=model_response_chunk.audio, citations=model_response_chunk.citations, created_at=model_response_chunk.created_at)
            elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                tool_calls_list = model_response_chunk.tool_calls
                if tool_calls_list is not None:
                    if run_response.tools is None:
                        run_response.tools = tool_calls_list
                    else:
                        run_response.tools.extend(tool_calls_list)
                run_response.formatted_tool_calls = format_tool_calls(run_response.tools)
                if stream_intermediate_steps:
                    yield self._create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_started, from_run_response=run_response)
            elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                tool_calls_list = model_response_chunk.tool_calls
                if tool_calls_list is not None:
                    if run_response.tools:
                        tool_call_index_map = {tc['tool_call_id']: i
                            for i, tc in enumerate(run_response.tools)
                            if tc.get('tool_call_id') is not None}
                        for tool_call_dict in tool_calls_list:
                            tool_call_id = tool_call_dict.get('tool_call_id')
                            index = tool_call_index_map.get(tool_call_id)
                            if index is not None:
                                run_response.tools[index] = tool_call_dict
                    else:
                        run_response.tools = tool_calls_list
                    if stream_intermediate_steps:
                        yield self._create_run_response(content=model_response_chunk.content, event=RunEvent.tool_call_completed, from_run_response=run_response)
        if (self.response_model is not None) and not self.use_json_mode and (full_model_response.parsed is not None):
            run_response.content = full_model_response.parsed
        run_response.created_at = full_model_response.created_at
        if full_model_response.content is not None:
            run_response.content = full_model_response.content
        if full_model_response.thinking is not None:
            run_response.thinking = full_model_response.thinking
        if full_model_response.audio is not None:
            run_response.response_audio = full_model_response.audio
        if full_model_response.citations is not None:
            run_response.citations = full_model_response.citations
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        run_response.messages = messages_for_run_response
        run_response.metrics = self._aggregate_metrics_from_messages(messages_for_run_response)
        if run_messages.system_message is not None:
            self.memory.add_system_message(run_messages.system_message, system_message_role='system')
        messages_for_memory: List[Message] = ([run_messages.user_message] if run_messages.user_message is not None else [])
        for _rm in run_messages.messages[index_of_last_user_message:]:
            if _rm.add_to_agent_memory:
                messages_for_memory.append(_rm)
        if len(messages_for_memory) > 0:
            self.memory.add_messages(messages=messages_for_memory)
        team_run = TeamRun(response=run_response)
        team_run.message = run_messages.user_message
        if self.memory is not None and self.memory.create_user_memories and self.memory.update_user_memories_after_run and run_messages.user_message is not None:
            await self.memory.aupdate_memory(input=run_messages.user_message.get_content_string())
        self.memory.add_team_run(team_run)
        self.session_metrics = self._calculate_session_metrics()
        self.full_team_session_metrics = self._calculate_full_team_session_metrics()
        self.write_to_storage()
        await self._alog_team_run()
        if stream_intermediate_steps:
            yield self._create_run_response(from_run_response=run_response, event=RunEvent.run_completed)
        print(f'Team Run End: {self.run_id}')

    def print_response(self, message: Optional[Union[List, Dict, str, Message]] = None, *, stream: bool = False, stream_intermediate_steps: bool = False, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, console: Optional[Any] = None, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: Optional[bool] = None, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        if markdown is None:
            markdown = self.markdown
        else:
            self.markdown = markdown
        if self.response_model is not None:
            stream = False
        if stream:
            self._print_response_stream(message=message, console=console, show_message=show_message, show_reasoning=show_reasoning, show_reasoning_verbose=show_reasoning_verbose, tags_to_include_in_markdown=tags_to_include_in_markdown, audio=audio, images=images, videos=videos, files=files, markdown=markdown, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
        else:
            self._print_response(message=message, console=console, show_message=show_message, show_reasoning=show_reasoning, show_reasoning_verbose=show_reasoning_verbose, tags_to_include_in_markdown=tags_to_include_in_markdown, audio=audio, images=images, videos=videos, files=files, markdown=markdown, **kwargs)

    def _print_response(self, message: Optional[Union[List, Dict, str, Message]] = None, console: Optional[Any] = None, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: bool = False, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        with Live(console=console) as live_console:
            status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
            live_console.update(status)
            response_timer = Timer()
            response_timer.start()
            panels = [status]
            if message and show_message:
                message_content = get_text_from_message(message)
                message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                panels.append(message_panel)
                live_console.update(Group(*panels))
            run_response: TeamRunResponse = self.run(message=message, images=images, audio=audio, videos=videos, files=files, stream=False, **kwargs)
            response_timer.stop()
            team_markdown = False
            member_markdown = {}
            if markdown:
                for member in self.members:
                    if isinstance(member, Agent) and member.agent_id is not None:
                        member_markdown[member.agent_id] = True
                    if isinstance(member, Team) and member.team_id is not None:
                        member_markdown[member.team_id] = True
                team_markdown = True
            if self.response_model is not None:
                team_markdown = False
            for member in self.members:
                if member.response_model is not None and isinstance(member, Agent) and member.agent_id is not None:
                    member_markdown[member.agent_id] = False
                if member.response_model is not None and isinstance(member, Team) and member.team_id is not None:
                    member_markdown[member.team_id] = False
            reasoning_steps = []
            if isinstance(run_response, TeamRunResponse) and run_response.extra_data is not None and run_response.extra_data.reasoning_steps is not None:
                reasoning_steps = run_response.extra_data.reasoning_steps
            if len(reasoning_steps) > 0 and show_reasoning:
                for i, step in enumerate(reasoning_steps, 1):
                    reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                    panels.append(reasoning_panel)
                live_console.update(Group(*panels))
            if isinstance(run_response, TeamRunResponse) and run_response.thinking is not None:
                thinking_panel = create_panel(content=Text(run_response.thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                panels.append(thinking_panel)
                live_console.update(Group(*panels))
            if isinstance(run_response, TeamRunResponse):
                if self.show_members_responses:
                    for member_response in run_response.member_responses:
                        reasoning_steps = []
                        if isinstance(member_response, RunResponse) and member_response.extra_data is not None and member_response.extra_data.reasoning_steps is not None:
                            reasoning_steps.extend(member_response.extra_data.reasoning_steps)
                        if len(reasoning_steps) > 0 and show_reasoning:
                            for i, step in enumerate(reasoning_steps, 1):
                                member_reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose, color='magenta')
                                panels.append(member_reasoning_panel)
                        if self.show_tool_calls and hasattr(member_response, 'tools') and member_response.tools:
                            member_name = None
                            if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                                member_name = self._get_member_name(member_response.agent_id)
                            elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                                member_name = self._get_member_name(member_response.team_id)
                            if member_name:
                                formatted_calls = format_tool_calls(member_response.tools)
                                if formatted_calls:
                                    console_width = console.width if console else 80
                                    panel_width = console_width + 30
                                    lines = []
                                    for call in formatted_calls:
                                        wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                        lines.append(wrapped_call)
                                    tool_calls_text = '\n\n'.join(lines)
                                    member_tool_calls_panel = create_panel(content=tool_calls_text, title=f'{member_name} Tool Calls', border_style='yellow')
                                    panels.append(member_tool_calls_panel)
                                    live_console.update(Group(*panels))
                        show_markdown = False
                        if member_markdown:
                            if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                                show_markdown = member_markdown.get(member_response.agent_id, False)
                            elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                                show_markdown = member_markdown.get(member_response.team_id, False)
                        member_response_content: Union[str, JSON, Markdown] = self._parse_response_content(member_response, tags_to_include_in_markdown, show_markdown=show_markdown)
                        if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                            member_response_panel = create_panel(content=member_response_content, title=f'{self._get_member_name(member_response.agent_id)} Response', border_style='magenta')
                        elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                            member_response_panel = create_panel(content=member_response_content, title=f'{self._get_member_name(member_response.team_id)} Response', border_style='magenta')
                        panels.append(member_response_panel)
                        if member_response.citations is not None and member_response.citations.urls is not None:
                            md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(member_response.citations.urls) if citation.url)
                            if md_content:
                                citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='magenta')
                                panels.append(citations_panel)
                    live_console.update(Group(*panels))
                if self.show_tool_calls and run_response.tools:
                    formatted_calls = format_tool_calls(run_response.tools)
                    if formatted_calls:
                        console_width = console.width if console else 80
                        panel_width = console_width + 30
                        lines = []
                        for call in formatted_calls:
                            wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                            lines.append(wrapped_call)
                        tool_calls_text = '\n\n'.join(lines)
                        team_tool_calls_panel = create_panel(content=tool_calls_text, title='Team Tool Calls', border_style='yellow')
                        panels.append(team_tool_calls_panel)
                        live_console.update(Group(*panels))
                response_content_batch: Union[str, JSON, Markdown] = self._parse_response_content(run_response, tags_to_include_in_markdown, show_markdown=team_markdown)
                response_panel = create_panel(content=response_content_batch, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                panels.append(response_panel)
                if run_response.citations is not None and run_response.citations.urls is not None:
                    md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(run_response.citations.urls) if citation.url)
                    if md_content:
                        citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                        panels.append(citations_panel)
            panels = [p for p in panels if not isinstance(p, Status)]
            live_console.update(Group(*panels))

    def _print_response_stream(self, message: Optional[Union[List, Dict, str, Message]] = None, console: Optional[Any] = None, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: bool = False, stream_intermediate_steps: bool = False, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        stream_intermediate_steps = True
        _response_content: str = ''
        _response_thinking: str = ''
        reasoning_steps: List[ReasoningStep] = []
        member_tool_calls = {}
        team_tool_calls = []
        processed_tool_calls = set()
        with Live(console=console) as live_console:
            status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
            live_console.update(status)
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
                live_console.update(Group(*panels))
            stream_resp = self.run(message=message, audio=audio, images=images, videos=videos, files=files, stream=True, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
            team_markdown = None
            member_markdown = {}
            member_response_panels = {}
            for resp in stream_resp:
                if team_markdown is None:
                    if markdown:
                        team_markdown = True
                    else:
                        team_markdown = False
                    if self.response_model is not None:
                        team_markdown = False
                if isinstance(resp, TeamRunResponse):
                    if resp.event == RunEvent.run_response:
                        if isinstance(resp.content, str):
                            _response_content += resp.content
                        if resp.thinking is not None:
                            _response_thinking += resp.thinking
                    if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                        reasoning_steps = resp.extra_data.reasoning_steps
                    if self.show_tool_calls and resp.tools:
                        for tool in resp.tools:
                            tool_id = tool.get('tool_call_id', str(hash(str(tool))))
                            if tool_id not in processed_tool_calls:
                                processed_tool_calls.add(tool_id)
                                team_tool_calls.append(tool)
                if self.show_tool_calls and resp.member_responses:
                    for member_response in resp.member_responses:
                        member_id = None
                        if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                            member_id = member_response.agent_id
                        elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                            member_id = member_response.team_id
                        if member_id and hasattr(member_response, 'tools') and member_response.tools:
                            if member_id not in member_tool_calls:
                                member_tool_calls[member_id] = []
                            for tool in member_response.tools:
                                tool_id = tool.get('tool_call_id', str(hash(str(tool))))
                                if tool_id not in processed_tool_calls:
                                    processed_tool_calls.add(tool_id)
                                    member_tool_calls[member_id].append(tool)
                response_content_stream: Union[str, Markdown] = _response_content
                if team_markdown:
                    escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                    response_content_stream = Markdown(escaped_content)
                panels = []
                if message and show_message:
                    render = True
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                if len(reasoning_steps) > 0 and show_reasoning:
                    render = True
                    for i, step in enumerate(reasoning_steps, 1):
                        reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                        panels.append(reasoning_panel)
                if len(_response_thinking) > 0:
                    render = True
                    thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                    panels.append(thinking_panel)
                elif _response_content == '':
                    panels.append(status)
                for member_response in resp.member_responses if hasattr(resp, 'member_responses') else []:
                    member_id = None
                    member_name = 'Team Member'
                    if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                        member_id = member_response.agent_id
                        member_name = self._get_member_name(member_id)
                    elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                        member_id = member_response.team_id
                        member_name = self._get_member_name(member_id)
                    if self.show_tool_calls and member_id in member_tool_calls and member_tool_calls[member_id]:
                        formatted_calls = format_tool_calls(member_tool_calls[member_id])
                        if formatted_calls:
                            console_width = console.width if console else 80
                            panel_width = console_width + 30
                            lines = []
                            for call in formatted_calls:
                                wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                lines.append(wrapped_call)
                            tool_calls_text = '\n\n'.join(lines)
                            member_tool_calls_panel = create_panel(content=tool_calls_text, title=f'{member_name} Tool Calls', border_style='yellow')
                            panels.append(member_tool_calls_panel)
                    if self.show_members_responses and member_id is not None:
                        show_markdown = False
                        if markdown:
                            show_markdown = True
                        member_response_content = self._parse_response_content(member_response, tags_to_include_in_markdown, show_markdown=show_markdown)
                        member_response_panel = create_panel(content=member_response_content, title=f'{member_name} Response', border_style='magenta')
                        panels.append(member_response_panel)
                        if member_id is not None:
                            member_response_panels[member_id] = member_response_panel
                if self.show_tool_calls and team_tool_calls:
                    formatted_calls = format_tool_calls(team_tool_calls)
                    if formatted_calls:
                        console_width = console.width if console else 80
                        panel_width = console_width + 30
                        lines = []
                        added_calls = set()
                        for call in formatted_calls:
                            if call not in added_calls:
                                added_calls.add(call)
                                wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                lines.append(wrapped_call)
                        tool_calls_text = '\n\n'.join(lines)
                        team_tool_calls_panel = create_panel(content=tool_calls_text, title='Team Tool Calls', border_style='yellow')
                        panels.append(team_tool_calls_panel)
                if len(_response_content) > 0:
                    render = True
                    response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                    panels.append(response_panel)
                if render or len(panels) > 0:
                    live_console.update(Group(*panels))
            response_timer.stop()
            if resp.citations is not None and resp.citations.urls is not None:
                md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})'
                    for i, citation in enumerate(resp.citations.urls)
                    if citation.url)
                if md_content:
                    citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                    panels.append(citations_panel)
                    live_console.update(Group(*panels))
            panels = [p for p in panels if not isinstance(p, Status)]
            if markdown:
                for member in self.members:
                    if isinstance(member, Agent) and member.agent_id is not None:
                        member_markdown[member.agent_id] = True
                    if isinstance(member, Team) and member.team_id is not None:
                        member_markdown[member.team_id] = True
            for member in self.members:
                if member.response_model is not None and isinstance(member, Agent) and member.agent_id is not None:
                    member_markdown[member.agent_id] = False
                if member.response_model is not None and isinstance(member, Team) and member.team_id is not None:
                    member_markdown[member.team_id] = False
            final_panels = []
            if message and show_message:
                message_content = get_text_from_message(message)
                message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                final_panels.append(message_panel)
            if reasoning_steps and show_reasoning:
                for i, step in enumerate(reasoning_steps, 1):
                    reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                    final_panels.append(reasoning_panel)
            if _response_thinking:
                thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                final_panels.append(thinking_panel)
            for i, member_response in enumerate(self.run_response.member_responses if self.run_response else []):
                member_id = None
                if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                    member_id = member_response.agent_id
                elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                    member_id = member_response.team_id
                if member_id:
                    if self.show_tool_calls and member_id in member_tool_calls and member_tool_calls[member_id]:
                        formatted_calls = format_tool_calls(member_tool_calls[member_id])
                        if formatted_calls:
                            console_width = console.width if console else 80
                            panel_width = console_width + 30
                            lines = []
                            for call in formatted_calls:
                                wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                lines.append(wrapped_call)
                            tool_calls_text = '\n\n'.join(lines)
                            member_name = self._get_member_name(member_id)
                            member_tool_calls_panel = create_panel(content=tool_calls_text, title=f'{member_name} Tool Calls', border_style='yellow')
                            final_panels.append(member_tool_calls_panel)
                    reasoning_steps = []
                    if member_response.extra_data is not None and member_response.extra_data.reasoning_steps is not None:
                        reasoning_steps = member_response.extra_data.reasoning_steps
                    if reasoning_steps and show_reasoning:
                        for j, step in enumerate(reasoning_steps, 1):
                            member_reasoning_panel = self._build_reasoning_step_panel(j, step, show_reasoning_verbose, color='magenta')
                            final_panels.append(member_reasoning_panel)
                    show_markdown = False
                    if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                        show_markdown = member_markdown.get(member_response.agent_id, False)
                    elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                        show_markdown = member_markdown.get(member_response.team_id, False)
                    member_response_content = self._parse_response_content(member_response, tags_to_include_in_markdown, show_markdown=show_markdown)
                    member_name = 'Team Member'
                    if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                        member_name = self._get_member_name(member_response.agent_id)
                    elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                        member_name = self._get_member_name(member_response.team_id)
                    member_response_panel = create_panel(content=member_response_content, title=f'{member_name} Response', border_style='magenta')
                    final_panels.append(member_response_panel)
                    if member_response.citations is not None and member_response.citations.urls is not None:
                        md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(member_response.citations.urls) if citation.url)
                        if md_content:
                            citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='magenta')
                            final_panels.append(citations_panel)
            if self.show_tool_calls and team_tool_calls:
                formatted_calls = format_tool_calls(team_tool_calls)
                if formatted_calls:
                    console_width = console.width if console else 80
                    panel_width = console_width + 30
                    lines = []
                    added_calls = set()
                    for call in formatted_calls:
                        if call not in added_calls:
                            added_calls.add(call)
                            wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                            lines.append(wrapped_call)
                    tool_calls_text = '\n\n'.join(lines)
                    team_tool_calls_panel = create_panel(content=tool_calls_text, title='Team Tool Calls', border_style='yellow')
                    final_panels.append(team_tool_calls_panel)
            if _response_content:
                response_content_stream = _response_content
                if team_markdown:
                    escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                    response_content_stream = Markdown(escaped_content)
                response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                final_panels.append(response_panel)
            if resp.citations is not None and resp.citations.urls is not None:
                md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(resp.citations.urls) if citation.url)
                if md_content:
                    citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                    final_panels.append(citations_panel)
            live_console.update(Group(*final_panels))

    async def aprint_response(self, message: Optional[Union[List, Dict, str, Message]] = None, *, stream: bool = False, stream_intermediate_steps: bool = False, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, console: Optional[Any] = None, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: Optional[bool] = None, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        if markdown is None:
            markdown = self.markdown
        else:
            self.markdown = markdown
        if self.response_model is not None:
            stream = False
        if stream:
            await self._aprint_response_stream(message=message, console=console, show_message=show_message, show_reasoning=show_reasoning, show_reasoning_verbose=show_reasoning_verbose, tags_to_include_in_markdown=tags_to_include_in_markdown, audio=audio, images=images, videos=videos, files=files, markdown=markdown, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
        else:
            await self._aprint_response(message=message, console=console, show_message=show_message, show_reasoning=show_reasoning, show_reasoning_verbose=show_reasoning_verbose, tags_to_include_in_markdown=tags_to_include_in_markdown, audio=audio, images=images, videos=videos, files=files, markdown=markdown, **kwargs)

    async def _aprint_response(self, message: Optional[Union[List, Dict, str, Message]] = None, console: Optional[Any] = None, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: bool = False, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        with Live(console=console) as live_console:
            status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
            live_console.update(status)
            response_timer = Timer()
            response_timer.start()
            panels = [status]
            if message and show_message:
                message_content = get_text_from_message(message)
                message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                panels.append(message_panel)
                live_console.update(Group(*panels))
            run_response: TeamRunResponse = await self.arun(message=message, images=images, audio=audio, videos=videos, files=files, stream=False, **kwargs)
            response_timer.stop()
            team_markdown = False
            member_markdown = {}
            if markdown:
                for member in self.members:
                    if isinstance(member, Agent) and member.agent_id is not None:
                        member_markdown[member.agent_id] = True
                    if isinstance(member, Team) and member.team_id is not None:
                        member_markdown[member.team_id] = True
                team_markdown = True
            if self.response_model is not None:
                team_markdown = False
            for member in self.members:
                if member.response_model is not None and isinstance(member, Agent) and member.agent_id is not None:
                    member_markdown[member.agent_id] = False
                if member.response_model is not None and isinstance(member, Team) and member.team_id is not None:
                    member_markdown[member.team_id] = False
            reasoning_steps = []
            if isinstance(run_response, TeamRunResponse) and run_response.extra_data is not None and run_response.extra_data.reasoning_steps is not None:
                reasoning_steps = run_response.extra_data.reasoning_steps
            if len(reasoning_steps) > 0 and show_reasoning:
                for i, step in enumerate(reasoning_steps, 1):
                    reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                    panels.append(reasoning_panel)
                live_console.update(Group(*panels))
            if isinstance(run_response, TeamRunResponse) and run_response.thinking is not None:
                thinking_panel = create_panel(content=Text(run_response.thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                panels.append(thinking_panel)
                live_console.update(Group(*panels))
            if isinstance(run_response, TeamRunResponse):
                if self.show_members_responses:
                    for member_response in run_response.member_responses:
                        reasoning_steps = []
                        if isinstance(member_response, RunResponse) and member_response.extra_data is not None and member_response.extra_data.reasoning_steps is not None:
                            reasoning_steps.extend(member_response.extra_data.reasoning_steps)
                        if len(reasoning_steps) > 0 and show_reasoning:
                            for i, step in enumerate(reasoning_steps, 1):
                                member_reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose, color='magenta')
                                panels.append(member_reasoning_panel)
                        if self.show_tool_calls and hasattr(member_response, 'tools') and member_response.tools:
                            member_name = None
                            if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                                member_name = self._get_member_name(member_response.agent_id)
                            elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                                member_name = self._get_member_name(member_response.team_id)
                            if member_name:
                                formatted_calls = format_tool_calls(member_response.tools)
                                if formatted_calls:
                                    console_width = console.width if console else 80
                                    panel_width = console_width + 30
                                    lines = []
                                    for call in formatted_calls:
                                        wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                        lines.append(wrapped_call)
                                    tool_calls_text = '\n\n'.join(lines)
                                    member_tool_calls_panel = create_panel(content=tool_calls_text, title=f'{member_name} Tool Calls', border_style='yellow')
                                    panels.append(member_tool_calls_panel)
                                    live_console.update(Group(*panels))
                        show_markdown = False
                        if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                            show_markdown = member_markdown.get(member_response.agent_id, False)
                        elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                            show_markdown = member_markdown.get(member_response.team_id, False)
                        member_response_content: Union[str, JSON, Markdown] = self._parse_response_content(member_response, tags_to_include_in_markdown, show_markdown=show_markdown)
                        if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                            member_response_panel = create_panel(content=member_response_content, title=f'{self._get_member_name(member_response.agent_id)} Response', border_style='magenta')
                        elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                            member_response_panel = create_panel(content=member_response_content, title=f'{self._get_member_name(member_response.team_id)} Response', border_style='magenta')
                        panels.append(member_response_panel)
                        if member_response.citations is not None and member_response.citations.urls is not None:
                            md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(member_response.citations.urls) if citation.url)
                            if md_content:
                                citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='magenta')
                                panels.append(citations_panel)
                    live_console.update(Group(*panels))
                if self.show_tool_calls and run_response.tools:
                    formatted_calls = format_tool_calls(run_response.tools)
                    if formatted_calls:
                        console_width = console.width if console else 80
                        panel_width = console_width + 30
                        lines = []
                        for call in formatted_calls:
                            wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                            lines.append(wrapped_call)
                        tool_calls_text = '\n\n'.join(lines)
                        team_tool_calls_panel = create_panel(content=tool_calls_text, title='Team Tool Calls', border_style='yellow')
                        panels.append(team_tool_calls_panel)
                        live_console.update(Group(*panels))
                response_content_batch: Union[str, JSON, Markdown] = self._parse_response_content(run_response, tags_to_include_in_markdown, show_markdown=team_markdown)
                response_panel = create_panel(content=response_content_batch, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                panels.append(response_panel)
                if run_response.citations is not None and run_response.citations.urls is not None:
                    md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(run_response.citations.urls) if citation.url)
                    if md_content:
                        citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                        panels.append(citations_panel)
            panels = [p for p in panels if not isinstance(p, Status)]
            live_console.update(Group(*panels))

    async def _aprint_response_stream(self, message: Optional[Union[List, Dict, str, Message]] = None, console: Optional[Any] = None, show_message: bool = True, show_reasoning: bool = True, show_reasoning_verbose: bool = False, tags_to_include_in_markdown: Optional[Set[str]] = None, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, markdown: bool = False, stream_intermediate_steps: bool = False, **kwargs: Any) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {'think', 'thinking'}
        stream_intermediate_steps = True
        self.run_response = cast(TeamRunResponse, self.run_response)
        _response_content: str = ''
        _response_thinking: str = ''
        reasoning_steps: List[ReasoningStep] = []
        member_tool_calls = {}
        team_tool_calls = []
        processed_tool_calls = set()
        final_panels = []
        with Live(console=console) as live_console:
            status = Status('Thinking...', spinner='aesthetic', speed=0.4, refresh_per_second=10)
            live_console.update(status)
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
                live_console.update(Group(*panels))
            stream_resp = await self.arun(message=message, audio=audio, images=images, videos=videos, files=files, stream=True, stream_intermediate_steps=stream_intermediate_steps, **kwargs)
            team_markdown = None
            member_markdown = {}
            async for resp in stream_resp:
                if team_markdown is None:
                    if markdown:
                        team_markdown = True
                    else:
                        team_markdown = False
                    if self.response_model is not None:
                        team_markdown = False
                if isinstance(resp, TeamRunResponse):
                    if resp.event == RunEvent.run_response:
                        if isinstance(resp.content, str):
                            _response_content += resp.content
                        if resp.thinking is not None:
                            _response_thinking += resp.thinking
                    if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                        reasoning_steps = resp.extra_data.reasoning_steps
                    if self.show_tool_calls and resp.tools:
                        for tool in resp.tools:
                            tool_id = tool.get('tool_call_id', str(hash(str(tool))))
                            if tool_id not in processed_tool_calls:
                                processed_tool_calls.add(tool_id)
                                team_tool_calls.append(tool)
                if self.show_tool_calls and resp.member_responses:
                    for member_response in resp.member_responses:
                        member_id = None
                        if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                            member_id = member_response.agent_id
                        elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                            member_id = member_response.team_id
                        if member_id and hasattr(member_response, 'tools') and member_response.tools:
                            if member_id not in member_tool_calls:
                                member_tool_calls[member_id] = []
                            for tool in member_response.tools:
                                tool_id = tool.get('tool_call_id', str(hash(str(tool))))
                                if tool_id not in processed_tool_calls:
                                    processed_tool_calls.add(tool_id)
                                    member_tool_calls[member_id].append(tool)
                response_content_stream: Union[str, Markdown] = _response_content
                if team_markdown:
                    escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                    response_content_stream = Markdown(escaped_content)
                panels = [status]
                if message and show_message:
                    render = True
                    message_content = get_text_from_message(message)
                    message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                    panels.append(message_panel)
                if render:
                    live_console.update(Group(*panels))
                if len(reasoning_steps) > 0 and show_reasoning:
                    render = True
                    for i, step in enumerate(reasoning_steps, 1):
                        reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                        panels.append(reasoning_panel)
                if render:
                    live_console.update(Group(*panels))
                if len(_response_thinking) > 0:
                    render = True
                    thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                    panels.append(thinking_panel)
                if render:
                    live_console.update(Group(*panels))
                if self.show_tool_calls and resp is not None and resp.formatted_tool_calls:
                    render = True
                    tool_calls_content = Text()
                    added_tool_calls = set()
                    for tool_call in resp.formatted_tool_calls:
                        if tool_call not in added_tool_calls:
                            added_tool_calls.add(tool_call)
                            tool_calls_content.append(f'• {tool_call}\n')
                    tool_calls_panel = create_panel(content=tool_calls_content.plain.rstrip(), title='Tool Calls', border_style='yellow')
                    panels.append(tool_calls_panel)
                if len(_response_content) > 0:
                    render = True
                    response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                    panels.append(response_panel)
                if render:
                    live_console.update(Group(*panels))
            response_timer.stop()
            if resp.citations is not None and resp.citations.urls is not None:
                md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(resp.citations.urls) if citation.url)
                if md_content:
                    citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                    panels.append(citations_panel)
                    live_console.update(Group(*panels))
            panels = [p for p in panels if not isinstance(p, Status)]
            if markdown:
                for member in self.members:
                    if isinstance(member, Agent) and member.agent_id is not None:
                        member_markdown[member.agent_id] = True
                    if isinstance(member, Team) and member.team_id is not None:
                        member_markdown[member.team_id] = True
            for member in self.members:
                if member.response_model is not None and isinstance(member, Agent) and member.agent_id is not None:
                    member_markdown[member.agent_id] = False
                if member.response_model is not None and isinstance(member, Team) and member.team_id is not None:
                    member_markdown[member.team_id] = False
            final_panels = []
            if message and show_message:
                message_content = get_text_from_message(message)
                message_panel = create_panel(content=Text(message_content, style='green'), title='Message', border_style='cyan')
                final_panels.append(message_panel)
            if reasoning_steps and show_reasoning:
                for i, step in enumerate(reasoning_steps, 1):
                    reasoning_panel = self._build_reasoning_step_panel(i, step, show_reasoning_verbose)
                    final_panels.append(reasoning_panel)
            if _response_thinking:
                thinking_panel = create_panel(content=Text(_response_thinking), title=f'Thinking ({response_timer.elapsed:.1f}s)', border_style='green')
                final_panels.append(thinking_panel)
            for i, member_response in enumerate(self.run_response.member_responses if self.run_response else []):
                member_id = None
                if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                    member_id = member_response.agent_id
                elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                    member_id = member_response.team_id
                if member_id:
                    if self.show_tool_calls and member_id in member_tool_calls and member_tool_calls[member_id]:
                        formatted_calls = format_tool_calls(member_tool_calls[member_id])
                        if formatted_calls:
                            console_width = console.width if console else 80
                            panel_width = console_width + 30
                            lines = []
                            added_calls = set()
                            for call in formatted_calls:
                                if call not in added_calls:
                                    added_calls.add(call)
                                    wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                                    lines.append(wrapped_call)
                            tool_calls_text = '\n\n'.join(lines)
                            member_name = self._get_member_name(member_id)
                            member_tool_calls_panel = create_panel(content=tool_calls_text, title=f'{member_name} Tool Calls', border_style='yellow')
                            final_panels.append(member_tool_calls_panel)
                    reasoning_steps = []
                    if member_response.extra_data is not None and member_response.extra_data.reasoning_steps is not None:
                        reasoning_steps = member_response.extra_data.reasoning_steps
                    if reasoning_steps and show_reasoning:
                        for j, step in enumerate(reasoning_steps, 1):
                            member_reasoning_panel = self._build_reasoning_step_panel(j, step, show_reasoning_verbose, color='magenta')
                            final_panels.append(member_reasoning_panel)
                    show_markdown = False
                    if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                        show_markdown = member_markdown.get(member_response.agent_id, False)
                    elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                        show_markdown = member_markdown.get(member_response.team_id, False)
                    member_response_content = self._parse_response_content(member_response, tags_to_include_in_markdown, show_markdown=show_markdown)
                    member_name = 'Team Member'
                    if isinstance(member_response, RunResponse) and member_response.agent_id is not None:
                        member_name = self._get_member_name(member_response.agent_id)
                    elif isinstance(member_response, TeamRunResponse) and member_response.team_id is not None:
                        member_name = self._get_member_name(member_response.team_id)
                    member_response_panel = create_panel(content=member_response_content, title=f'{member_name} Response', border_style='magenta')
                    final_panels.append(member_response_panel)
                    if member_response.citations is not None and member_response.citations.urls is not None:
                        md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(member_response.citations.urls) if citation.url)
                        if md_content:
                            citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='magenta')
                            final_panels.append(citations_panel)
            if self.show_tool_calls and team_tool_calls:
                formatted_calls = format_tool_calls(team_tool_calls)
                if formatted_calls:
                    console_width = console.width if console else 80
                    panel_width = console_width + 30
                    lines = []
                    added_calls = set()
                    for call in formatted_calls:
                        if call not in added_calls:
                            added_calls.add(call)
                            wrapped_call = textwrap.fill(f'• {call}', width=panel_width, subsequent_indent='  ')
                            lines.append(wrapped_call)
                    tool_calls_text = '\n\n'.join(lines)
                    team_tool_calls_panel = create_panel(content=tool_calls_text, title='Team Tool Calls', border_style='yellow')
                    final_panels.append(team_tool_calls_panel)
            if _response_content:
                response_content_stream = _response_content
                if team_markdown:
                    escaped_content = escape_markdown_tags(_response_content, tags_to_include_in_markdown)
                    response_content_stream = Markdown(escaped_content)
                response_panel = create_panel(content=response_content_stream, title=f'Response ({response_timer.elapsed:.1f}s)', border_style='blue')
                final_panels.append(response_panel)
            if resp.citations is not None and resp.citations.urls is not None:
                md_content = '\n'.join(f'{i + 1}. [{citation.title or citation.url}]({citation.url})' for i, citation in enumerate(resp.citations.urls) if citation.url)
                if md_content:
                    citations_panel = create_panel(content=Markdown(md_content), title='Citations', border_style='green')
                    final_panels.append(citations_panel)
            live_console.update(Group(*final_panels))

    def _build_reasoning_step_panel(self, step_idx: int, step: ReasoningStep, show_reasoning_verbose: bool = False, color: str = 'green'):
        step_content = Text.assemble()
        if step.title is not None:
            step_content.append(f'{step.title}\n', 'bold')
        if step.action is not None:
            step_content.append(f'{step.action}\n', 'dim')
        if step.result is not None:
            step_content.append(Text.from_markup(step.result, style='dim'))
        if show_reasoning_verbose:
            if step.reasoning is not None:
                step_content.append(Text.from_markup(f'\n[bold]Reasoning:[/bold] {step.reasoning}', style='dim'))
            if step.confidence is not None:
                step_content.append(Text.from_markup(f'\n[bold]Confidence:[/bold] {step.confidence}', style='dim'))
        return create_panel(content=step_content, title=f'Reasoning step {step_idx}', border_style=color)

    def _get_member_name(self, entity_id: str) -> str:
        for member in self.members:
            if isinstance(member, Agent):
                if member.agent_id == entity_id:
                    return member.name or entity_id
            elif isinstance(member, Team):
                if member.team_id == entity_id:
                    return member.name or entity_id
        return entity_id

    def _parse_response_content(self, run_response: Union[TeamRunResponse, RunResponse], tags_to_include_in_markdown: Set[str], show_markdown: bool = True) -> Any:
        if isinstance(run_response.content, str):
            if show_markdown:
                escaped_content = escape_markdown_tags(run_response.content, tags_to_include_in_markdown)
                return Markdown(escaped_content)
            else:
                return run_response.get_content_as_string(indent=4)
        elif isinstance(run_response.content, BaseModel):
            try:
                return JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
            except Exception as e:
                print(f'Failed to convert response to JSON: {e}')
        else:
            try:
                return JSON(json.dumps(run_response.content), indent=4)
            except Exception as e:
                print(f'Failed to convert response to JSON: {e}')

    def cli_app(self, message: Optional[str] = None, user: str = 'User', emoji: str = ':sunglasses:', stream: bool = False, markdown: bool = False, exit_on: Optional[List[str]] = None, **kwargs: Any) -> None:
        if message:
            self.print_response(message=message, stream=stream, markdown=markdown, **kwargs)
        _exit_on = exit_on or ['exit', 'quit', 'bye']
        while True:
            message = Prompt.ask(f'[bold] {emoji} {user} [/bold]')
            if message in _exit_on:
                break
            self.print_response(message=message, stream=stream, markdown=markdown, **kwargs)

    def _calculate_session_metrics(self) -> SessionMetrics:
        self.memory = cast(TeamMemory, self.memory)
        session_metrics = SessionMetrics()
        assistant_message_role = self.model.assistant_message_role if self.model is not None else 'assistant'
        for m in self.memory.messages:
            if m.role == assistant_message_role and m.metrics is not None:
                session_metrics += m.metrics
        return session_metrics

    def _calculate_full_team_session_metrics(self) -> SessionMetrics:
        current_session_metrics = self.session_metrics or self._calculate_session_metrics()
        current_session_metrics = replace(current_session_metrics)
        assistant_message_role = self.model.assistant_message_role if self.model is not None else 'assistant'
        for member in self.members:
            if member.memory is not None:
                for m in member.memory.messages:
                    if m.role == assistant_message_role and m.metrics is not None:
                        current_session_metrics += m.metrics
        return current_session_metrics

    def _aggregate_metrics_from_messages(self, messages: List[Message]) -> Dict[str, Any]:
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

    def _get_reasoning_agent(self, reasoning_model: Model) -> Optional[Agent]:
        return Agent(model=reasoning_model, monitoring=self.monitoring, telemetry=self.telemetry, debug_mode=self.debug_mode)

    def _reason(self, run_response: TeamRunResponse, run_messages: RunMessages, stream_intermediate_steps: bool = False) -> Iterator[TeamRunResponse]:
        if stream_intermediate_steps:
            yield self._create_run_response(from_run_response=run_response, content='Reasoning started', event=RunEvent.reasoning_started)
        print('Reasoning started')
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            reasoning_model = self.model.__class__(id=self.model.id)
        if reasoning_model is None:
            print('Reasoning error. Reasoning model is None, continuing regular session...')
            return
        if reasoning_model_provided and self.reasoning_model is not None:
            reasoning_message: Optional[Message] = None
            if self.reasoning_model.__class__.__name__ == 'DeepSeek' and self.reasoning_model.id.lower() == 'deepseek-reasoner':
                reasoning_agent = self._get_reasoning_agent(self.reasoning_model)
                reasoning_message = get_deepseek_reasoning(reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages())
                if reasoning_message is None:
                    print('Reasoning error. Reasoning response is None, continuing regular session...')
                    return
            elif reasoning_model.__class__.__name__ == 'OpenAIChat' and reasoning_model.id.startswith('o3'):
                reasoning_agent = self._get_reasoning_agent(self.reasoning_model)
                reasoning_message = get_openai_reasoning(reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages())
                if reasoning_message is None:
                    print('Reasoning error. Reasoning response is None, continuing regular session...')
                    return
            else:
                print(f'Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model.')
            if reasoning_message:
                run_messages.messages.append(reasoning_message)
                update_run_response_with_reasoning(run_response=run_response, reasoning_steps=[ReasoningStep(result=reasoning_message.content)], reasoning_agent_messages=[reasoning_message])
        else:
            use_json_mode: bool = self.use_json_mode
            reasoning_agent: Agent = get_default_reasoning_agent(reasoning_model=reasoning_model, min_steps=self.reasoning_min_steps, max_steps=self.reasoning_max_steps, monitoring=self.monitoring, telemetry=self.telemetry, debug_mode=self.debug_mode, use_json_mode=use_json_mode)
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
                        print('Reasoning error. Reasoning response is empty, continuing regular session...')
                        break
                    if reasoning_agent_response.content.reasoning_steps is None:
                        print('Reasoning error. Reasoning steps are empty, continuing regular session...')
                        break
                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    if stream_intermediate_steps:
                        for reasoning_step in reasoning_steps:
                            yield self._create_run_response(content=reasoning_step, content_type=reasoning_step.__class__.__name__, event=RunEvent.reasoning_step)
                    first_assistant_index = next((i for i, m in enumerate(reasoning_agent_response.messages) if m.role == 'assistant'), len(reasoning_agent_response.messages))
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]
                    update_run_response_with_reasoning(run_response=run_response, reasoning_steps=reasoning_steps, reasoning_agent_messages=reasoning_agent_response.messages)
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    print(f'Reasoning error: {e}')
                    break
            print(f'Total Reasoning steps: {len(all_reasoning_steps)}')
            print('Reasoning finished')
            update_messages_with_reasoning(run_messages=run_messages, reasoning_messages=reasoning_messages)
        if stream_intermediate_steps:
            yield self._create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]), content_type=ReasoningSteps.__class__.__name__, event=RunEvent.reasoning_completed)

    async def _areason(self, run_response: TeamRunResponse, run_messages: RunMessages, stream_intermediate_steps: bool = False) -> AsyncIterator[TeamRunResponse]:
        if stream_intermediate_steps:
            yield self._create_run_response(from_run_response=run_response, content='Reasoning started', event=RunEvent.reasoning_started)
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            reasoning_model = self.model.__class__(id=self.model.id)
        if reasoning_model is None:
            print('Reasoning error. Reasoning model is None, continuing regular session...')
            return
        if reasoning_model_provided and self.reasoning_model is not None:
            reasoning_message: Optional[Message] = None
            if self.reasoning_model.__class__.__name__ == 'DeepSeek' and self.reasoning_model.id.lower() == 'deepseek-reasoner':
                reasoning_agent = self._get_reasoning_agent(self.reasoning_model)
                reasoning_message = await aget_deepseek_reasoning(reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages())
                if reasoning_message is None:
                    print('Reasoning error. Reasoning response is None, continuing regular session...')
                    return
            elif reasoning_model.__class__.__name__ == 'OpenAIChat' and reasoning_model.id.startswith('o3'):
                reasoning_agent = self._get_reasoning_agent(self.reasoning_model)
                reasoning_message = await aget_openai_reasoning(reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages())
                if reasoning_message is None:
                    print('Reasoning error. Reasoning response is None, continuing regular session...')
                    return
            else:
                print(f'Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model.')
            if reasoning_message:
                run_messages.messages.append(reasoning_message)
                update_run_response_with_reasoning(run_response=run_response, reasoning_steps=[ReasoningStep(result=reasoning_message.content)], reasoning_agent_messages=[reasoning_message])
        else:
            use_json_mode: bool = self.use_json_mode
            reasoning_agent: Agent = get_default_reasoning_agent(reasoning_model=reasoning_model, min_steps=self.reasoning_min_steps, max_steps=self.reasoning_max_steps, monitoring=self.monitoring, telemetry=self.telemetry, debug_mode=self.debug_mode, use_json_mode=use_json_mode)
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
                        print('Reasoning error. Reasoning response is empty, continuing regular session...')
                        break
                    if reasoning_agent_response.content.reasoning_steps is None:
                        print('Reasoning error. Reasoning steps are empty, continuing regular session...')
                        break
                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    if stream_intermediate_steps:
                        for reasoning_step in reasoning_steps:
                            yield self._create_run_response(content=reasoning_step, content_type=reasoning_step.__class__.__name__, event=RunEvent.reasoning_step)
                    first_assistant_index = next((i for i, m in enumerate(reasoning_agent_response.messages) if m.role == 'assistant'), len(reasoning_agent_response.messages))
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]
                    update_run_response_with_reasoning(run_response=run_response, reasoning_steps=reasoning_steps, reasoning_agent_messages=reasoning_agent_response.messages)
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    print(f'Reasoning error: {e}')
                    break
            print(f'Total Reasoning steps: {len(all_reasoning_steps)}')
            print('Reasoning finished')
            update_messages_with_reasoning(run_messages=run_messages, reasoning_messages=reasoning_messages)
        if stream_intermediate_steps:
            yield self._create_run_response(content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]), content_type=ReasoningSteps.__class__.__name__, event=RunEvent.reasoning_completed)

    def _create_run_response(self, content: Optional[Any] = None, content_type: Optional[str] = None, thinking: Optional[str] = None, event: RunEvent = RunEvent.run_response, tools: Optional[List[Dict[str, Any]]] = None, audio: Optional[List[AudioArtifact]] = None, images: Optional[List[ImageArtifact]] = None, videos: Optional[List[VideoArtifact]] = None, response_audio: Optional[AudioResponse] = None, citations: Optional[Citations] = None, model: Optional[str] = None, messages: Optional[List[Message]] = None, created_at: Optional[int] = None, from_run_response: Optional[TeamRunResponse] = None) -> TeamRunResponse:
        extra_data = None
        member_responses = None
        formatted_tool_calls = None
        if from_run_response:
            content = from_run_response.content
            content_type = from_run_response.content_type
            tools = from_run_response.tools
            audio = from_run_response.audio
            images = from_run_response.images
            videos = from_run_response.videos
            response_audio = from_run_response.response_audio
            model = from_run_response.model
            messages = from_run_response.messages
            extra_data = from_run_response.extra_data
            member_responses = from_run_response.member_responses
            citations = from_run_response.citations
            tools = from_run_response.tools
            formatted_tool_calls = from_run_response.formatted_tool_calls
        rr = TeamRunResponse(run_id=self.run_id, session_id=self.session_id, team_id=self.team_id, content=content, thinking=thinking, tools=tools, audio=audio, images=images, videos=videos, response_audio=response_audio, citations=citations, model=model, messages=messages, extra_data=extra_data, event=event.value)
        if formatted_tool_calls:
            rr.formatted_tool_calls = formatted_tool_calls
        if member_responses:
            rr.member_responses = member_responses
        if content_type is not None:
            rr.content_type = content_type
        if created_at is not None:
            rr.created_at = created_at
        return rr

    def _resolve_run_context(self) -> None:
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

    def _configure_model(self, show_tool_calls: bool = False) -> None:
        if self.model is None:
            print('Setting default model to OpenAI Chat')
            self.model = Ollama()
        if self.response_model is None:
            self.model.response_format = None
        else:
            json_response_format = {'type': 'json_object'}
            if self.model.supports_native_structured_outputs:
                if self.use_json_mode:
                    self.model.response_format = json_response_format
                    self.model.structured_outputs = False
                else:
                    print('Setting Model.response_format to Agent.response_model')
                    self.model.response_format = self.response_model
                    self.model.structured_outputs = True
            elif self.model.supports_json_schema_outputs:
                if self.use_json_mode:
                    print('Setting Model.response_format to JSON response mode')
                    self.model.response_format = {'type': 'json_schema', 'json_schema': {'name': self.response_model.__name__, 'schema': self.response_model.model_json_schema()}}
                else:
                    self.model.response_format = None
                self.model.structured_outputs = False
            else:
                self.model.response_format = json_response_format if self.use_json_mode else None
                self.model.structured_outputs = False
        self.model.show_tool_calls = show_tool_calls
        if self.tool_choice is not None:
            self.model.tool_choice = self.tool_choice
        if self.tool_call_limit is not None:
            self.model.tool_call_limit = self.tool_call_limit

    def _add_tools_to_model(self, model: Model, tools: List[Union[Function, Callable, Toolkit, Dict]]) -> None:
        self._functions_for_model = {}
        self._tools_for_model = []
        if len(tools) > 0:
            print('Processing tools for model')
            strict = False
            if self.response_model is not None and not self.use_json_mode and model.supports_native_structured_outputs:
                strict = True
            for tool in tools:
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

    def get_members_system_message_content(self, indent: int = 0) -> str:
        system_message_content = ''
        for idx, member in enumerate(self.members):
            if isinstance(member, Team):
                system_message_content += f'{indent * " "} - Team: {member.name}\n'
                if member.members is not None:
                    system_message_content += member.get_members_system_message_content(indent=indent + 2)
            else:
                system_message_content += f'{indent * " "} - Agent {idx + 1}:\n'
                if member.name is not None:
                    system_message_content += f'{indent * " "}   - Name: {member.name}\n'
                if member.role is not None:
                    system_message_content += f'{indent * " "}   - Role: {member.role}\n'
                if member.description is not None:
                    system_message_content += f'{indent * " "}   - Description: {member.description}\n'
                if member.tools is not None:
                    system_message_content += f'{indent * " "}   - Available tools:\n'
                    tool_name_and_description = []
                    for _tool in member.tools:
                        if isinstance(_tool, Toolkit):
                            for _func in _tool.functions.values():
                                if _func.entrypoint:
                                    tool_name_and_description.append((_func.name, get_entrypoint_docstring(_func.entrypoint)))
                        elif isinstance(_tool, Function) and _tool.entrypoint:
                            tool_name_and_description.append((_tool.name, get_entrypoint_docstring(_tool.entrypoint)))
                        elif callable(_tool):
                            tool_name_and_description.append((_tool.__name__, get_entrypoint_docstring(_tool)))
                    for _tool_name, _tool_description in tool_name_and_description:
                        system_message_content += f'{indent * " "}    - {_tool_name}: {_tool_description}\n'
        return system_message_content

    def get_system_message(self, audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None) -> Optional[Message]:
        self.model = cast(Model, self.model)
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
            additional_information.append(f'The current time is {datetime.now()}')
        system_message_content: str = ''
        if self.mode == 'coordinate':
            system_message_content += 'You are the leader of a team of AI Agents and possible Sub-Teams:\n'
            system_message_content += self.get_members_system_message_content()
            system_message_content += ('''\n-您可以直接响应，也可以将任务转移给团队中的其他代理，具体取决于他们可用的工具及其角色。\n-如果将任务转移给另一个代理，请确保包括：\n-agent_name（str）：要将任务传输到的代理的名称。\n-task_description（str）：任务的清晰描述。\n-expected_output（str）：预期输出。\n-您可以同时将任务传递给多个成员。\n-在响应用户之前，您必须始终验证其他代理的输出。\n-评估其他代理人的反应。如果你觉得任务已经完成，你可以停下来回应用户。\n如果你对结果不满意，可以重新分配任务。\n''')
        elif self.mode == 'route':
            system_message_content += 'You are the leader of a team of AI Agents and possible Sub-Teams:\n'
            system_message_content += self.get_members_system_message_content()
            system_message_content += '-您充当用户请求的路由器。您必须选择正确的代理来转发用户的请求。这应该是完成任务可能性最高的代理。\n-将任务转发给另一个代理时，请确保包括：\n-agent_name（str）：要将任务传输到的代理的名称。\n-expected_output（str）：预期输出。\n你应该尽最大努力把任务交给一个代理人。\n-如果用户请求需要它（例如，如果他们要求多个东西），您可以一次转发给多个代理。\n'
        elif self.mode == 'collaborate':
            system_message_content += 'You are leading a collaborative team of Agents and possible Sub-Teams:\n'
            system_message_content += self.get_members_system_message_content()
            system_message_content += '-对于团队中的所有代理，只调用run_member_agent一次。\n-考虑其他代理的所有回复，并评估任务是否已完成。\n如果你觉得任务已经完成，你可以停下来回应用户。\n'
            system_message_content += '\n'
        if self.enable_agentic_context:
            system_message_content += 'You can and should update the context of the team. Use the `set_team_context` tool to update the shared team context.\n'
        if self.name is not None:
            system_message_content += f'Your name is: {self.name}.\n\n'
        if self.success_criteria:
            system_message_content += f'<success_criteria>\nThe team will be considered successful if the following criteria are met: {self.success_criteria}\nStop the team run when the criteria are met.\n</success_criteria>\n\n'
        if self.description is not None:
            system_message_content += f'<description>\n{self.description}\n</description>\n\n'
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
        if audio is not None or images is not None or videos is not None or files is not None:
            system_message_content += '<attached_media>\n'
            system_message_content += 'You have the following media attached to your message:\n'
            if audio is not None and len(audio) > 0:
                system_message_content += ' - Audio\n'
            if images is not None and len(images) > 0:
                system_message_content += ' - Images\n'
            if videos is not None and len(videos) > 0:
                system_message_content += ' - Videos\n'
            if files is not None and len(files) > 0:
                system_message_content += ' - Files\n'
            system_message_content += '</attached_media>\n\n'
        if self.add_state_in_messages:
            system_message_content = self._format_message_with_state_variables(system_message_content)
        system_message_from_model = self.model.get_system_message_for_model()
        if system_message_from_model is not None:
            system_message_content += system_message_from_model
        if self.expected_output is not None:
            system_message_content += f'<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n'
        if self.response_model is not None and self.use_json_mode and self.model and self.model.supports_native_structured_outputs:
            system_message_content += f'{self._get_json_output_prompt()}'
        return Message(role='system', content=system_message_content.strip())

    def get_run_messages(self, *, run_response: TeamRunResponse, message: Union[str, List, Dict, Message], audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, **kwargs: Any) -> RunMessages:
        """此函数返回具有以下属性的RunMessages对象：
        -system_message：此运行的系统消息
        -user_message：此运行的用户消息
        -messages：要发送到模型的消息列表
        要构建RunMessages对象，请执行以下操作：
        1.将系统消息添加到run_message
        2.向run_message添加历史记录
        3.将用户消息添加到run_message
        """
        self.memory = cast(TeamMemory, self.memory)
        run_messages = RunMessages()
        system_message = self.get_system_message(images=images, audio=audio, videos=videos, files=files)
        if system_message is not None:
            run_messages.system_message = system_message
            run_messages.messages.append(system_message)
        if self.enable_team_history:
            history: List[Message] = self.memory.get_messages_from_last_n_runs(last_n=self.num_of_interactions_from_history, skip_role='system')
            if len(history) > 0:
                history_copy = [deepcopy(msg) for msg in history]
                for _msg in history_copy:
                    _msg.from_history = True
                print(f'Adding {len(history_copy)} messages from history')
                run_messages.messages += history_copy
        user_message = self._get_user_message(message, audio=audio, images=images, videos=videos, files=files, **kwargs)
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)
        return run_messages

    def _get_user_message(self, message: Union[str, List, Dict, Message], audio: Optional[Sequence[Audio]] = None, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, files: Optional[Sequence[File]] = None, **kwargs):
        user_message_content: str = ''
        if isinstance(message, str) or isinstance(message, list):
            if self.add_state_in_messages:
                if isinstance(message, str):
                    user_message_content = self._format_message_with_state_variables(message)
                elif isinstance(message, list):
                    user_message_content = '\n'.join([self._format_message_with_state_variables(msg) for msg in message])
            else:
                if isinstance(message, str):
                    user_message_content = message
                else:
                    user_message_content = '\n'.join(message)
            if self.add_context and self.context is not None:
                user_message_content += '\n\n<context>\n'
                user_message_content += self._convert_context_to_string(self.context) + '\n'
                user_message_content += '</context>'
            return Message(role='user', content=user_message_content, audio=audio, images=images, videos=videos, files=files, **kwargs)
        elif isinstance(message, Message):
            return message
        elif isinstance(message, dict):
            try:
                return Message.model_validate(message)
            except Exception as e:
                print(f'Failed to validate message: {e}')

    def _format_message_with_state_variables(self, message: str) -> Any:
        format_variables = ChainMap(self.session_state or {}, self.context or {}, self.extra_data or {}, {'user_id': self.user_id} if self.user_id is not None else {})
        return self._formatter.format(message, **format_variables)

    def _convert_context_to_string(self, context: Dict[str, Any]) -> str:
        if context is None:
            return ''
        try:
            return json.dumps(context, indent=2, default=str)
        except (TypeError, ValueError, OverflowError) as e:
            print(f'Failed to convert context to JSON: {e}')
            sanitized_context = {}
            for key, value in context.items():
                try:
                    json.dumps({key: value}, default=str)
                    sanitized_context[key] = value
                except Exception as e:
                    print(f'Failed to serialize to JSON: {e}')
                    sanitized_context[key] = str(value)
            try:
                return json.dumps(sanitized_context, indent=2)
            except Exception as e:
                print(f'Failed to convert sanitized context to JSON: {e}')
                return str(context)

    def _get_json_output_prompt(self) -> str:
        json_output_prompt = 'Provide your output as a JSON containing the following fields:'
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
                            response_model_properties[field_name] = formatted_field_properties
                    json_schema_defs = json_schema.get('$defs')
                    if json_schema_defs is not None:
                        response_model_properties['$defs'] = {}
                        for def_name, def_properties in json_schema_defs.items():
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
                print(f'Could not build json schema for {self.response_model}')
        else:
            json_output_prompt += 'Provide the output as JSON.'
        json_output_prompt += '\nStart your response with `{` and end it with `}`.'
        json_output_prompt += '\nYour output will be passed to json.loads() to convert it to a Python object.'
        json_output_prompt += '\nMake sure it only contains valid JSON.'
        return json_output_prompt

    def _update_team_state(self, run_response: Union[TeamRunResponse, RunResponse]) -> None:
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

    def get_team_history(self, num_chats: Optional[int] = None) -> str:
        history: List[Dict[str, Any]] = []
        all_chats = self.memory.get_all_messages()
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

    def set_team_context(self, state: Union[str, dict]) -> str:
        if isinstance(state, str):
            self.memory.set_team_context_text(state)
        elif isinstance(state, dict):
            self.memory.set_team_context_text(json.dumps(state))
        msg = f'Current team context: {self.memory.get_team_context_str()}'
        print(msg)
        return msg

    def get_run_member_agents_function(self, stream: bool = False, async_mode: bool = False, images: Optional[List[Image]] = None, videos: Optional[List[Video]] = None, audio: Optional[List[Audio]] = None, files: Optional[List[File]] = None) -> Function:
        if not images:
            images = []
        if not videos:
            videos = []
        if not audio:
            audio = []
        if not files:
            files = []
        def run_member_agents(task_description: str, expected_output: Optional[str] = None) -> Iterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            team_context_str = None
            if self.enable_agentic_context:
                team_context_str = self.memory.get_team_context_str()
            team_member_interactions_str = None
            if self.share_member_interactions:
                team_member_interactions_str = self.memory.get_team_member_interactions_str()
                if context_images := self.memory.get_team_context_images():
                    images.extend([Image.from_artifact(img) for img in context_images])
                if context_videos := self.memory.get_team_context_videos():
                    videos.extend([Video.from_artifact(vid) for vid in context_videos])
                if context_audio := self.memory.get_team_context_audio():
                    audio.extend([Audio.from_artifact(aud) for aud in context_audio])
            member_agent_task = 'You are a member of a team of agents that collaborate to complete a task.'
            if expected_output is not None:
                member_agent_task += f'\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if team_context_str:
                member_agent_task += f'\n\n{team_context_str}'
            if team_member_interactions_str:
                member_agent_task += f'\n\n{team_member_interactions_str}'
            member_agent_task += f'\n\n<task>\n{task_description}\n</task>'
            for member_agent_index, member_agent in enumerate(self.members):
                if stream:
                    member_agent_run_response_stream = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=True)
                    for member_agent_run_response_chunk in member_agent_run_response_stream:
                        check_if_run_cancelled(member_agent_run_response_chunk)
                        yield member_agent_run_response_chunk.content or ''
                else:
                    member_agent_run_response = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
                    check_if_run_cancelled(member_agent_run_response)
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
                            yield json.dumps(member_agent_run_response.content, indent=2)
                        except Exception as e:
                            yield str(e)
                member_name = member_agent.name if member_agent.name else f'agent_{member_agent_index}'
                self.memory = cast(TeamMemory, self.memory)
                self.memory.add_interaction_to_team_context(member_name=member_name, task=task_description, run_response=member_agent.run_response)
                self.run_response = cast(TeamRunResponse, self.run_response)
                self.run_response.add_member_run(member_agent.run_response)
                self._update_team_state(member_agent.run_response)

        async def arun_member_agents(task_description: str, expected_output: Optional[str] = None) -> AsyncIterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            team_context_str = None
            if self.enable_agentic_context:
                team_context_str = self.memory.get_team_context_str()
            team_member_interactions_str = None
            if self.share_member_interactions:
                team_member_interactions_str = self.memory.get_team_member_interactions_str()
                if context_images := self.memory.get_team_context_images():
                    images.extend([Image.from_artifact(img) for img in context_images])
                if context_videos := self.memory.get_team_context_videos():
                    videos.extend([Video.from_artifact(vid) for vid in context_videos])
                if context_audio := self.memory.get_team_context_audio():
                    audio.extend([Audio.from_artifact(aud) for aud in context_audio])
            member_agent_task = 'You are a member of a team of agents that collaborate to complete a task.'
            if expected_output is not None:
                member_agent_task += f'\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if team_context_str:
                member_agent_task += f'\n\n{team_context_str}'
            if team_member_interactions_str:
                member_agent_task += f'\n\n{team_member_interactions_str}'
            member_agent_task += f'\n\n<task>\n{task_description}\n</task>'
            tasks = []
            for member_agent_index, member_agent in enumerate(self.members):
                current_agent = member_agent
                current_index = member_agent_index
                async def run_member_agent(agent=current_agent, idx=current_index) -> str:
                    response = await agent.arun(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
                    check_if_run_cancelled(response)
                    member_name = agent.name if agent.name else f'agent_{idx}'
                    self.memory = cast(TeamMemory, self.memory)
                    self.memory.add_interaction_to_team_context(member_name=member_name, task=task_description, run_response=agent.run_response)
                    self.run_response = cast(TeamRunResponse, self.run_response)
                    self.run_response.add_member_run(agent.run_response)
                    self._update_team_state(agent.run_response)
                    if response.content is None:
                        return f'Agent {member_name}: No response from the member agent.'
                    elif isinstance(response.content, str):
                        return f'Agent {member_name}: {response.content}'
                    elif issubclass(type(response.content), BaseModel):
                        try:
                            return f'Agent {member_name}: {response.content.model_dump_json(indent=2)}'
                        except Exception as e:
                            return f'Agent {member_name}: Error - {str(e)}'
                    else:
                        try:
                            return f'Agent {member_name}: {json.dumps(response.content, indent=2)}'
                        except Exception as e:
                            return f'Agent {member_name}: Error - {str(e)}'
                tasks.append(run_member_agent)
            results = await asyncio.gather(*[task() for task in tasks])
            for result in results:
                yield result
        if async_mode:
            run_member_agents_function = arun_member_agents
        else:
            run_member_agents_function = run_member_agents
        run_member_agents_func = Function.from_callable(run_member_agents_function, strict=True)
        return run_member_agents_func

    def get_transfer_task_function(self, stream: bool = False, async_mode: bool = False, images: Optional[List[Image]] = None, videos: Optional[List[Video]] = None, audio: Optional[List[Audio]] = None, files: Optional[List[File]] = None) -> Function:
        if not images:
            images = []
        if not videos:
            videos = []
        if not audio:
            audio = []
        if not files:
            files = []
        def transfer_task_to_member(agent_name: str, task_description: str, expected_output: str) -> Iterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            result = self._find_member_by_name(agent_name)
            if result is None:
                yield f'Agent with name {agent_name} not found in the team or any subteams. Please choose the correct agent from the list of agents.'
                return
            member_agent_index, member_agent = result
            self._initialize_member(member_agent)
            team_context_str = None
            if self.enable_agentic_context:
                team_context_str = self.memory.get_team_context_str()
            team_member_interactions_str = None
            if self.share_member_interactions:
                team_member_interactions_str = self.memory.get_team_member_interactions_str()
                if context_images := self.memory.get_team_context_images():
                    images.extend([Image.from_artifact(img) for img in context_images])
                if context_videos := self.memory.get_team_context_videos():
                    videos.extend([Video.from_artifact(vid) for vid in context_videos])
                if context_audio := self.memory.get_team_context_audio():
                    audio.extend([Audio.from_artifact(aud) for aud in context_audio])
            member_agent_task = f'You are a member of a team of agents. Your goal is to complete the following task:\n\n{task_description}\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if team_context_str:
                member_agent_task += f'\n\n{team_context_str}'
            if team_member_interactions_str:
                member_agent_task += f'\n\n{team_member_interactions_str}'
            if stream:
                member_agent_run_response_stream = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=True)
                for member_agent_run_response_chunk in member_agent_run_response_stream:
                    check_if_run_cancelled(member_agent_run_response_chunk)
                    yield member_agent_run_response_chunk.content or ''
            else:
                member_agent_run_response = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
                check_if_run_cancelled(member_agent_run_response)
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
                        yield json.dumps(member_agent_run_response.content, indent=2)
                    except Exception as e:
                        yield str(e)
            member_name = member_agent.name if member_agent.name else f'agent_{member_agent_index}'
            self.memory = cast(TeamMemory, self.memory)
            self.memory.add_interaction_to_team_context(member_name=member_name, task=task_description, run_response=member_agent.run_response)
            self.run_response = cast(TeamRunResponse, self.run_response)
            self.run_response.add_member_run(member_agent.run_response)
            self._update_team_state(member_agent.run_response)

        async def atransfer_task_to_member(agent_name: str, task_description: str, expected_output: str) -> AsyncIterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            result = self._find_member_by_name(agent_name)
            if result is None:
                yield f'Agent with name {agent_name} not found in the team or any subteams. Please choose the correct agent from the list of agents.'
                return
            member_agent_index, member_agent = result
            self._initialize_member(member_agent)
            team_context_str = None
            if self.enable_agentic_context:
                team_context_str = self.memory.get_team_context_str()
            team_member_interactions_str = None
            if self.share_member_interactions:
                team_member_interactions_str = self.memory.get_team_member_interactions_str()
                if context_images := self.memory.get_team_context_images():
                    images.extend([Image.from_artifact(img) for img in context_images])
                if context_videos := self.memory.get_team_context_videos():
                    videos.extend([Video.from_artifact(vid) for vid in context_videos])
                if context_audio := self.memory.get_team_context_audio():
                    audio.extend([Audio.from_artifact(aud) for aud in context_audio])
            member_agent_task = f'You are a member of a team of agents. Your goal is to complete the following task:\n\n{task_description}\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if team_context_str:
                member_agent_task += f'\n\n{team_context_str}'
            if team_member_interactions_str:
                member_agent_task += f'\n\n{team_member_interactions_str}'
            if stream:
                member_agent_run_response_stream = await member_agent.arun(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=True)
                async for member_agent_run_response_chunk in member_agent_run_response_stream:
                    check_if_run_cancelled(member_agent_run_response_chunk)
                    yield member_agent_run_response_chunk.content or ''
            else:
                member_agent_run_response = await member_agent.arun(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
                check_if_run_cancelled(member_agent_run_response)
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
                        yield json.dumps(member_agent_run_response.content, indent=2)
                    except Exception as e:
                        yield str(e)
            member_name = member_agent.name if member_agent.name else f'agent_{member_agent_index}'
            self.memory.add_interaction_to_team_context(member_name=member_name, task=task_description, run_response=member_agent.run_response)
            self.run_response = cast(TeamRunResponse, self.run_response)
            self.run_response.add_member_run(member_agent.run_response)
            self._update_team_state(member_agent.run_response)
        if async_mode:
            transfer_function = atransfer_task_to_member
        else:
            transfer_function = transfer_task_to_member
        transfer_func = Function.from_callable(transfer_function, strict=True)
        return transfer_func

    def _find_member_by_name(self, agent_name: str):
        for i, member in enumerate(self.members):
            if member.name == agent_name:
                return (i, member)
            if isinstance(member, Team):
                result = member._find_member_by_name(agent_name)
                if result is not None:
                    return (i, member)
        return None

    def get_forward_task_function(self, message: Message, stream: bool = False, async_mode: bool = False, images: Optional[Sequence[Image]] = None, videos: Optional[Sequence[Video]] = None, audio: Optional[Sequence[Audio]] = None, files: Optional[Sequence[File]] = None) -> Function:
        if not images:
            images = []
        if not videos:
            videos = []
        if not audio:
            audio = []
        if not files:
            files = []
        def forward_task_to_member(agent_name: str, expected_output: Optional[str] = None) -> Iterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            self._member_response_model = None
            result = self._find_member_by_name(agent_name)
            if result is None:
                yield f'Agent with name {agent_name} not found in the team or any subteams. Please choose the correct agent from the list of agents.'
                return
            member_agent_index, member_agent = result
            self._initialize_member(member_agent)
            if member_agent.response_model is not None:
                self._member_response_model = member_agent.response_model
            member_agent_task = message.get_content_string()
            if expected_output:
                member_agent_task += f'\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if stream:
                member_agent_run_response_stream = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=True)
                for member_agent_run_response_chunk in member_agent_run_response_stream:
                    yield member_agent_run_response_chunk.content or ''
            else:
                member_agent_run_response = member_agent.run(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
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
                        yield json.dumps(member_agent_run_response.content, indent=2)
                    except Exception as e:
                        yield str(e)
            member_name = member_agent.name if member_agent.name else f'agent_{member_agent_index}'
            self.memory = cast(TeamMemory, self.memory)
            self.memory.add_interaction_to_team_context(member_name=member_name, task=message.get_content_string(), run_response=member_agent.run_response)
            self.run_response = cast(TeamRunResponse, self.run_response)
            self.run_response.add_member_run(member_agent.run_response)
            self._update_team_state(member_agent.run_response)

        async def aforward_task_to_member(agent_name: str, expected_output: Optional[str] = None) -> AsyncIterator[str]:
            self.memory = cast(TeamMemory, self.memory)
            self._member_response_model = None
            result = self._find_member_by_name(agent_name)
            if result is None:
                yield f'Agent with name {agent_name} not found in the team or any subteams. Please choose the correct agent from the list of agents.'
                return
            member_agent_index, member_agent = result
            self._initialize_member(member_agent)
            if member_agent.response_model is not None:
                self._member_response_model = member_agent.response_model
            member_agent_task = message.get_content_string()
            if expected_output:
                member_agent_task += f'\n\n<expected_output>\n{expected_output}\n</expected_output>'
            if stream:
                member_agent_run_response_stream = await member_agent.arun(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=True)
                async for member_agent_run_response_chunk in member_agent_run_response_stream:
                    check_if_run_cancelled(member_agent_run_response_chunk)
                    yield member_agent_run_response_chunk.content or ''
            else:
                member_agent_run_response = await member_agent.arun(member_agent_task, images=images, videos=videos, audio=audio, files=files, stream=False)
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
                        yield json.dumps(member_agent_run_response.content, indent=2)
                    except Exception as e:
                        yield str(e)
            member_name = member_agent.name if member_agent.name else f'agent_{member_agent_index}'
            self.memory = cast(TeamMemory, self.memory)
            self.memory.add_interaction_to_team_context(member_name=member_name, task=message.get_content_string(), run_response=member_agent.run_response)
            self.run_response = cast(TeamRunResponse, self.run_response)
            self.run_response.add_member_run(member_agent.run_response)
            self._update_team_state(member_agent.run_response)
        if async_mode:
            forward_function = aforward_task_to_member
        else:
            forward_function = forward_task_to_member
        forward_func = Function.from_callable(forward_function, strict=True)
        forward_func.stop_after_tool_call = True
        forward_func.show_result = True
        return forward_func

    def load_user_memories(self) -> None:
        self.memory = cast(TeamMemory, self.memory)
        if self.memory and self.memory.create_user_memories:
            if self.user_id is not None:
                self.memory.user_id = self.user_id
            self.memory.load_user_memories()
            if self.user_id is not None:
                print(f'Memories loaded for user: {self.user_id}')
            else:
                print('Memories loaded')

    def read_from_storage(self) -> Optional[TeamSession]:
        if self.storage is not None and self.session_id is not None:
            self.team_session = cast(TeamSession, self.storage.read(session_id=self.session_id))
            if self.team_session is not None:
                self.load_team_session(session=self.team_session)
            self.load_user_memories()
        return self.team_session

    def write_to_storage(self) -> Optional[TeamSession]:
        if self.storage is not None:
            self.team_session = cast(TeamSession, self.storage.upsert(session=self._get_team_session()))
        return self.team_session

    def rename_session(self, session_name: str) -> None:
        self.read_from_storage()
        self.session_name = session_name
        self.write_to_storage()
        self._log_team_session()

    def delete_session(self, session_id: str) -> None:
        if self.storage is not None:
            self.storage.delete_session(session_id=session_id)

    def load_team_session(self, session: TeamSession):
        if self.team_id is None and session.team_id is not None:
            self.team_id = session.team_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if session.team_data is not None:
            if self.name is None and 'name' in session.team_data:
                self.name = session.team_data.get('name')
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
        if not isinstance(self.memory, TeamMemory):
            if isinstance(self.memory, dict):
                self.memory = TeamMemory(**self.memory)
            elif self.memory is not None:
                raise TypeError(f'Expected memory to be a dict or TeamMemory, but got {type(self.memory)}')
        if session.memory is not None and self.memory is not None:
            try:
                if 'runs' in session.memory:
                    try:
                        self.memory.runs = [TeamRun.from_dict(m) for m in session.memory['runs']]
                    except Exception as e:
                        print(f'Failed to load runs from memory: {e}')
                if 'messages' in session.memory:
                    try:
                        self.memory.messages = [Message.model_validate(m) for m in session.memory['messages']]
                    except Exception as e:
                        print(f'Failed to load messages from memory: {e}')
                if 'memories' in session.memory:
                    try:
                        self.memory.memories = [Memory.model_validate(m) for m in session.memory['memories']]
                    except Exception as e:
                        print(f'Failed to load user memories: {e}')
            except Exception as e:
                print(f'Failed to load AgentMemory: {e}')
        print(f'-*- TeamSession loaded: {session.session_id}')

    def _create_run_data(self) -> Dict[str, Any]:
        run_response_format = 'text'
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

    def _get_team_data(self) -> Dict[str, Any]:
        team_data: Dict[str, Any] = {}
        if self.name is not None:
            team_data['name'] = self.name
        if self.team_id is not None:
            team_data['team_id'] = self.team_id
        if self.model is not None:
            team_data['model'] = self.model.to_dict()
        return team_data

    def _get_session_data(self) -> Dict[str, Any]:
        session_data: Dict[str, Any] = {}
        if self.session_name is not None:
            session_data['session_name'] = self.session_name
        if self.session_state is not None and len(self.session_state) > 0:
            session_data['session_state'] = self.session_state
        if self.session_metrics is not None:
            session_data['session_metrics'] = asdict(self.session_metrics) if self.session_metrics is not None else None
        if self.images is not None:
            session_data['images'] = [img.model_dump() for img in self.images]
        if self.videos is not None:
            session_data['videos'] = [vid.model_dump() for vid in self.videos]
        if self.audio is not None:
            session_data['audio'] = [aud.model_dump() for aud in self.audio]
        return session_data

    def _get_team_session(self) -> TeamSession:
        return TeamSession(session_id=self.session_id, team_id=self.team_id, user_id=self.user_id, team_session_id=self.team_session_id, memory=self.memory.to_dict() if self.memory is not None else None, team_data=self._get_team_data(), session_data=self._get_session_data(), extra_data=self.extra_data, created_at=int(time()))

    def _log_team_run(self) -> None:
        if not self.telemetry and not self.monitoring:
            return
        try:
            run_data = self._create_run_data()
            team_session: TeamSession = self.team_session or self._get_team_session()
        except Exception as e:
            print(f'Could not create team event: {e}')
    async def _alog_team_run(self) -> None:
        if not self.telemetry and not self.monitoring:
            return
        try:
            run_data = self._create_run_data()
            team_session: TeamSession = self.team_session or self._get_team_session()
        except Exception as e:
            print(f'Could not create team event: {e}')

    def _log_team_session(self):
        if not (self.telemetry or self.monitoring):
            return
        try:
            team_session: TeamSession = self.team_session or self._get_team_session()
        except Exception as e:
            print(f'Could not create team monitor: {e}')

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> 'Team':
        attributes = self.__dict__.copy()
        excluded_fields = ['team_session', 'session_name', '_functions_for_model']
        copied_attributes = {}
        for field_name, field_value in attributes.items():
            if field_name in excluded_fields:
                continue
            copied_attributes[field_name] = self._deep_copy_field(field_name, field_value)
        team_copy = Team.__new__(Team)
        team_copy.__dict__ = copied_attributes
        if update:
            for key, value in update.items():
                setattr(team_copy, key, value)
        return team_copy

    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        if field_name == 'members':
            if field_value is not None:
                return [member.deep_copy() for member in field_value]
            return None
        if field_name == 'memory' and field_value is not None:
            return field_value.deep_copy()
        elif field_name in ('storage', 'model', 'reasoning_model') and field_value is not None:
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
            except Exception as e:
                print(f'Failed to deepcopy field: {field_name} - {e}')
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
            return copy(field_value)
        except Exception:
            return field_value
