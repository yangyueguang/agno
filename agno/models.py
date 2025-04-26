import json
import asyncio
import collections.abc
from uuid import uuid4
from enum import Enum
from time import time, perf_counter
from abc import ABC, abstractmethod
from types import AsyncGeneratorType, GeneratorType
from agno.tools import AgentRunException, Function, FunctionCall
from dataclasses import asdict, dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from agno.media import Audio, AudioResponse, File, Image, ImageArtifact, Video
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple, Union, Sequence


class Timer:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return self.elapsed_time or (perf_counter() - self.start_time) if self.start_time else 0.0

    def start(self) -> float:
        self.start_time = perf_counter()
        return self.start_time

    def stop(self) -> float:
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
        return self.end_time

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time


class MessageReferences(BaseModel):
    query: str
    references: Optional[List[Dict[str, Any]]] = None
    time: Optional[float] = None


class DocumentCitation(BaseModel):
    document_title: Optional[str] = None
    cited_text: Optional[str] = None
    file_name: Optional[str] = None


class Citations(BaseModel):
    raw: Optional[Any] = None
    urls: Optional[List[Dict]] = None
    documents: Optional[List[DocumentCitation]] = None


@dataclass
class MessageMetrics:
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

    def __add__(self, other: 'MessageMetrics') -> 'MessageMetrics':
        result = MessageMetrics(input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens, total_tokens=self.total_tokens + other.total_tokens, prompt_tokens=self.prompt_tokens + other.prompt_tokens, completion_tokens=self.completion_tokens + other.completion_tokens)
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

    def __radd__(self, other: 'MessageMetrics') -> 'MessageMetrics':
        if other == 0:
            return self
        return self + other


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool'] = 'user'
    content: Optional[Union[List[Any], str]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Sequence[Audio]] = None
    images: Optional[Sequence[Image]] = None
    videos: Optional[Sequence[Video]] = None
    files: Optional[Sequence[File]] = None
    audio_output: Optional[AudioResponse] = None
    image_output: Optional[ImageArtifact] = None
    thinking: Optional[str] = None
    redacted_thinking: Optional[str] = None
    provider_data: Optional[Dict[str, Any]] = None
    citations: Optional[Citations] = None
    reasoning_content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_call_error: Optional[bool] = None
    stop_after_tool_call: bool = False
    add_to_agent_memory: bool = True
    from_history: bool = False
    metrics: MessageMetrics = Field(default_factory=MessageMetrics)
    references: Optional[MessageReferences] = None
    created_at: int = Field(default_factory=lambda: int(time()))
    model_config = ConfigDict(extra='allow', populate_by_name=True, arbitrary_types_allowed=True)

    def get_content_string(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            if len(self.content) > 0 and isinstance(self.content[0], dict) and 'text' in self.content[0]:
                return self.content[0].get('text', '')
            else:
                return json.dumps(self.content)
        return ''

    def to_dict(self) -> Dict[str, Any]:
        message_dict = {'content': self.content, 'reasoning_content': self.reasoning_content, 'from_history': self.from_history, 'stop_after_tool_call': self.stop_after_tool_call, 'role': self.role, 'name': self.name, 'tool_call_id': self.tool_call_id, 'tool_name': self.tool_name, 'tool_args': self.tool_args, 'tool_call_error': self.tool_call_error, 'tool_calls': self.tool_calls, 'thinking': self.thinking, 'redacted_thinking': self.redacted_thinking}
        message_dict = {k: v for k, v in message_dict.items() if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)}
        if self.images:
            message_dict['images'] = [img.to_dict() for img in self.images]
        if self.audio:
            message_dict['audio'] = [aud.to_dict() for aud in self.audio]
        if self.videos:
            message_dict['videos'] = [vid.to_dict() for vid in self.videos]
        if self.audio_output:
            message_dict['audio_output'] = self.audio_output.to_dict()
        if self.references:
            message_dict['references'] = self.references.model_dump()
        if self.metrics:
            mdict = asdict(self.metrics)
            mdict.pop('timer')
            message_dict['metrics'] = {k: v for k, v in mdict.items() if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)}
            if not message_dict['metrics']:
                message_dict.pop('metrics')
        message_dict['created_at'] = self.created_at
        return message_dict

    def to_function_call_dict(self) -> Dict[str, Any]:
        return {'content': self.content, 'tool_call_id': self.tool_call_id, 'tool_name': self.tool_name, 'tool_args': self.tool_args, 'tool_call_error': self.tool_call_error, 'metrics': self.metrics, 'created_at': self.created_at}

    def log(self, metrics: bool = True, level=None):
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
        except Exception:
            terminal_width = 80
        header = f' {self.role} '
        print(f'{header.center(terminal_width - 20, "=")}')
        if self.name:
            print(f'名字: {self.name}')
        if self.tool_call_id:
            print(f'Tool call Id: {self.tool_call_id}')
        if self.thinking:
            print(f'<thinking>\n{self.thinking}\n</thinking>')
        if self.content:
            if isinstance(self.content, str) or isinstance(self.content, list):
                print(self.content)
            elif isinstance(self.content, dict):
                print(json.dumps(self.content, indent=2))
        if self.tool_calls:
            tool_calls_list = ['Tool Calls:']
            for tool_call in self.tool_calls:
                tool_id = tool_call.get('id')
                function_name = tool_call.get('function', {}).get('name')
                tool_calls_list.append(f'  - ID: "{tool_id}"') if tool_id else None
                tool_calls_list.append(f'    Name: "{function_name}"') if function_name else None
                tool_call_arguments = tool_call.get('function', {}).get('arguments')
                if tool_call_arguments:
                    try:
                        arguments = ', '.join(f'{k}: {v}' for k, v in json.loads(tool_call_arguments).items())
                        tool_calls_list.append(f'    Arguments: "{arguments}"')
                    except json.JSONDecodeError:
                        tool_calls_list.append('    Arguments: "Invalid JSON format"')
            tool_calls_str = '\n'.join(tool_calls_list)
            print(tool_calls_str)
        if self.images:
            print(f'图片: {len(self.images)}')
        if self.videos:
            print(f'视频: {len(self.videos)}')
        if self.audio:
            print(f'声音: {len(self.audio)}')
        if self.files:
            print(f'文件: {len(self.files)}')
        metrics_header = ' TOOL METRICS ' if self.role == 'tool' else ' METRICS '
        if metrics and self.metrics is not None and self.metrics != MessageMetrics():
            print(metrics_header)
            token_metrics = []
            if self.metrics.input_tokens:
                token_metrics.append(f'input={self.metrics.input_tokens}')
            if self.metrics.output_tokens:
                token_metrics.append(f'output={self.metrics.output_tokens}')
            if self.metrics.total_tokens:
                token_metrics.append(f'total={self.metrics.total_tokens}')
            if token_metrics:
                print(f'* Tokens:                      {", ".join(token_metrics)}')
            if self.metrics.prompt_tokens_details:
                print(f'* Prompt tokens details:       {self.metrics.prompt_tokens_details}')
            if self.metrics.completion_tokens_details:
                print(f'* Completion tokens details:   {self.metrics.completion_tokens_details}')
            if self.metrics.time is not None:
                print(f'* Time:                        {self.metrics.time:.4f}s')
            if self.metrics.output_tokens and self.metrics.time:
                print(f'* Tokens per second:           {self.metrics.output_tokens / self.metrics.time:.4f} tokens/s')
            if self.metrics.time_to_first_token is not None:
                print(f'* Time to first token:         {self.metrics.time_to_first_token:.4f}s')
            if self.metrics.additional_metrics:
                print(f'* Additional metrics:          {self.metrics.additional_metrics}')
            print(metrics_header)

    def content_is_valid(self) -> bool:
        return self.content is not None and len(self.content) > 0


class ModelResponseEvent(str, Enum):
    tool_call_started = 'ToolCallStarted'
    tool_call_completed = 'ToolCallCompleted'
    assistant_response = 'AssistantResponse'


@dataclass
class ModelResponse:
    role: Optional[str] = None
    content: Optional[str] = None
    parsed: Optional[Any] = None
    audio: Optional[AudioResponse] = None
    image: Optional[ImageArtifact] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    event: str = ModelResponseEvent.assistant_response.value
    provider_data: Optional[Dict[str, Any]] = None
    thinking: Optional[str] = None
    redacted_thinking: Optional[str] = None
    reasoning_content: Optional[str] = None
    citations: Optional[Citations] = None
    response_usage: Optional[Any] = None
    created_at: int = int(time())
    extra: Optional[Dict[str, Any]] = None


class FileType(str, Enum):
    MP4 = 'mp4'
    GIF = 'gif'
    MP3 = 'mp3'


def get_function_call(name: str, arguments: Optional[str] = None, call_id: Optional[str] = None, functions: Optional[Dict[str, Function]] = None) -> Optional[FunctionCall]:
    print(f'Getting function {name}')
    if functions is None:
        return None
    function_to_call: Optional[Function] = None
    if name in functions:
        function_to_call = functions[name]
    if function_to_call is None:
        print(f'Function {name} not found')
        return None
    function_call = FunctionCall(function=function_to_call)
    if call_id is not None:
        function_call.call_id = call_id
    if arguments is not None and arguments != '':
        try:
            if function_to_call.sanitize_arguments:
                if 'None' in arguments:
                    arguments = arguments.replace('None', 'null')
                if 'True' in arguments:
                    arguments = arguments.replace('True', 'true')
                if 'False' in arguments:
                    arguments = arguments.replace('False', 'false')
            _arguments = json.loads(arguments)
        except Exception as e:
            print(f'无法解码函数参数:\n{arguments}\nError: {e}')
            function_call.error = f'解码函数参数时出错:{e}\n\n请确保我们可以json.loads()参数并重试。'
            return function_call
        if not isinstance(_arguments, dict):
            print(f'函数参数不是有效的JSON对象: {arguments}')
            function_call.error = '函数参数不是有效的JSON对象。\n\n请修复并重试。'
            return function_call
        try:
            clean_arguments: Dict[str, Any] = {}
            for k, v in _arguments.items():
                if isinstance(v, str):
                    _v = v.strip().lower()
                    if _v in ('none', 'null'):
                        clean_arguments[k] = None
                    elif _v == 'true':
                        clean_arguments[k] = True
                    elif _v == 'false':
                        clean_arguments[k] = False
                    else:
                        clean_arguments[k] = v.strip()
                else:
                    clean_arguments[k] = v
            function_call.arguments = clean_arguments
        except Exception as e:
            print(f'无法解析函数参数:\n{arguments}\nError: {e}')
            function_call.error = f'解析函数参数时出错:{e}\n\n请修复并重试。'
            return function_call
    return function_call


def get_function_call_for_tool_call(tool_call: Dict[str, Any], functions: Optional[Dict[str, Function]] = None) -> Optional[FunctionCall]:
    if tool_call.get('type') == 'function':
        _tool_call_id = tool_call.get('id')
        _tool_call_function = tool_call.get('function')
        if _tool_call_function is not None:
            _tool_call_function_name = _tool_call_function.get('name')
            _tool_call_function_arguments_str = _tool_call_function.get('arguments')
            if _tool_call_function_name is not None:
                return get_function_call(name=_tool_call_function_name, arguments=_tool_call_function_arguments_str, call_id=_tool_call_id, functions=functions)
    return None


@dataclass
class MessageData:
    response_role: Optional[Literal['system', 'user', 'assistant', 'tool']] = None
    response_content: Any = ''
    response_thinking: Any = ''
    response_redacted_thinking: Any = ''
    response_citations: Optional[Citations] = None
    response_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_audio: Optional[AudioResponse] = None
    response_image: Optional[ImageArtifact] = None
    response_provider_data: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Model(ABC):
    id: str
    name: Optional[str] = None
    provider: Optional[str] = None
    response_format: Optional[Any] = None  # 不要直接设置，而是在代理上设置response_model属性
    structured_outputs: bool = False
    supports_native_structured_outputs: bool = False
    supports_json_schema_outputs: bool = False
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    show_tool_calls: Optional[bool] = None
    tool_call_limit: Optional[int] = None
    _tools: Optional[List[Dict]] = None
    _functions: Optional[Dict[str, Function]] = None
    _function_call_stack: Optional[List[FunctionCall]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[List[str]] = None
    tool_message_role: str = 'tool'
    assistant_message_role: str = 'assistant'

    def __post_init__(self):
        if self.provider is None and self.name is not None:
            self.provider = f'{self.name} ({self.id})'

    def to_dict(self) -> Dict[str, Any]:
        fields = {'name', 'id', 'provider'}
        _dict = {field: getattr(self, field) for field in fields if getattr(self, field) is not None}
        if self._functions:
            _dict['functions'] = {k: v.to_dict() for k, v in self._functions.items()}
            _dict['tool_call_limit'] = self.tool_call_limit
        return _dict

    def get_provider(self) -> str:
        return self.provider or self.name or self.__class__.__name__

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def ainvoke(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def invoke_stream(self, *args, **kwargs) -> Iterator[Any]:
        pass

    @abstractmethod
    async def ainvoke_stream(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        pass

    @abstractmethod
    def parse_provider_response(self, response: Any) -> ModelResponse:
        pass

    @abstractmethod
    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        pass

    def set_tools(self, tools: List[Dict]) -> None:
        self._tools = tools

    def set_functions(self, functions: Dict[str, Function]) -> None:
        if len(functions) > 0:
            self._functions = functions

    def response(self, messages: List[Message]) -> ModelResponse:
        print(f'{self.get_provider()} Response Start\nModel: {self.id}')
        self._log_messages(messages)
        model_response = ModelResponse()
        while True:
            assistant_message, has_tool_calls = self._process_model_response(messages=messages, model_response=model_response)
            if has_tool_calls:
                function_calls_to_run = self._prepare_function_calls(assistant_message=assistant_message, messages=messages, model_response=model_response)
                function_call_results: List[Message] = []
                for function_call_response in self.run_function_calls(function_calls=function_calls_to_run, function_call_results=function_call_results):
                    if function_call_response.event == ModelResponseEvent.tool_call_completed.value and function_call_response.tool_calls is not None:
                        model_response.tool_calls.extend(function_call_response.tool_calls)
                    elif function_call_response.event not in [ModelResponseEvent.tool_call_started.value, ModelResponseEvent.tool_call_completed.value]:
                        if function_call_response.content:
                            model_response.content += function_call_response.content
                self.format_function_call_results(messages=messages, function_call_results=function_call_results, **model_response.extra or {})
                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)
                if any(m.stop_after_tool_call for m in function_call_results):
                    break
                continue
            break
        print(f'{self.get_provider()} Response End')
        return model_response

    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        print(f'{self.get_provider()} Async Response Start\nModel: {self.id}')
        self._log_messages(messages)
        model_response = ModelResponse()
        while True:
            assistant_message, has_tool_calls = await self._aprocess_model_response(messages=messages, model_response=model_response)
            if has_tool_calls:
                function_calls_to_run = self._prepare_function_calls(assistant_message=assistant_message, messages=messages, model_response=model_response)
                function_call_results: List[Message] = []
                async for function_call_response in self.arun_function_calls(function_calls=function_calls_to_run, function_call_results=function_call_results):
                    if function_call_response.event == ModelResponseEvent.tool_call_completed.value and function_call_response.tool_calls is not None:
                        model_response.tool_calls.extend(function_call_response.tool_calls)
                    elif function_call_response.event not in [ModelResponseEvent.tool_call_started.value, ModelResponseEvent.tool_call_completed.value]:
                        if function_call_response.content:
                            model_response.content += function_call_response.content
                self.format_function_call_results(messages=messages, function_call_results=function_call_results, **model_response.extra or {})
                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)
                if any(m.stop_after_tool_call for m in function_call_results):
                    break
                continue
            break
        print(f'{self.get_provider()} Async Response End')
        return model_response

    def _process_model_response(self, messages: List[Message], model_response: ModelResponse) -> Tuple[Message, bool]:
        assistant_message = Message(role=self.assistant_message_role)
        assistant_message.metrics.start_timer()
        response = self.invoke(messages=messages)
        assistant_message.metrics.stop_timer()
        provider_response: ModelResponse = self.parse_provider_response(response)
        if provider_response.parsed is not None:
            model_response.parsed = provider_response.parsed
        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)
        messages.append(assistant_message)
        assistant_message.log(metrics=True)
        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.thinking is not None:
            model_response.thinking = assistant_message.thinking
        if assistant_message.redacted_thinking is not None:
            model_response.redacted_thinking = assistant_message.redacted_thinking
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.image = assistant_message.image_output
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)
        return assistant_message, bool(assistant_message.tool_calls)

    async def _aprocess_model_response(self, messages: List[Message], model_response: ModelResponse) -> Tuple[Message, bool]:
        assistant_message = Message(role=self.assistant_message_role)
        assistant_message.metrics.start_timer()
        response = await self.ainvoke(messages=messages)
        assistant_message.metrics.stop_timer()
        provider_response: ModelResponse = self.parse_provider_response(response)
        if provider_response.parsed is not None:
            model_response.parsed = provider_response.parsed
        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)
        messages.append(assistant_message)
        assistant_message.log(metrics=True)
        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.thinking is not None:
            model_response.thinking = assistant_message.thinking
        if assistant_message.redacted_thinking is not None:
            model_response.redacted_thinking = assistant_message.redacted_thinking
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.image = assistant_message.image_output
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)
        return assistant_message, bool(assistant_message.tool_calls)

    def _populate_assistant_message(self, assistant_message: Message, provider_response: ModelResponse) -> Message:
        if provider_response.role is not None:
            assistant_message.role = provider_response.role
        if provider_response.content is not None:
            assistant_message.content = provider_response.content
        if provider_response.tool_calls is not None and len(provider_response.tool_calls) > 0:
            assistant_message.tool_calls = provider_response.tool_calls
        if provider_response.audio is not None:
            assistant_message.audio_output = provider_response.audio
        if provider_response.image is not None:
            assistant_message.image_output = provider_response.image
        if provider_response.thinking is not None:
            assistant_message.thinking = provider_response.thinking
        if provider_response.redacted_thinking is not None:
            assistant_message.redacted_thinking = provider_response.redacted_thinking
        if provider_response.reasoning_content is not None:
            assistant_message.reasoning_content = provider_response.reasoning_content
        if provider_response.provider_data is not None:
            assistant_message.provider_data = provider_response.provider_data
        if provider_response.citations is not None:
            assistant_message.citations = provider_response.citations
        if provider_response.response_usage is not None:
            self._add_usage_metrics_to_assistant_message(assistant_message=assistant_message, response_usage=provider_response.response_usage)
        return assistant_message

    def process_response_stream(self, messages: List[Message], assistant_message: Message, stream_data: MessageData) -> Iterator[ModelResponse]:
        for response_delta in self.invoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta)
            yield from self._populate_stream_data_and_assistant_message(stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta)

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        print(f'{self.get_provider()} Response Stream Start\nModel: {self.id}')
        self._log_messages(messages)
        while True:
            assistant_message = Message(role=self.assistant_message_role)
            stream_data = MessageData()
            assistant_message.metrics.start_timer()
            yield from self.process_response_stream(messages=messages, assistant_message=assistant_message, stream_data=stream_data)
            assistant_message.metrics.stop_timer()
            if stream_data.response_content:
                assistant_message.content = stream_data.response_content
            if stream_data.response_thinking:
                assistant_message.thinking = stream_data.response_thinking
            if stream_data.response_redacted_thinking:
                assistant_message.redacted_thinking = stream_data.response_redacted_thinking
            if stream_data.response_provider_data:
                assistant_message.provider_data = stream_data.response_provider_data
            if stream_data.response_citations:
                assistant_message.citations = stream_data.response_citations
            if stream_data.response_audio:
                assistant_message.audio_output = stream_data.response_audio
            if stream_data.response_tool_calls and len(stream_data.response_tool_calls) > 0:
                assistant_message.tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)
            messages.append(assistant_message)
            assistant_message.log(metrics=True)
            if assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(assistant_message, messages)
                function_call_results: List[Message] = []
                for function_call_response in self.run_function_calls(function_calls=function_calls_to_run, function_call_results=function_call_results):
                    yield function_call_response
                if stream_data.extra is not None:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results, **stream_data.extra)
                else:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results)
                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)
                if any(m.stop_after_tool_call for m in function_call_results):
                    break
                continue
            break
        print(f'{self.get_provider()} Response Stream End')

    async def aprocess_response_stream(self, messages: List[Message], assistant_message: Message, stream_data: MessageData) -> AsyncIterator[ModelResponse]:
        async for response_delta in self.ainvoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta)
            for model_response in self._populate_stream_data_and_assistant_message(stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta):
                yield model_response

    async def aresponse_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        print(f'{self.get_provider()} Async Response Stream Start\nModel: {self.id}')
        self._log_messages(messages)
        while True:
            assistant_message = Message(role=self.assistant_message_role)
            stream_data = MessageData()
            assistant_message.metrics.start_timer()
            async for response in self.aprocess_response_stream(messages=messages, assistant_message=assistant_message, stream_data=stream_data):
                yield response
            assistant_message.metrics.stop_timer()
            if stream_data.response_content:
                assistant_message.content = stream_data.response_content
            if stream_data.response_thinking:
                assistant_message.thinking = stream_data.response_thinking
            if stream_data.response_redacted_thinking:
                assistant_message.redacted_thinking = stream_data.response_redacted_thinking
            if stream_data.response_provider_data:
                assistant_message.provider_data = stream_data.response_provider_data
            if stream_data.response_audio:
                assistant_message.audio_output = stream_data.response_audio
            if stream_data.response_tool_calls and len(stream_data.response_tool_calls) > 0:
                assistant_message.tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)
            messages.append(assistant_message)
            assistant_message.log(metrics=True)
            if assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(assistant_message, messages)
                function_call_results: List[Message] = []
                async for function_call_response in self.arun_function_calls(function_calls=function_calls_to_run, function_call_results=function_call_results):
                    yield function_call_response
                if stream_data.extra is not None:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results, **stream_data.extra)
                else:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results)
                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)
                if any(m.stop_after_tool_call for m in function_call_results):
                    break
                continue
            break
        print(f'{self.get_provider()} Async Response Stream End')

    def _populate_stream_data_and_assistant_message(self, stream_data: MessageData, assistant_message: Message, model_response: ModelResponse) -> Iterator[ModelResponse]:
        if not assistant_message.metrics.time_to_first_token:
            assistant_message.metrics.set_time_to_first_token()
        if model_response.role is not None:
            assistant_message.role = model_response.role
        should_yield = False
        if model_response.content is not None:
            stream_data.response_content += model_response.content
            should_yield = True
        if model_response.thinking is not None:
            stream_data.response_thinking += model_response.thinking
            should_yield = True
        if model_response.redacted_thinking is not None:
            stream_data.response_redacted_thinking += model_response.redacted_thinking
            should_yield = True
        if model_response.citations is not None:
            stream_data.response_citations = model_response.citations
            should_yield = True
        if model_response.provider_data:
            if stream_data.response_provider_data is None:
                stream_data.response_provider_data = {}
            stream_data.response_provider_data.update(model_response.provider_data)
        if model_response.tool_calls is not None:
            if stream_data.response_tool_calls is None:
                stream_data.response_tool_calls = []
            stream_data.response_tool_calls.extend(model_response.tool_calls)
            should_yield = True
        if model_response.audio is not None:
            if stream_data.response_audio is None:
                stream_data.response_audio = AudioResponse(id=str(uuid4()), content='', transcript='')
            if model_response.audio.id is not None:
                stream_data.response_audio.id = model_response.audio.id
            if model_response.audio.content is not None:
                stream_data.response_audio.content += model_response.audio.content
            if model_response.audio.transcript is not None:
                stream_data.response_audio.transcript += model_response.audio.transcript
            if model_response.audio.expires_at is not None:
                stream_data.response_audio.expires_at = model_response.audio.expires_at
            if model_response.audio.mime_type is not None:
                stream_data.response_audio.mime_type = model_response.audio.mime_type
            stream_data.response_audio.sample_rate = model_response.audio.sample_rate
            stream_data.response_audio.channels = model_response.audio.channels
            should_yield = True
        if model_response.image:
            if stream_data.response_image is None:
                stream_data.response_image = model_response.image
        if model_response.extra is not None:
            if stream_data.extra is None:
                stream_data.extra = {}
            stream_data.extra.update(model_response.extra)
        if model_response.response_usage is not None:
            self._add_usage_metrics_to_assistant_message(assistant_message=assistant_message, response_usage=model_response.response_usage)
        if should_yield:
            yield model_response

    def parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tool_calls_data

    def get_function_calls_to_run(self, assistant_message: Message, messages: List[Message], error_response_role: str = 'user') -> List[FunctionCall]:
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls is not None:
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get('id')
                _function_call = get_function_call_for_tool_call(tool_call, self._functions)
                if _function_call is None:
                    messages.append(Message(role=error_response_role, content='Could not find function to call.'))
                    continue
                if _function_call.error is not None:
                    messages.append(Message(role=error_response_role, tool_call_id=_tool_call_id, content=_function_call.error))
                    continue
                function_calls_to_run.append(_function_call)
        return function_calls_to_run

    def _handle_agent_exception(self, a_exc: AgentRunException, additional_messages: List[Message]) -> None:
        if a_exc.user_message is not None:
            msg = (Message(role='user', content=a_exc.user_message)
                if isinstance(a_exc.user_message, str)
                else a_exc.user_message)
            additional_messages.append(msg)
        if a_exc.agent_message is not None:
            msg = (Message(role='assistant', content=a_exc.agent_message)
                if isinstance(a_exc.agent_message, str)
                else a_exc.agent_message)
            additional_messages.append(msg)
        if a_exc.messages:
            for m in a_exc.messages:
                if isinstance(m, Message):
                    additional_messages.append(m)
                elif isinstance(m, dict):
                    try:
                        additional_messages.append(Message(**m))
                    except Exception as e:
                        print(f'无法将字典转换为消息: {e}')
        if a_exc.stop_execution:
            for m in additional_messages:
                m.stop_after_tool_call = True

    def _create_function_call_result(self, fc: FunctionCall, success: bool, output: Optional[Union[List[Any], str]], timer: Timer) -> Message:
        return Message(role=self.tool_message_role, content=output if success else fc.error, tool_call_id=fc.call_id, tool_name=fc.function.name, tool_args=fc.arguments, tool_call_error=not success, stop_after_tool_call=fc.function.stop_after_tool_call, metrics=MessageMetrics(time=timer.elapsed))

    def run_function_calls(self, function_calls: List[FunctionCall], function_call_results: List[Message]) -> Iterator[ModelResponse]:
        if self._function_call_stack is None:
            self._function_call_stack = []
        additional_messages: List[Message] = []
        for fc in function_calls:
            function_call_timer = Timer()
            function_call_timer.start()
            yield ModelResponse(content=fc.get_call_str(), tool_calls=[{'role': self.tool_message_role, 'tool_call_id': fc.call_id, 'tool_name': fc.function.name, 'tool_args': fc.arguments}], event=ModelResponseEvent.tool_call_started.value)
            function_call_success = False
            try:
                function_call_success = fc.execute()
            except AgentRunException as a_exc:
                self._handle_agent_exception(a_exc, additional_messages)
                function_call_success = False
            except Exception as e:
                print(f'Error executing function {fc.function.name}: {e}')
                function_call_success = False
                raise e
            function_call_timer.stop()
            function_call_output: Optional[Union[List[Any], str]] = ''
            if isinstance(fc.result, (GeneratorType, collections.abc.Iterator)):
                for item in fc.result:
                    function_call_output += item
                    if fc.function.show_result:
                        yield ModelResponse(content=item)
            else:
                function_call_output = fc.result
                if fc.function.show_result:
                    yield ModelResponse(content=function_call_output)
            function_call_result = self._create_function_call_result(fc, function_call_success, function_call_output, function_call_timer)
            yield ModelResponse(content=f'{fc.get_call_str()} completed in {function_call_timer.elapsed:.4f}s.', tool_calls=[function_call_result.to_function_call_dict()], event=ModelResponseEvent.tool_call_completed.value)
            function_call_results.append(function_call_result)
            self._function_call_stack.append(fc)
            if self.tool_call_limit and len(self._function_call_stack) >= self.tool_call_limit:
                self.tool_choice = 'none'
                break
        if additional_messages:
            function_call_results.extend(additional_messages)

    async def _arun_function_call(self, function_call: FunctionCall) -> Tuple[Union[bool, AgentRunException], Timer, FunctionCall]:
        from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction
        function_call_timer = Timer()
        function_call_timer.start()
        success: Union[bool, AgentRunException] = False
        try:
            if iscoroutinefunction(function_call.function.entrypoint) or isasyncgenfunction(function_call.function.entrypoint) or iscoroutine(function_call.function.entrypoint):
                success = await function_call.aexecute()
            else:
                success = await asyncio.to_thread(function_call.execute)
        except AgentRunException as e:
            success = e
        except Exception as e:
            print(f'Error executing function {function_call.function.name}: {e}')
            success = False
            raise e
        function_call_timer.stop()
        return success, function_call_timer, function_call

    async def arun_function_calls(self, function_calls: List[FunctionCall], function_call_results: List[Message]):
        if self._function_call_stack is None:
            self._function_call_stack = []
        additional_messages: List[Message] = []
        for fc in function_calls:
            yield ModelResponse(content=fc.get_call_str(), tool_calls=[{'role': self.tool_message_role, 'tool_call_id': fc.call_id, 'tool_name': fc.function.name, 'tool_args': fc.arguments}], event=ModelResponseEvent.tool_call_started.value)
        results = await asyncio.gather(*(self._arun_function_call(fc) for fc in function_calls), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                print(f'Error during function call: {result}')
                raise result
            function_call_success, function_call_timer, fc = result
            if isinstance(function_call_success, AgentRunException):
                a_exc = function_call_success
                self._handle_agent_exception(a_exc, additional_messages)
                function_call_success = False
            function_call_output: Optional[Union[List[Any], str]] = ''
            if isinstance(fc.result, (GeneratorType, collections.abc.Iterator)):
                for item in fc.result:
                    function_call_output += item
                    if fc.function.show_result:
                        yield ModelResponse(content=item)
            elif isinstance(fc.result, (AsyncGeneratorType, collections.abc.AsyncIterator)):
                async for item in fc.result:
                    function_call_output += item
                    if fc.function.show_result:
                        yield ModelResponse(content=item)
            else:
                function_call_output = fc.result
                if fc.function.show_result:
                    yield ModelResponse(content=function_call_output)
            function_call_result = self._create_function_call_result(fc, function_call_success, function_call_output, function_call_timer)
            yield ModelResponse(content=f'{fc.get_call_str()} completed in {function_call_timer.elapsed:.4f}s.', tool_calls=[function_call_result.to_function_call_dict()], event=ModelResponseEvent.tool_call_completed.value)
            function_call_results.append(function_call_result)
            self._function_call_stack.append(fc)
            if self.tool_call_limit and len(self._function_call_stack) >= self.tool_call_limit:
                self.tool_choice = 'none'
                break
        if additional_messages:
            function_call_results.extend(additional_messages)

    def _prepare_function_calls(self, assistant_message: Message, messages: List[Message], model_response: ModelResponse) -> List[FunctionCall]:
        if model_response.content is None:
            model_response.content = ''
        if model_response.tool_calls is None:
            model_response.tool_calls = []
        function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(assistant_message, messages)
        return function_calls_to_run

    def format_function_call_results(self, messages: List[Message], function_call_results: List[Message], **kwargs) -> None:
        if len(function_call_results) > 0:
            messages.extend(function_call_results)

    def _add_usage_metrics_to_assistant_message(self, assistant_message: Message, response_usage: Any) -> None:
        if isinstance(response_usage, dict):
            if 'input_tokens' in response_usage:
                assistant_message.metrics.input_tokens = response_usage.get('input_tokens', 0)
            if 'output_tokens' in response_usage:
                assistant_message.metrics.output_tokens = response_usage.get('output_tokens', 0)
            if 'prompt_tokens' in response_usage:
                assistant_message.metrics.input_tokens = response_usage.get('prompt_tokens', 0)
            if 'completion_tokens' in response_usage:
                assistant_message.metrics.output_tokens = response_usage.get('completion_tokens', 0)
            if 'total_tokens' in response_usage:
                assistant_message.metrics.total_tokens = response_usage.get('total_tokens', 0)
            else:
                assistant_message.metrics.total_tokens = (assistant_message.metrics.input_tokens + assistant_message.metrics.output_tokens)
        else:
            if hasattr(response_usage, 'input_tokens') and response_usage.input_tokens:
                assistant_message.metrics.input_tokens = response_usage.input_tokens
            if hasattr(response_usage, 'output_tokens') and response_usage.output_tokens:
                assistant_message.metrics.output_tokens = response_usage.output_tokens
            if hasattr(response_usage, 'prompt_tokens') and response_usage.prompt_tokens is not None:
                assistant_message.metrics.input_tokens = response_usage.prompt_tokens
                assistant_message.metrics.prompt_tokens = response_usage.prompt_tokens
            if hasattr(response_usage, 'completion_tokens') and response_usage.completion_tokens is not None:
                assistant_message.metrics.output_tokens = response_usage.completion_tokens
                assistant_message.metrics.completion_tokens = response_usage.completion_tokens
            if hasattr(response_usage, 'total_tokens') and response_usage.total_tokens is not None:
                assistant_message.metrics.total_tokens = response_usage.total_tokens
            else:
                assistant_message.metrics.total_tokens = (assistant_message.metrics.input_tokens + assistant_message.metrics.output_tokens)
        if isinstance(response_usage, dict) and 'additional_metrics' in response_usage:
            assistant_message.metrics.additional_metrics = response_usage['additional_metrics']
        if hasattr(response_usage, 'prompt_tokens_details'):
            if isinstance(response_usage.prompt_tokens_details, dict):
                assistant_message.metrics.prompt_tokens_details = response_usage.prompt_tokens_details
            elif hasattr(response_usage.prompt_tokens_details, 'model_dump'):
                assistant_message.metrics.prompt_tokens_details = response_usage.prompt_tokens_details.model_dump(exclude_none=True)
        if hasattr(response_usage, 'completion_tokens_details'):
            if isinstance(response_usage.completion_tokens_details, dict):
                assistant_message.metrics.completion_tokens_details = response_usage.completion_tokens_details
            elif hasattr(response_usage.completion_tokens_details, 'model_dump'):
                assistant_message.metrics.completion_tokens_details = (response_usage.completion_tokens_details.model_dump(exclude_none=True))

    def _log_messages(self, messages: List[Message]) -> None:
        for m in messages:
            m.log(metrics=False)

    def get_system_message_for_model(self) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.instructions

    def clear(self) -> None:
        self.response_format = None
        self._functions = None
        self._function_call_stack = None

    def __deepcopy__(self, memo: dict):
        from copy import deepcopy
        cls = self.__class__
        new_model = cls.__new__(cls)
        memo[id(self)] = new_model
        for k, v in self.__dict__.items():
            if k in {'response_format', 'tools', '_functions', '_function_call_stack'}:
                continue
            setattr(new_model, k, deepcopy(v, memo))
        new_model.clear()
        return new_model
