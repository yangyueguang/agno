import json
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, AsyncIterator
from pydantic import BaseModel
from agno.tools import FunctionCall
from agno.models import Model, Message, ModelResponse, MessageMetrics, Timer
from ollama import AsyncClient as AsyncOllamaClient, Client as OllamaClient
from ollama._types import ChatResponse, Message as OllamaMessage


def extract_tool_call_from_string(text: str, start_tag: str = '<tool_call>', end_tag: str = '</tool_call>'):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)
    return text[start_index:end_index].strip()


def remove_tool_calls_from_string(text: str, start_tag: str = '<tool_call>', end_tag: str = '</tool_call>'):
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text


@dataclass
class OllamaResponseUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_duration: int = 0
    eval_duration: int = 0


@dataclass
class Ollama(Model):
    id: str = 'llama3.1'
    name: str = 'Ollama'
    provider: str = 'Ollama'
    supports_native_structured_outputs: bool = True
    format: Optional[Any] = None
    options: Optional[Any] = None
    keep_alive: Optional[Union[float, str]] = None
    request_params: Optional[Dict[str, Any]] = None
    host: Optional[str] = None
    timeout: Optional[Any] = None
    client_params: Optional[Dict[str, Any]] = None
    client: Optional[OllamaClient] = None
    async_client: Optional[AsyncOllamaClient] = None

    def _get_client_params(self) -> Dict[str, Any]:
        base_params = {'host': self.host, 'timeout': self.timeout}
        client_params = {k: v for k, v in base_params.items() if v is not None}
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def get_client(self) -> OllamaClient:
        if self.client is not None:
            return self.client
        self.client = OllamaClient(**self._get_client_params())
        return self.client

    def get_async_client(self) -> AsyncOllamaClient:
        if self.async_client is not None:
            return self.async_client
        return AsyncOllamaClient(**self._get_client_params())

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        base_params = {'format': self.format, 'options': self.options, 'keep_alive': self.keep_alive, 'request_params': self.request_params}
        request_params = {k: v for k, v in base_params.items() if v is not None}
        if self._tools is not None and len(self._tools) > 0:
            request_params['tools'] = self._tools
            for tool in request_params['tools']:
                if 'parameters' in tool['function'] and 'properties' in tool['function']['parameters']:
                    for _, obj in tool['function']['parameters'].get('properties', {}).items():
                        if 'type' in obj and isinstance(obj['type'], list) and len(obj['type']) > 1:
                            obj['type'] = obj['type'][0]
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def to_dict(self) -> Dict[str, Any]:
        model_dict = super().to_dict()
        model_dict.update({'format': self.format, 'options': self.options, 'keep_alive': self.keep_alive, 'request_params': self.request_params})
        if self._tools is not None:
            model_dict['tools'] = self._tools
        cleaned_dict = {k: v for k, v in model_dict.items() if v is not None}
        return cleaned_dict

    def _format_message(self, message: Message) -> Dict[str, Any]:
        _message: Dict[str, Any] = {'role': message.role, 'content': message.content}
        if message.role == 'user':
            if message.images is not None:
                message_images = []
                for image in message.images:
                    if image.url is not None:
                        message_images.append(image.image_url_content)
                    if image.filepath is not None:
                        message_images.append(image.filepath)
                    if image.content is not None and isinstance(image.content, bytes):
                        message_images.append(image.content)
                if message_images:
                    _message['images'] = message_images
        return _message

    def _prepare_request_kwargs_for_invoke(self) -> Dict[str, Any]:
        request_kwargs = self.request_kwargs
        if self.response_format is not None and self.structured_outputs:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                print('Using structured outputs')
                format_schema = self.response_format.model_json_schema()
                if 'format' not in request_kwargs:
                    request_kwargs['format'] = format_schema
        return request_kwargs

    def invoke(self, messages: List[Message]) -> Mapping[str, Any]:
        request_kwargs = self._prepare_request_kwargs_for_invoke()
        return self.get_client().chat(model=self.id.strip(), messages=[self._format_message(m) for m in messages], **request_kwargs)

    async def ainvoke(self, messages: List[Message]) -> Mapping[str, Any]:
        request_kwargs = self._prepare_request_kwargs_for_invoke()
        return await self.get_async_client().chat(model=self.id.strip(), messages=[self._format_message(m) for m in messages], **request_kwargs)

    def invoke_stream(self, messages: List[Message]) -> Iterator[Mapping[str, Any]]:
        yield from self.get_client().chat(model=self.id, messages=[self._format_message(m) for m in messages], stream=True, **self.request_kwargs)

    async def ainvoke_stream(self, messages: List[Message]) -> Any:
        async_stream = await self.get_async_client().chat(model=self.id.strip(), messages=[self._format_message(m) for m in messages], stream=True, **self.request_kwargs)
        async for chunk in async_stream:
            yield chunk

    def parse_provider_response(self, response: ChatResponse) -> ModelResponse:
        model_response = ModelResponse()
        response_message: OllamaMessage = response.get('message')
        try:
            if self.response_format is not None and self.structured_outputs and issubclass(self.response_format, BaseModel):
                parsed_object = response_message.content
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            print(f'Error retrieving structured outputs: {e}')
        if response_message.get('role') is not None:
            model_response.role = response_message.get('role')
        if response_message.get('content') is not None:
            model_response.content = response_message.get('content')
        if response_message.get('tool_calls') is not None:
            if model_response.tool_calls is None:
                model_response.tool_calls = []
            for block in response_message.get('tool_calls'):
                tool_call = block.get('function')
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('arguments')
                function_def = {'name': tool_name, 'arguments': (json.dumps(tool_args) if tool_args is not None else None)}
                model_response.tool_calls.append({'type': 'function', 'function': function_def})
        if response.get('done'):
            model_response.response_usage = {'input_tokens': response.get('prompt_eval_count', 0), 'output_tokens': response.get('eval_count', 0), 'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0), 'additional_metrics': {'total_duration': response.get('total_duration', 0), 'load_duration': response.get('load_duration', 0), 'prompt_eval_duration': response.get('prompt_eval_duration', 0), 'eval_duration': response.get('eval_duration', 0)}}
        return model_response

    def parse_provider_response_delta(self, response_delta: ChatResponse) -> ModelResponse:
        model_response = ModelResponse()
        response_message = response_delta.get('message')
        if response_message is not None:
            content_delta = response_message.get('content')
            if content_delta is not None and content_delta != '':
                model_response.content = content_delta
            tool_calls = response_message.get('tool_calls')
            if tool_calls is not None:
                for tool_call in tool_calls:
                    tc = tool_call.get('function')
                    tool_name = tc.get('name')
                    tool_args = tc.get('arguments')
                    function_def = {'name': tool_name, 'arguments': json.dumps(tool_args) if tool_args is not None else None}
                    model_response.tool_calls.append({'type': 'function', 'function': function_def})
        if response_delta.get('done'):
            model_response.response_usage = {'input_tokens': response_delta.get('prompt_eval_count', 0), 'output_tokens': response_delta.get('eval_count', 0), 'total_tokens': response_delta.get('prompt_eval_count', 0) + response_delta.get('eval_count', 0), 'additional_metrics': {'total_duration': response_delta.get('total_duration', 0), 'load_duration': response_delta.get('load_duration', 0), 'prompt_eval_duration': response_delta.get('prompt_eval_duration', 0), 'eval_duration': response_delta.get('eval_duration', 0)}}
        return model_response


class ToolCall:
    def __init__(self, tool_calls: List[Dict[str, Any]] = None, response_usage: Optional[Mapping[str, Any]] = None, response_is_tool_call=False, is_closing_tool_call_tag=False, tool_calls_counter=0, tool_call_content=''):
        self.tool_calls = tool_calls or []
        self.response_usage = response_usage
        self.response_is_tool_call = response_is_tool_call
        self.is_closing_tool_call_tag = is_closing_tool_call_tag
        self.tool_calls_counter = tool_calls_counter
        self.tool_call_content = tool_call_content


class OllamaTools(Ollama):
    def __init__(self, id='llama3.2', name='OllamaTools', provider='Ollama'):
        self.id = id
        self.name = name
        self.provider = provider

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        base_params: Dict[str, Any] = {'format': self.format, 'options': self.options, 'keep_alive': self.keep_alive, 'request_params': self.request_params}
        request_params: Dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def parse_provider_response(self, response: ChatResponse) -> ModelResponse:
        model_response = ModelResponse()
        response_message = response.get('message')
        if response_message.get('role') is not None:
            model_response.role = response_message.get('role')
        content = response_message.get('content')
        if content is not None:
            model_response.content = content
            if '<tool_call>' in content and '</tool_call>' in content:
                if model_response.tool_calls is None:
                    model_response.tool_calls = []
                tool_call_responses = content.split('</tool_call>')
                for tool_call_response in tool_call_responses:
                    if tool_call_response != tool_call_responses[-1]:
                        tool_call_response += '</tool_call>'
                    if '<tool_call>' in tool_call_response and '</tool_call>' in tool_call_response:
                        tool_call_content = extract_tool_call_from_string(tool_call_response)
                        try:
                            tool_call_dict = json.loads(tool_call_content)
                        except json.JSONDecodeError:
                            raise ValueError(f'Could not parse tool call from: {tool_call_content}')
                        tool_call_name = tool_call_dict.get('name')
                        tool_call_args = tool_call_dict.get('arguments')
                        function_def = {'name': tool_call_name, 'arguments': json.dumps(tool_call_args) if tool_call_args is not None else None}
                        model_response.tool_calls.append({'type': 'function', 'function': function_def})
        if response.get('done'):
            model_response.response_usage = OllamaResponseUsage(input_tokens=response.get('prompt_eval_count', 0), output_tokens=response.get('eval_count', 0), total_duration=response.get('total_duration', 0), load_duration=response.get('load_duration', 0), prompt_eval_duration=response.get('prompt_eval_duration', 0), eval_duration=response.get('eval_duration', 0))
            if model_response.response_usage.input_tokens or model_response.response_usage.output_tokens:
                model_response.response_usage.total_tokens = (model_response.response_usage.input_tokens + model_response.response_usage.output_tokens)
        return model_response

    def _create_function_call_result(self, fc: FunctionCall, success: bool, output: Optional[Union[List[Any], str]], timer: Timer) -> Message:
        content = ('<tool_response>\n'
            + json.dumps({'name': fc.function.name, 'content': output if success else fc.error})
            + '\n</tool_response>')
        return Message(role=self.tool_message_role, content=content, tool_call_id=fc.call_id, tool_name=fc.function.name, tool_args=fc.arguments, tool_call_error=not success, stop_after_tool_call=fc.function.stop_after_tool_call, metrics=MessageMetrics(time=timer.elapsed))

    def format_function_call_results(self, function_call_results: List[Message], messages: List[Message]) -> None:
        if len(function_call_results) > 0:
            for _fc_message in function_call_results:
                _fc_message.content = ('<tool_response>\n'
                    + json.dumps({'name': _fc_message.tool_name, 'content': _fc_message.content})
                    + '\n</tool_response>')
                messages.append(_fc_message)

    def _prepare_function_calls(self, assistant_message: Message, messages: List[Message], model_response: ModelResponse) -> List[FunctionCall]:
        if model_response.content is None:
            model_response.content = ''
        if model_response.tool_calls is None:
            model_response.tool_calls = []
        model_response.content = str(remove_tool_calls_from_string(assistant_message.get_content_string()))
        model_response.content += '\n\n'
        function_calls_to_run = self.get_function_calls_to_run(assistant_message, messages)
        return function_calls_to_run

    def process_response_stream(self, messages: List[Message], assistant_message: Message, stream_data) -> Iterator[ModelResponse]:
        tool_call_data = ToolCall()
        for response_delta in self.invoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta, tool_call_data)
            if model_response_delta:
                yield from self._populate_stream_data_and_assistant_message(stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta)

    async def aprocess_response_stream(self, messages: List[Message], assistant_message: Message, stream_data) -> AsyncIterator[ModelResponse]:
        tool_call_data = ToolCall()
        async for response_delta in self.ainvoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta, tool_call_data)
            if model_response_delta:
                for model_response in self._populate_stream_data_and_assistant_message(stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta):
                    yield model_response

    def parse_provider_response_delta(self, response_delta, tool_call_data: ToolCall) -> ModelResponse:
        model_response = ModelResponse()
        response_message = response_delta.get('message')
        if response_message is not None:
            content_delta = response_message.get('content', '')
            if content_delta is not None and content_delta != '':
                tool_call_data.tool_call_content += content_delta
            if not tool_call_data.response_is_tool_call and '<tool' in content_delta:
                tool_call_data.response_is_tool_call = True
            if tool_call_data.response_is_tool_call:
                if '<tool' in content_delta:
                    tool_call_data.tool_calls_counter += 1
                if tool_call_data.tool_call_content.strip().endswith('</tool_call>'):
                    tool_call_data.tool_calls_counter -= 1
                if tool_call_data.tool_calls_counter == 0 and content_delta.strip().endswith('>'):
                    tool_call_data.response_is_tool_call = False
                    tool_call_data.is_closing_tool_call_tag = True
                    try:
                        model_response.tool_calls = _parse_tool_calls_from_content(tool_call_data.tool_call_content)
                        tool_call_data = ToolCall()
                    except Exception as e:
                        print(e)
                        pass
            if not tool_call_data.response_is_tool_call and content_delta is not None:
                if tool_call_data.is_closing_tool_call_tag and content_delta.strip().endswith('>'):
                    tool_call_data.is_closing_tool_call_tag = False
                model_response.content = content_delta
        if response_delta.get('done'):
            model_response.response_usage = {'input_tokens': response_delta.get('prompt_eval_count', 0), 'output_tokens': response_delta.get('eval_count', 0), 'total_tokens': response_delta.get('prompt_eval_count', 0) + response_delta.get('eval_count', 0), 'additional_metrics': {'total_duration': response_delta.get('total_duration', 0), 'load_duration': response_delta.get('load_duration', 0), 'prompt_eval_duration': response_delta.get('prompt_eval_duration', 0), 'eval_duration': response_delta.get('eval_duration', 0)}}
        return model_response

    def get_instructions_to_generate_tool_calls(self) -> List[str]:
        if self._functions is not None:
            return [
                '在第一回合，你没有<tool_results>，所以你不应该编造结果。',
                '要回复用户消息，一次只能使用一个工具。',
                '使用工具时，只能通过工具调用进行响应。没有别的。不要添加任何额外的注释、解释或空格',
                '在任务完成或达到最大迭代次数10之前，不要停止调用函数。']
        return []

    def get_tool_call_prompt(self) -> Optional[str]:
        if self._functions is not None and len(self._functions) > 0:
            tool_call_prompt = dedent('''
            您是一个使用语言模型调用的函数。
            您将在<tools></tools>XML标签中获得函数签名。
            您可以使用代理框架进行推理和规划，以帮助用户查询。
            请调用一个函数，并等待在下一次迭代中向您提供函数结果。
            不要对要插入函数的值做出假设。
            调用函数时，不要添加任何额外的注释、解释或空格。
            调用函数后，结果将在<tool_response></tool_response>XML标签中提供给您。
            如果由于函数尚未执行而不存在<tool_response>XML标记，则不要对工具结果做出假设。
            一旦你得到结果，就分析它们，并在需要时调用另一个函数。
            您的最终响应应该直接回答用户查询，并对函数调用的结果进行分析或总结。
            ''')
            tool_call_prompt += '\nHere are the available tools:'
            tool_call_prompt += '\n<tools>\n'
            tool_definitions: List[str] = []
            for _f_name, _function in self._functions.items():
                _function_def = _function.get_definition_for_prompt()
                if _function_def:
                    tool_definitions.append(_function_def)
            tool_call_prompt += '\n'.join(tool_definitions)
            tool_call_prompt += '\n</tools>\n\n'
            tool_call_prompt += dedent('''\
            对您将进行的每个工具调用使用以下pydantic模型json模式: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
            对于每个函数调用，返回一个json对象，其中包含函数名和参数 <tool_call></tool_call> XML标签如下:
            <tool_call>
            {'arguments': <args-dict>, 'name': <function-name>}
            </tool_call>\n
            ''')
            return tool_call_prompt
        return None

    def get_system_message_for_model(self) -> Optional[str]:
        return self.get_tool_call_prompt()

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.get_instructions_to_generate_tool_calls()


def _parse_tool_calls_from_content(response_content: str) -> List[Dict[str, Any]]:
    tool_calls = []
    if '<tool_call>' in response_content and '</tool_call>' in response_content:
        tool_call_responses = response_content.split('</tool_call>')
        for tool_call_response in tool_call_responses:
            if tool_call_response != tool_call_responses[-1]:
                tool_call_response += '</tool_call>'
            if '<tool_call>' in tool_call_response and '</tool_call>' in tool_call_response:
                tool_call_content = extract_tool_call_from_string(tool_call_response)
                try:
                    tool_call_dict = json.loads(tool_call_content)
                except json.JSONDecodeError:
                    raise ValueError(f'Could not parse tool call from: {tool_call_content}')
                tool_call_name = tool_call_dict.get('name')
                tool_call_args = tool_call_dict.get('arguments')
                function_def = {'name': tool_call_name}
                if tool_call_args is not None:
                    function_def['arguments'] = json.dumps(tool_call_args)
                tool_calls.append({'type': 'function', 'function': function_def})
    return tool_calls
