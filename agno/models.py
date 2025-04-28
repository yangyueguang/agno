import json
import shutil
import asyncio
import collections.abc
from uuid import uuid4
from enum import Enum
from copy import deepcopy
from time import time, perf_counter
from abc import ABC, abstractmethod
from types import AsyncGeneratorType, GeneratorType
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field, validate_call
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterator, List, Literal, Tuple, Sequence, Mapping, Optional, Union, Callable, get_type_hints, get_args, get_origin
import base64
import zlib
import requests
from pathlib import Path
from textwrap import dedent
from ollama import AsyncClient as AsyncOllamaClient, Client as OllamaClient
from ollama._types import ChatResponse, Message as OllamaMessage
from functools import update_wrapper, wraps, partial
from docstring_parser import parse
from collections import OrderedDict
from tempfile import gettempdir
from hashlib import md5
from inspect import isasyncgen, isasyncgenfunction, iscoroutinefunction, isgenerator, iscoroutine, getdoc, signature
from types import UnionType


class Dot(dict):
    def __init__(self, seq=None, **kwargs):
        if not isinstance(seq, dict):
            seq = {'value': seq}
        super(Dot, self).__init__(seq, **kwargs)

    def __getattr__(self, attr):
        res = self.get(attr)
        if isinstance(res, dict):
            return Dot(res)
        return res
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __call__(self, keypath: str, *args, **kwargs):
        temp = self
        for i in keypath.split('.'):
            temp = temp[int(i) if i.isnumeric() else i]
        return temp


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


class AgentRunException(Exception):
    def __init__(self, exc, user_message: str = None, agent_message: str = None, messages: Optional[List[dict]] = None, stop_execution: bool = False):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution


def get_json_schema_for_arg(t: Any) -> Optional[Dict[str, Any]]:
    type_args = get_args(t)
    type_origin = get_origin(t)
    if type_origin is not None:
        if type_origin in (list, tuple, set, frozenset):
            json_schema_for_items = get_json_schema_for_arg(type_args[0]) if type_args else {'type': 'string'}
            return {'type': 'array', 'items': json_schema_for_items}
        elif type_origin is dict:
            key_schema = get_json_schema_for_arg(type_args[0]) if type_args else {'type': 'string'}
            value_schema = get_json_schema_for_arg(type_args[1]) if len(type_args) > 1 else {'type': 'string'}
            return {'type': 'object', 'propertyNames': key_schema, 'additionalProperties': value_schema}
        elif type_origin in [Union, UnionType]:
            types = []
            for arg in type_args:
                if schema := get_json_schema_for_arg(arg):
                    types.append(schema)
            return {'anyOf': types} if types else None
    types = {
        'number': ['int', 'float', 'complex', 'Decimal'],
        'string': ['str', 'string'],
        'boolean': ['bool', 'boolean'],
        'null': ['NoneType', 'None'],
        'array': ['list', 'tuple', 'set', 'frozenset'],
        'object': ['dict', 'mapping']}
    json_schema = {'type': object}
    for k, v in types.items():
        if t.__name__ in v:
            json_schema['type'] = k
    if json_schema['type'] == 'object':
        json_schema['properties'] = {}
        json_schema['additionalProperties'] = False
    return json_schema


def get_json_schema(type_hints: Dict[str, Any], param_descriptions: Optional[Dict[str, str]] = None, strict: bool = False) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {'type': 'object', 'properties': {}}
    if strict:
        json_schema['additionalProperties'] = False
    for k, v in type_hints.items():
        if k == 'return':
            continue
        try:
            type_origin = get_origin(v)
            type_args = get_args(v)
            is_optional = type_origin is Union and len(type_args) == 2 and any(arg is type(None) for arg in type_args)
            if is_optional:
                v = next(arg for arg in type_args if arg is not type(None))
            if v:
                arg_json_schema = get_json_schema_for_arg(v)
            else:
                arg_json_schema = {}
            if arg_json_schema is not None:
                if is_optional:
                    if isinstance(arg_json_schema['type'], list):
                        arg_json_schema['type'].append('null')
                    else:
                        arg_json_schema['type'] = [arg_json_schema['type'], 'null']
                if param_descriptions and k in param_descriptions and param_descriptions[k]:
                    arg_json_schema['description'] = param_descriptions[k]
                json_schema['properties'][k] = arg_json_schema
            else:
                print(f'Could not parse argument {k} of type {v}')
        except Exception as e:
            print(f'Error processing argument {k}: {str(e)}')
            continue
    return json_schema


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=lambda: {'type': 'object', 'properties': {}, 'required': []}, description='JSON Schema object describing function parameters')
    strict: Optional[bool] = None
    entrypoint: Optional[Callable] = None
    skip_entrypoint_processing: bool = False
    sanitize_arguments: bool = True
    show_result: bool = False
    stop_after_tool_call: bool = False
    pre_hook: Optional[Callable] = None
    post_hook: Optional[Callable] = None
    cache_results: bool = False
    cache_dir: Optional[str] = None
    cache_ttl: int = 3600
    _agent: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={'name', 'description', 'parameters', 'strict'})

    @classmethod
    def get_entrypoint_docstring(cls, entrypoint: Callable) -> str:
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

    @classmethod
    def from_callable(cls, c: Callable, strict: bool = False) -> 'Function':
        function_name = c.__name__
        parameters = {'type': 'object', 'properties': {}, 'required': []}
        try:
            sig = signature(c)
            type_hints = get_type_hints(c)
            if 'agent' in sig.parameters:
                del type_hints['agent']
            param_type_hints = {name: type_hints.get(name) for name in sig.parameters if name != 'return' and name != 'agent'}
            param_descriptions = {}
            if docstring := getdoc(c):
                parsed_doc = parse(docstring)
                param_docs = parsed_doc.params
                if param_docs is not None:
                    for param in param_docs:
                        param_name = param.arg_name
                        param_type = param.type_name
                        param_descriptions[param_name] = f'({param_type}) {param.description}'
            parameters = get_json_schema(type_hints=param_type_hints, param_descriptions=param_descriptions, strict=strict)
            if strict:
                parameters['required'] = [name for name in parameters['properties'] if name != 'agent']
            else:
                parameters['required'] = [name for name, param in sig.parameters.items() if param.default == param.empty and name != 'self' and name != 'agent']
        except Exception as e:
            print(f'Could not parse args for {function_name}: {e}', exc_info=True)
        if isasyncgenfunction(c):
            entrypoint = c
        else:
            entrypoint = validate_call(c, config=dict(arbitrary_types_allowed=True))
        return cls(name=function_name, description=cls.get_entrypoint_docstring(entrypoint=c), parameters=parameters, entrypoint=entrypoint)

    def process_entrypoint(self, strict: bool = False):
        if self.skip_entrypoint_processing:
            return
        if self.entrypoint is None:
            return
        parameters = {'type': 'object', 'properties': {}, 'required': []}
        params_set_by_user = False
        if self.parameters != parameters:
            params_set_by_user = True
        try:
            sig = signature(self.entrypoint)
            type_hints = get_type_hints(self.entrypoint)
            if 'agent' in sig.parameters:
                del type_hints['agent']
            param_type_hints = {name: type_hints.get(name) for name in sig.parameters if name != 'return' and name != 'agent'}
            param_descriptions = {}
            if docstring := getdoc(self.entrypoint):
                parsed_doc = parse(docstring)
                param_docs = parsed_doc.params
                if param_docs is not None:
                    for param in param_docs:
                        param_name = param.arg_name
                        param_type = param.type_name
                        param_descriptions[param_name] = f'({param_type}) {param.description}'
            parameters = get_json_schema(type_hints=param_type_hints, param_descriptions=param_descriptions, strict=strict)
            if strict:
                parameters['required'] = [name for name in parameters['properties'] if name != 'agent']
            else:
                parameters['required'] = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == param.empty and name != 'self' and name != 'agent'
                ]
            if params_set_by_user:
                self.parameters['additionalProperties'] = False
                if strict:
                    self.parameters['required'] = [name for name in self.parameters['properties'] if name != 'agent']
                else:
                    self.parameters['required'] = [name for name, param in sig.parameters.items() if param.default == param.empty and name != 'self' and name != 'agent']
        except Exception as e:
            print(f'Could not parse args for {self.name}: {e}', exc_info=True)
        self.description = self.description or self.get_entrypoint_docstring(self.entrypoint)
        if not params_set_by_user:
            self.parameters = parameters
        try:
            if not isasyncgenfunction(self.entrypoint):
                self.entrypoint = validate_call(self.entrypoint, config=dict(arbitrary_types_allowed=True))
        except Exception as e:
            print(f'Failed to add validate decorator to entrypoint: {e}')

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        if self.entrypoint is None:
            return None
        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get('return', None)
        returns = None
        if return_type is not None:
            name = str(return_type)
            if 'list' in name or 'dict' in name:
                returns = name
            else:
                returns = return_type.__name__
        function_info = {'name': self.name, 'description': self.description, 'arguments': self.parameters.get('properties', {}), 'returns': returns}
        return function_info

    def get_definition_for_prompt(self) -> Optional[str]:
        function_info = self.get_definition_for_prompt_dict()
        if function_info is not None:
            return json.dumps(function_info, indent=2)
        return None

    def _get_cache_key(self, entrypoint_args: Dict[str, Any], call_args: Optional[Dict[str, Any]] = None) -> str:
        copy_entrypoint_args = entrypoint_args.copy()
        if 'agent' in copy_entrypoint_args:
            del copy_entrypoint_args['agent']
        args_str = str(copy_entrypoint_args)
        kwargs_str = str(sorted((call_args or {}).items()))
        key_str = f'{self.name}:{args_str}:{kwargs_str}'
        return md5(key_str.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        base_cache_dir = self.cache_dir or Path(gettempdir()) / 'agno_cache'
        func_cache_dir = Path(base_cache_dir) / 'functions' / self.name
        func_cache_dir.mkdir(parents=True, exist_ok=True)
        return str(func_cache_dir / f'{cache_key}.json')

    def _get_cached_result(self, cache_file: str) -> Optional[Any]:
        cache_path = Path(cache_file)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open('r') as f:
                cache_data = json.load(f)
            timestamp = cache_data.get('timestamp', 0)
            result = cache_data.get('result')
            if time() - timestamp <= self.cache_ttl:
                return result
            cache_path.unlink()
        except Exception as e:
            print(f'Error reading cache: {e}')
        return None

    def _save_to_cache(self, cache_file: str, result: Any):
        try:
            with open(cache_file, 'w') as f:
                json.dump({'timestamp': time(), 'result': result}, f)
        except Exception as e:
            print(f'Error writing cache: {e}')


class FunctionCall(BaseModel):
    function: Function
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    call_id: Optional[str] = None
    error: Optional[str] = None

    def get_call_str(self) -> str:
        term_width = shutil.get_terminal_size().columns or 80
        max_arg_len = max(20, (term_width - len(self.function.name) - 4) // 2)
        if self.arguments is None:
            return f'{self.function.name}()'
        trimmed_arguments = {}
        for k, v in self.arguments.items():
            if isinstance(v, str) and len(str(v)) > max_arg_len:
                trimmed_arguments[k] = '...'
            else:
                trimmed_arguments[k] = v
        call_str = f'{self.function.name}({", ".join([f"{k}={v}" for k, v in trimmed_arguments.items()])})'
        if len(call_str) > term_width:
            return f'{self.function.name}(...)'
        return call_str

    def _handle_pre_hook(self):
        if self.function.pre_hook is not None:
            try:
                pre_hook_args = {}
                if 'agent' in signature(self.function.pre_hook).parameters:
                    pre_hook_args['agent'] = self.function._agent
                if 'fc' in signature(self.function.pre_hook).parameters:
                    pre_hook_args['fc'] = self
                self.function.pre_hook(**pre_hook_args)
            except AgentRunException as e:
                print(f'{e.__class__.__name__}: {e}')
                self.error = str(e)
                raise
            except Exception as e:
                print(f'Error in pre-hook callback: {e}')
                print(e)

    def _handle_post_hook(self):
        if self.function.post_hook is not None:
            try:
                post_hook_args = {}
                if 'agent' in signature(self.function.post_hook).parameters:
                    post_hook_args['agent'] = self.function._agent
                if 'fc' in signature(self.function.post_hook).parameters:
                    post_hook_args['fc'] = self
                self.function.post_hook(**post_hook_args)
            except AgentRunException as e:
                print(f'{e.__class__.__name__}: {e}')
                self.error = str(e)
                raise
            except Exception as e:
                print(f'Error in post-hook callback: {e}')
                print(e)

    def _build_entrypoint_args(self) -> Dict[str, Any]:
        entrypoint_args = {}
        if 'agent' in signature(self.function.entrypoint).parameters:
            entrypoint_args['agent'] = self.function._agent
        if 'fc' in signature(self.function.entrypoint).parameters:
            entrypoint_args['fc'] = self
        return entrypoint_args

    def execute(self) -> bool:
        if self.function.entrypoint is None:
            return False
        print(f'Running: {self.get_call_str()}')
        function_call_success = False
        self._handle_pre_hook()
        entrypoint_args = self._build_entrypoint_args()
        if self.function.cache_results and not isgenerator(self.function.entrypoint):
            cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
            cache_file = self.function._get_cache_file_path(cache_key)
            cached_result = self.function._get_cached_result(cache_file)
            if cached_result is not None:
                print(f'Cache hit for: {self.get_call_str()}')
                self.result = cached_result
                function_call_success = True
                return function_call_success
        try:
            if self.arguments == {} or self.arguments is None:
                result = self.function.entrypoint(**entrypoint_args)
            else:
                result = self.function.entrypoint(**entrypoint_args, **self.arguments)
            if isgenerator(result):
                self.result = result
            else:
                self.result = result
                if self.function.cache_results:
                    cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
                    cache_file = self.function._get_cache_file_path(cache_key)
                    self.function._save_to_cache(cache_file, self.result)
            function_call_success = True
        except AgentRunException as e:
            print(f'{e.__class__.__name__}: {e}')
            self.error = str(e)
            raise
        except Exception as e:
            print(f'Could not run function {self.get_call_str()}')
            print(e)
            self.error = str(e)
            return function_call_success
        self._handle_post_hook()
        return function_call_success

    async def _handle_pre_hook_async(self):
        if self.function.pre_hook is not None:
            try:
                pre_hook_args = {}
                if 'agent' in signature(self.function.pre_hook).parameters:
                    pre_hook_args['agent'] = self.function._agent
                if 'fc' in signature(self.function.pre_hook).parameters:
                    pre_hook_args['fc'] = self
                await self.function.pre_hook(**pre_hook_args)
            except AgentRunException as e:
                print(f'{e.__class__.__name__}: {e}')
                self.error = str(e)
                raise
            except Exception as e:
                print(f'Error in pre-hook callback: {e}')
                print(e)

    async def _handle_post_hook_async(self):
        if self.function.post_hook is not None:
            try:
                post_hook_args = {}
                if 'agent' in signature(self.function.post_hook).parameters:
                    post_hook_args['agent'] = self.function._agent
                if 'fc' in signature(self.function.post_hook).parameters:
                    post_hook_args['fc'] = self
                await self.function.post_hook(**post_hook_args)
            except AgentRunException as e:
                print(f'{e.__class__.__name__}: {e}')
                self.error = str(e)
                raise
            except Exception as e:
                print(f'Error in post-hook callback: {e}')
                print(e)

    async def aexecute(self) -> bool:
        if self.function.entrypoint is None:
            return False
        print(f'Running: {self.get_call_str()}')
        function_call_success = False
        if iscoroutinefunction(self.function.pre_hook):
            await self._handle_pre_hook_async()
        else:
            self._handle_pre_hook()
        entrypoint_args = self._build_entrypoint_args()
        if self.function.cache_results and not (isasyncgen(self.function.entrypoint) or isgenerator(self.function.entrypoint)):
            cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
            cache_file = self.function._get_cache_file_path(cache_key)
            cached_result = self.function._get_cached_result(cache_file)
            if cached_result is not None:
                print(f'Cache hit for: {self.get_call_str()}')
                self.result = cached_result
                function_call_success = True
                return function_call_success
        try:
            if self.arguments == {} or self.arguments is None:
                result = self.function.entrypoint(**entrypoint_args)
                if isasyncgen(self.function.entrypoint) or isasyncgenfunction(self.function.entrypoint):
                    self.result = result
                else:
                    self.result = await result
            else:
                result = self.function.entrypoint(**entrypoint_args, **self.arguments)
                if isasyncgen(self.function.entrypoint) or isasyncgenfunction(self.function.entrypoint):
                    self.result = result
                else:
                    self.result = await result
            if self.function.cache_results and not (isgenerator(self.result) or isasyncgen(self.result)):
                cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
                cache_file = self.function._get_cache_file_path(cache_key)
                self.function._save_to_cache(cache_file, self.result)
            function_call_success = True
        except AgentRunException as e:
            print(f'{e.__class__.__name__}: {e}')
            self.error = str(e)
            raise
        except Exception as e:
            print(f'Could not run function {self.get_call_str()}')
            print(e)
            self.error = str(e)
            return function_call_success
        if iscoroutinefunction(self.function.post_hook):
            await self._handle_post_hook_async()
        else:
            self._handle_post_hook()
        return function_call_success


def tool(name: Optional[str] = None, description: Optional[str] = None, strict: Optional[bool] = None, sanitize_arguments: Optional[bool] = None,
         show_result: Optional[bool] = None, stop_after_tool_call: Optional[bool] = None, pre_hook: Optional[Callable] = None, post_hook: Optional[Callable] = None,
         cache_results: bool = False, cache_dir: Optional[str] = None, cache_ttl: int = 3600) -> Function:
    """Decorator将函数转换为代理可以使用的函数。
    Args：
    name:可选[str]-覆盖函数名
    description:可选[str]-函数描述的覆盖
    strict:可选[bool]-用于严格参数检查的标志
    sanctize_arguments：可选[bool]-如果为True，则在传递给函数之前对参数进行净化
    show_result：可选[bool]-如果为True，则显示函数调用后的结果
    stop_after_tool_call：可选[bool]-如果为True，代理将在函数调用后停止。
    pre_hook：可选[Calable]-在函数执行之前运行的钩子。
    post_hook：可选[Calable]-在函数执行后运行的钩子。
    cache_results:bool-如果为True，则启用函数结果的缓存
    cache_dir：可选[str]-存储缓存文件的目录
    cache_ttl:int-缓存结果的生存时间（秒）
    return:
    Union[函数，可调用[[F]，函数]]：装饰函数或装饰器"""
    def decorator(func: Callable) -> Function:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Error in tool {func.__name__!r}: {e!r}', exc_info=True)
                raise
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f'Error in async tool {func.__name__!r}: {e!r}', exc_info=True)
                raise
        @wraps(func)
        async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Error in async generator tool {func.__name__!r}: {e!r}', exc_info=True)
                raise
        if isasyncgenfunction(func):
            wrapper = async_gen_wrapper
        elif iscoroutinefunction(func) or iscoroutine(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        update_wrapper(wrapper, func)
        return Function(**{'name': name or func.__name__, 'description': description or getdoc(func), 'entrypoint': wrapper, 'cache_results': cache_results, 'cache_dir': cache_dir, 'cache_ttl': cache_ttl})
    if callable(name):
        return decorator(name)
    return decorator


class Toolkit:
    def __init__(self, name: str = 'toolkit', cache_results: bool = False, cache_ttl: int = 3600, cache_dir: Optional[str] = None):
        self.name: str = name
        self.functions: Dict[str, Function] = OrderedDict()
        self.cache_results: bool = cache_results
        self.cache_ttl: int = cache_ttl
        self.cache_dir: Optional[str] = cache_dir

    def register(self, function: Callable[..., Any], sanitize_arguments: bool = True):
        try:
            f = Function(name=function.__name__, entrypoint=function, sanitize_arguments=sanitize_arguments, cache_results=self.cache_results, cache_dir=self.cache_dir, cache_ttl=self.cache_ttl)
            self.functions[f.name] = f
            print(f'Function: {f.name} registered with {self.name}')
        except Exception as e:
            print(f'Failed to create Function for: {function.__name__}')
            raise e

    def instructions(self) -> str:
        return ''

    def __repr__(self):
        return f'<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>'

    def __str__(self):
        return self.__repr__()


class Media(BaseModel):
    id: str
    original_prompt: Optional[str] = None
    revised_prompt: Optional[str] = None


class VideoArtifact(Media):
    url: str
    eta: Optional[str] = None
    length: Optional[str] = None


class ImageArtifact(Media):
    url: Optional[str] = None
    content: Optional[bytes] = None
    mime_type: Optional[str] = None
    alt_text: Optional[str] = None


class AudioArtifact(Media):
    url: Optional[str] = None
    base64_audio: Optional[str] = None
    length: Optional[str] = None
    mime_type: Optional[str] = None


class Video(BaseModel):
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    format: Optional[str] = 'mp4'

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8') if self.content else None, 'filepath': self.filepath, 'format': self.format}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: VideoArtifact) -> 'Video':
        return cls(url=artifact.url)


class Audio(BaseModel):
    content: Optional[Any] = None
    filepath: Optional[Union[Path, str]] = None
    url: Optional[str] = None
    format: Optional[str] = None

    @property
    def audio_url_content(self) -> Optional[bytes]:
        if self.url:
            return requests.get(self.url).content
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8')
            if self.content
            else None, 'filepath': self.filepath, 'format': self.format}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: AudioArtifact) -> 'Audio':
        return cls(url=artifact.url, content=artifact.base64_audio, format=artifact.mime_type)


class AudioResponse(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    expires_at: Optional[int] = None
    transcript: Optional[str] = None
    mime_type: Optional[str] = None
    sample_rate: Optional[int] = 24000
    channels: Optional[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'id': self.id, 'content': base64.b64encode(self.content).decode('utf-8')
            if isinstance(self.content, bytes)
            else self.content, 'expires_at': self.expires_at, 'transcript': self.transcript, 'mime_type': self.mime_type, 'sample_rate': self.sample_rate, 'channels': self.channels}
        return {k: v for k, v in response_dict.items() if v is not None}


class Image(BaseModel):
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    format: Optional[str] = 'jpeg'
    detail: Optional[str] = None
    id: Optional[str] = None

    @property
    def image_url_content(self) -> Optional[bytes]:
        if self.url:
            return requests.get(self.url).content
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8')
            if self.content
            else None, 'filepath': self.filepath, 'url': self.url, 'detail': self.detail}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: ImageArtifact) -> 'Image':
        return cls(url=artifact.url)


class File(BaseModel):
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    mime_type: Optional[str] = None

    @classmethod
    def valid_mime_types(cls) -> List[str]:
        return ['application/pdf', 'application/x-javascript', 'text/javascript', 'application/x-python', 'text/x-python', 'text/plain', 'text/html', 'text/css', 'text/md', 'text/csv', 'text/xml', 'text/rtf']

    @property
    def file_url_content(self) -> Optional[Tuple[bytes, str]]:
        if self.url:
            response = requests.get(self.url)
            content = response.content
            mime_type = response.headers.get('Content-Type', '').split(';')[0]
            return content, mime_type
        else:
            return None


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
            mdict = self.metrics.__dict__
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
        cls = self.__class__
        new_model = cls.__new__(cls)
        memo[id(self)] = new_model
        for k, v in self.__dict__.items():
            if k in {'response_format', 'tools', '_functions', '_function_call_stack'}:
                continue
            setattr(new_model, k, deepcopy(v, memo))
        new_model.clear()
        return new_model


@dataclass
class Ollama(Model):
    id: str = 'llama3.1:8b'
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
    @property
    def request_kwargs(self) -> Dict[str, Any]:
        base_params: Dict[str, Any] = {'format': self.format, 'options': self.options, 'keep_alive': self.keep_alive, 'request_params': self.request_params}
        request_params: Dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def extract_tool_call_from_string(self, text: str, start_tag: str = '<tool_call>', end_tag: str = '</tool_call>'):
        start_index = text.find(start_tag) + len(start_tag)
        end_index = text.find(end_tag)
        return text[start_index:end_index].strip()

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
                        tool_call_content = self.extract_tool_call_from_string(tool_call_response)
                        try:
                            tool_call_dict = json.loads(tool_call_content)
                        except json.JSONDecodeError:
                            raise ValueError(f'Could not parse tool call from: {tool_call_content}')
                        tool_call_name = tool_call_dict.get('name')
                        tool_call_args = tool_call_dict.get('arguments')
                        function_def = {'name': tool_call_name, 'arguments': json.dumps(tool_call_args) if tool_call_args is not None else None}
                        model_response.tool_calls.append({'type': 'function', 'function': function_def})
        if response.get('done'):
            model_response.response_usage = Dot(input_tokens=response.get('prompt_eval_count', 0), output_tokens=response.get('eval_count', 0), total_duration=response.get('total_duration', 0), load_duration=response.get('load_duration', 0), prompt_eval_duration=response.get('prompt_eval_duration', 0), eval_duration=response.get('eval_duration', 0))
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
        start_tag: str = '<tool_call>'
        end_tag: str = '</tool_call>'
        text = assistant_message.get_content_string()
        while start_tag in text and end_tag in text:
            start_index = text.find(start_tag)
            end_index = text.find(end_tag) + len(end_tag)
            text = text[:start_index] + text[end_index:]
        model_response.content = str(text)
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

                        tool_calls = []
                        response_content = tool_call_data.tool_call_content
                        if '<tool_call>' in response_content and '</tool_call>' in response_content:
                            tool_call_responses = response_content.split('</tool_call>')
                            for tool_call_response in tool_call_responses:
                                if tool_call_response != tool_call_responses[-1]:
                                    tool_call_response += '</tool_call>'
                                if '<tool_call>' in tool_call_response and '</tool_call>' in tool_call_response:
                                    tool_call_content = self.extract_tool_call_from_string(tool_call_response)
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

                        model_response.tool_calls = tool_calls
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
