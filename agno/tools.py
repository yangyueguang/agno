from functools import update_wrapper, wraps, partial
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints, overload, Union, get_args, get_origin, List
from docstring_parser import parse
from pydantic import BaseModel, Field, validate_call
from collections import OrderedDict

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
ToolConfig = TypeVar('ToolConfig', bound=Dict[str, Any])


class AgentRunException(Exception):
    def __init__(self, exc, user_message: str = None, agent_message: str = None, messages: Optional[List[dict]] = None, stop_execution: bool = False):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution


def is_origin_union_type(origin: Any) -> bool:
    import sys
    if sys.version_info.minor >= 10:
        from types import UnionType
        return origin in [Union, UnionType]
    return origin is Union


def get_json_type_for_py_type(arg: str) -> str:
    if arg in ('int', 'float', 'complex', 'Decimal'):
        return 'number'
    elif arg in ('str', 'string'):
        return 'string'
    elif arg in ('bool', 'boolean'):
        return 'boolean'
    elif arg in ('NoneType', 'None'):
        return 'null'
    elif arg in ('list', 'tuple', 'set', 'frozenset'):
        return 'array'
    elif arg in ('dict', 'mapping'):
        return 'object'
    return 'object'


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
        elif is_origin_union_type(type_origin):
            types = []
            for arg in type_args:
                try:
                    schema = get_json_schema_for_arg(arg)
                    if schema:
                        types.append(schema)
                except Exception:
                    continue
            return {'anyOf': types} if types else None
    json_schema: Dict[str, Any] = {'type': get_json_type_for_py_type(t.__name__)}
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
    def from_callable(cls, c: Callable, strict: bool = False) -> 'Function':
        from inspect import getdoc, isasyncgenfunction, signature
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
        return cls(name=function_name, description=get_entrypoint_docstring(entrypoint=c), parameters=parameters, entrypoint=entrypoint)

    def process_entrypoint(self, strict: bool = False):
        from inspect import getdoc, isasyncgenfunction, signature
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
        self.description = self.description or get_entrypoint_docstring(self.entrypoint)
        if not params_set_by_user:
            self.parameters = parameters
        try:
            if not isasyncgenfunction(self.entrypoint):
                self.entrypoint = validate_call(self.entrypoint, config=dict(arbitrary_types_allowed=True))
        except Exception as e:
            print(f'Failed to add validate decorator to entrypoint: {e}')

    def get_type_name(self, t: Type[T]):
        name = str(t)
        if 'list' in name or 'dict' in name:
            return name
        else:
            return t.__name__

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        if self.entrypoint is None:
            return None
        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get('return', None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)
        function_info = {'name': self.name, 'description': self.description, 'arguments': self.parameters.get('properties', {}), 'returns': returns}
        return function_info

    def get_definition_for_prompt(self) -> Optional[str]:
        import json
        function_info = self.get_definition_for_prompt_dict()
        if function_info is not None:
            return json.dumps(function_info, indent=2)
        return None

    def _get_cache_key(self, entrypoint_args: Dict[str, Any], call_args: Optional[Dict[str, Any]] = None) -> str:
        from hashlib import md5
        copy_entrypoint_args = entrypoint_args.copy()
        if 'agent' in copy_entrypoint_args:
            del copy_entrypoint_args['agent']
        args_str = str(copy_entrypoint_args)
        kwargs_str = str(sorted((call_args or {}).items()))
        key_str = f'{self.name}:{args_str}:{kwargs_str}'
        return md5(key_str.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        from pathlib import Path
        from tempfile import gettempdir
        base_cache_dir = self.cache_dir or Path(gettempdir()) / 'agno_cache'
        func_cache_dir = Path(base_cache_dir) / 'functions' / self.name
        func_cache_dir.mkdir(parents=True, exist_ok=True)
        return str(func_cache_dir / f'{cache_key}.json')

    def _get_cached_result(self, cache_file: str) -> Optional[Any]:
        import json
        from pathlib import Path
        from time import time
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
        import json
        from time import time
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
        import shutil
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
                from inspect import signature
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
                from inspect import signature
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
        from inspect import signature
        entrypoint_args = {}
        if 'agent' in signature(self.function.entrypoint).parameters:
            entrypoint_args['agent'] = self.function._agent
        if 'fc' in signature(self.function.entrypoint).parameters:
            entrypoint_args['fc'] = self
        return entrypoint_args

    def execute(self) -> bool:
        from inspect import isgenerator
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
                from inspect import signature
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
                from inspect import signature
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
        from inspect import isasyncgen, isasyncgenfunction, iscoroutinefunction, isgenerator
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


@overload
def tool() -> Callable[[F], Function]: ...


@overload
def tool(*, name: Optional[str] = None, 
    description: Optional[str] = None, strict: Optional[bool] = None, sanitize_arguments: Optional[bool] = None, show_result: Optional[bool] = None, stop_after_tool_call: Optional[bool] = None, pre_hook: Optional[Callable] = None, post_hook: Optional[Callable] = None, cache_results: bool = False, cache_dir: Optional[str] = None, cache_ttl: int = 3600) -> Callable[[F], Function]: ...


@overload
def tool(func: F) -> Function: ...


def tool(*args, **kwargs) -> Union[Function, Callable[[F], Function]]:
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
    VALID_KWARGS = frozenset({'name', 'description', 'strict', 'sanitize_arguments', 'show_result', 'stop_after_tool_call', 'pre_hook', 'post_hook', 'cache_results', 'cache_dir', 'cache_ttl'})
    invalid_kwargs = set(kwargs.keys()) - VALID_KWARGS
    if invalid_kwargs:
        raise ValueError(f'Invalid tool configuration arguments: {invalid_kwargs}. Valid arguments are: {sorted(VALID_KWARGS)}')

    def decorator(func: F) -> Function:
        from inspect import getdoc, isasyncgenfunction, iscoroutine, iscoroutinefunction
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
        tool_config = {'name': kwargs.get('name', func.__name__), 'description': kwargs.get('description', getdoc(func)), 'entrypoint': wrapper, 'cache_results': kwargs.get('cache_results', False), 'cache_dir': kwargs.get('cache_dir'), 'cache_ttl': kwargs.get('cache_ttl', 3600), **{k: v
                for k, v in kwargs.items()
                if k not in ['name', 'description', 'cache_results', 'cache_dir', 'cache_ttl'] and v is not None}}
        return Function(**tool_config)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


def get_entrypoint_docstring(entrypoint: Callable) -> str:
    from inspect import getdoc
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
