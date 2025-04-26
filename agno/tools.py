from functools import update_wrapper, wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload
from functools import partial
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints
from docstring_parser import parse
from pydantic import BaseModel, Field, validate_call
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
ToolConfig = TypeVar("ToolConfig", bound=Dict[str, Any])
from typing import Any, Dict, Optional, Union, get_args, get_origin
from typing import List, Optional, Union

class AgentRunException(Exception):

    def __init__(self,
        exc,
        user_message: str = None,
        agent_message: str = None,
        messages: Optional[List[dict]] = None,
        stop_execution: bool = False):
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
    """
    Get the JSON schema type for a given type.
    :param arg: The type to get the JSON schema type for.
    :return: The JSON schema type.
    """
    # print(f"Getting JSON type for: {arg}")
    if arg in ("int", "float", "complex", "Decimal"):
        return "number"
    elif arg in ("str", "string"):
        return "string"
    elif arg in ("bool", "boolean"):
        return "boolean"
    elif arg in ("NoneType", "None"):
        return "null"
    elif arg in ("list", "tuple", "set", "frozenset"):
        return "array"
    elif arg in ("dict", "mapping"):
        return "object"
    # If the type is not recognized, return "object"
    return "object"

def get_json_schema_for_arg(t: Any) -> Optional[Dict[str, Any]]:
    # print(f"Getting JSON schema for arg: {t}")
    type_args = get_args(t)
    # print(f"Type args: {type_args}")
    type_origin = get_origin(t)
    # print(f"Type origin: {type_origin}")
    if type_origin is not None:
        if type_origin in (list, tuple, set, frozenset):
            json_schema_for_items = get_json_schema_for_arg(type_args[0]) if type_args else {"type": "string"}
            return {"type": "array", "items": json_schema_for_items}
        elif type_origin is dict:
            # Handle both key and value types for dictionaries
            key_schema = get_json_schema_for_arg(type_args[0]) if type_args else {"type": "string"}
            value_schema = get_json_schema_for_arg(type_args[1]) if len(type_args) > 1 else {"type": "string"}
            return {"type": "object", "propertyNames": key_schema, "additionalProperties": value_schema}
        elif is_origin_union_type(type_origin):
            types = []
            for arg in type_args:
                try:
                    schema = get_json_schema_for_arg(arg)
                    if schema:
                        types.append(schema)
                except Exception:
                    continue
            return {"anyOf": types} if types else None
    json_schema: Dict[str, Any] = {"type": get_json_type_for_py_type(t.__name__)}
    if json_schema["type"] == "object":
        json_schema["properties"] = {}
        json_schema["additionalProperties"] = False
    return json_schema

def get_json_schema(type_hints: Dict[str, Any], param_descriptions: Optional[Dict[str, str]] = None, strict: bool = False) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    if strict:
        json_schema["additionalProperties"] = False
    for k, v in type_hints.items():
        # print(f"Parsing arg: {k} | {v}")
        if k == "return":
            continue
        try:
            # Check if type is Optional (Union with NoneType)
            type_origin = get_origin(v)
            type_args = get_args(v)
            is_optional = type_origin is Union and len(type_args) == 2 and any(arg is type(None) for arg in type_args)
            # Get the actual type if it's Optional
            if is_optional:
                v = next(arg for arg in type_args if arg is not type(None))
            # Handle cases with no type hint
            if v:
                arg_json_schema = get_json_schema_for_arg(v)
            else:
                arg_json_schema = {}
            if arg_json_schema is not None:
                if is_optional:
                    # Handle null type for optional fields
                    if isinstance(arg_json_schema["type"], list):
                        arg_json_schema["type"].append("null")
                    else:
                        arg_json_schema["type"] = [arg_json_schema["type"], "null"]
                # Add description
                if param_descriptions and k in param_descriptions and param_descriptions[k]:
                    arg_json_schema["description"] = param_descriptions[k]
                json_schema["properties"][k] = arg_json_schema
            else:
                print(f"Could not parse argument {k} of type {v}")
        except Exception as e:
            print(f"Error processing argument {k}: {str(e)}")
            continue
    return json_schema

class Function(BaseModel):
    """Model for storing functions that can be called by an agent."""
    # The name of the function to be called.
    # Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: str
    # A description of what the function does, used by the model to choose when and how to call the function.

    description: Optional[str] = None
    # The parameters the functions accepts, described as a JSON Schema object.
    # To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON Schema object describing function parameters")
    strict: Optional[bool] = None
    # The function to be called.
    entrypoint: Optional[Callable] = None
    # If True, the entrypoint processing is skipped and the Function is used as is.
    skip_entrypoint_processing: bool = False
    # If True, the arguments are sanitized before being passed to the function.
    sanitize_arguments: bool = True
    # If True, the function call will show the result along with sending it to the model.
    show_result: bool = False
    # If True, the agent will stop after the function call.
    stop_after_tool_call: bool = False
    # Hook that runs before the function is executed.
    # If defined, can accept the FunctionCall instance as a parameter.
    pre_hook: Optional[Callable] = None
    # Hook that runs after the function is executed, regardless of success/failure.
    # If defined, can accept the FunctionCall instance as a parameter.
    post_hook: Optional[Callable] = None
    # Caching configuration
    cache_results: bool = False
    cache_dir: Optional[str] = None
    cache_ttl: int = 3600
    # --*-- FOR INTERNAL USE ONLY --*--
    # The agent that the function is associated with
    _agent: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={"name", "description", "parameters", "strict"})
    @classmethod

    def from_callable(cls, c: Callable, strict: bool = False) -> "Function":
        from inspect import getdoc, isasyncgenfunction, signature

        function_name = c.__name__
        parameters = {"type": "object", "properties": {}, "required": []}
        try:
            sig = signature(c)
            type_hints = get_type_hints(c)
            # If function has an the agent argument, remove the agent parameter from the type hints
            if "agent" in sig.parameters:
                del type_hints["agent"]
            # print(f"Type hints for {function_name}: {type_hints}")
            # Filter out return type and only process parameters
            param_type_hints = {
                name: type_hints.get(name) for name in sig.parameters if name != "return" and name != "agent"
            }
            # Parse docstring for parameters
            param_descriptions = {}
            if docstring := getdoc(c):
                parsed_doc = parse(docstring)
                param_docs = parsed_doc.params
                if param_docs is not None:
                    for param in param_docs:
                        param_name = param.arg_name
                        param_type = param.type_name
                        param_descriptions[param_name] = f"({param_type}) {param.description}"
            # Get JSON schema for parameters only
            parameters = get_json_schema(type_hints=param_type_hints, param_descriptions=param_descriptions, strict=strict)
            # If strict=True mark all fields as required
            # See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas#all-fields-must-be-required
            if strict:
                parameters["required"] = [name for name in parameters["properties"] if name != "agent"]
            else:
                # Mark a field as required if it has no default value
                parameters["required"] = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == param.empty and name != "self" and name != "agent"
                ]
            # print(f"JSON schema for {function_name}: {parameters}")
        except Exception as e:
            print(f"Could not parse args for {function_name}: {e}", exc_info=True)
        # Don't wrap async generator with validate_call
        if isasyncgenfunction(c):
            entrypoint = c
        else:
            entrypoint = validate_call(c, config=dict(arbitrary_types_allowed=True))
        return cls(name=function_name,
            description=get_entrypoint_docstring(entrypoint=c),
            parameters=parameters,
            entrypoint=entrypoint)

    def process_entrypoint(self, strict: bool = False):
        """Process the entrypoint and make it ready for use by an agent."""
        from inspect import getdoc, isasyncgenfunction, signature

        if self.skip_entrypoint_processing:
            return
        if self.entrypoint is None:
            return
        parameters = {"type": "object", "properties": {}, "required": []}
        params_set_by_user = False
        # If the user set the parameters (i.e. they are different from the default), we should keep them
        if self.parameters != parameters:
            params_set_by_user = True
        try:
            sig = signature(self.entrypoint)
            type_hints = get_type_hints(self.entrypoint)
            # If function has an the agent argument, remove the agent parameter from the type hints
            if "agent" in sig.parameters:
                del type_hints["agent"]
            # print(f"Type hints for {self.name}: {type_hints}")
            # Filter out return type and only process parameters
            param_type_hints = {
                name: type_hints.get(name) for name in sig.parameters if name != "return" and name != "agent"
            }
            # Parse docstring for parameters
            param_descriptions = {}
            if docstring := getdoc(self.entrypoint):
                parsed_doc = parse(docstring)
                param_docs = parsed_doc.params
                if param_docs is not None:
                    for param in param_docs:
                        param_name = param.arg_name
                        param_type = param.type_name
                        # TODO: We should use type hints first, then map param types in docs to json schema types.
                        # This is temporary to not lose information
                        param_descriptions[param_name] = f"({param_type}) {param.description}"
            # Get JSON schema for parameters only
            parameters = get_json_schema(type_hints=param_type_hints, param_descriptions=param_descriptions, strict=strict)
            # If strict=True mark all fields as required
            # See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas#all-fields-must-be-required
            if strict:
                parameters["required"] = [name for name in parameters["properties"] if name != "agent"]
            else:
                # Mark a field as required if it has no default value
                parameters["required"] = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == param.empty and name != "self" and name != "agent"
                ]
            if params_set_by_user:
                self.parameters['additionalProperties'] = False
                if strict:
                    self.parameters["required"] = [name for name in self.parameters["properties"] if name != "agent"]
                else:
                    # Mark a field as required if it has no default value
                    self.parameters["required"] = [
                        name
                        for name, param in sig.parameters.items()
                        if param.default == param.empty and name != "self" and name != "agent"
                    ]
            # print(f"JSON schema for {self.name}: {parameters}")
        except Exception as e:
            print(f"Could not parse args for {self.name}: {e}", exc_info=True)
        self.description = self.description or get_entrypoint_docstring(self.entrypoint)
        if not params_set_by_user:
            self.parameters = parameters
        try:
            # Don't wrap async generator with validate_call
            if not isasyncgenfunction(self.entrypoint):
                self.entrypoint = validate_call(self.entrypoint,
                                                config=dict(arbitrary_types_allowed=True))
        except Exception as e:
            print(f"Failed to add validate decorator to entrypoint: {e}")

    def get_type_name(self, t: Type[T]):
        name = str(t)
        if "list" in name or "dict" in name:
            return name
        else:
            return t.__name__

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        """Returns a function definition that can be used in a prompt."""
        if self.entrypoint is None:
            return None
        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)
        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return function_info

    def get_definition_for_prompt(self) -> Optional[str]:
        """Returns a function definition that can be used in a prompt."""
        import json
        function_info = self.get_definition_for_prompt_dict()
        if function_info is not None:
            return json.dumps(function_info, indent=2)
        return None

    def _get_cache_key(self, entrypoint_args: Dict[str, Any], call_args: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key based on function name and arguments."""
        from hashlib import md5
        copy_entrypoint_args = entrypoint_args.copy()
        # Remove agent from entrypoint_args
        if "agent" in copy_entrypoint_args:
            del copy_entrypoint_args["agent"]
        args_str = str(copy_entrypoint_args)
        kwargs_str = str(sorted((call_args or {}).items()))
        key_str = f"{self.name}:{args_str}:{kwargs_str}"
        return md5(key_str.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get the full path for the cache file."""
        from pathlib import Path
        from tempfile import gettempdir
        base_cache_dir = self.cache_dir or Path(gettempdir()) / "agno_cache"
        func_cache_dir = Path(base_cache_dir) / "functions" / self.name
        func_cache_dir.mkdir(parents=True, exist_ok=True)
        return str(func_cache_dir / f"{cache_key}.json")

    def _get_cached_result(self, cache_file: str) -> Optional[Any]:
        """Retrieve cached result if valid."""
        import json
        from pathlib import Path
        from time import time
        cache_path = Path(cache_file)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r") as f:
                cache_data = json.load(f)
            timestamp = cache_data.get("timestamp", 0)
            result = cache_data.get("result")
            if time() - timestamp <= self.cache_ttl:
                return result
            # Remove expired entry
            cache_path.unlink()
        except Exception as e:
            print(f"Error reading cache: {e}")
        return None

    def _save_to_cache(self, cache_file: str, result: Any):
        """Save result to cache."""
        import json
        from time import time
        try:
            with open(cache_file, "w") as f:
                json.dump({"timestamp": time(), "result": result}, f)
        except Exception as e:
            print(f"Error writing cache: {e}")

class FunctionCall(BaseModel):
    """Model for Function Calls"""
    # The function to be called.
    function: Function
    # The arguments to call the function with.
    arguments: Optional[Dict[str, Any]] = None
    # The result of the function call.
    result: Optional[Any] = None
    # The ID of the function call.
    call_id: Optional[str] = None
    # Error while parsing arguments or running the function.
    error: Optional[str] = None

    def get_call_str(self) -> str:
        """Returns a string representation of the function call."""
        import shutil
        # Get terminal width, default to 80 if can't determine
        term_width = shutil.get_terminal_size().columns or 80
        max_arg_len = max(20, (term_width - len(self.function.name) - 4) // 2)
        if self.arguments is None:
            return f"{self.function.name}()"
        trimmed_arguments = {}
        for k, v in self.arguments.items():
            if isinstance(v, str) and len(str(v)) > max_arg_len:
                trimmed_arguments[k] = "..."
            else:
                trimmed_arguments[k] = v
        call_str = f"{self.function.name}({', '.join([f'{k}={v}' for k, v in trimmed_arguments.items()])})"
        # If call string is too long, truncate arguments
        if len(call_str) > term_width:
            return f"{self.function.name}(...)"
        return call_str

    def _handle_pre_hook(self):
        """Handles the pre-hook for the function call."""
        if self.function.pre_hook is not None:
            try:
                from inspect import signature
                pre_hook_args = {}
                # Check if the pre-hook has and agent argument
                if "agent" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["agent"] = self.function._agent
                # Check if the pre-hook has an fc argument
                if "fc" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["fc"] = self
                self.function.pre_hook(**pre_hook_args)
            except AgentRunException as e:
                print(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                print(f"Error in pre-hook callback: {e}")
                print(e)

    def _handle_post_hook(self):
        """Handles the post-hook for the function call."""
        if self.function.post_hook is not None:
            try:
                from inspect import signature
                post_hook_args = {}
                # Check if the post-hook has and agent argument
                if "agent" in signature(self.function.post_hook).parameters:
                    post_hook_args["agent"] = self.function._agent
                # Check if the post-hook has an fc argument
                if "fc" in signature(self.function.post_hook).parameters:
                    post_hook_args["fc"] = self
                self.function.post_hook(**post_hook_args)
            except AgentRunException as e:
                print(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                print(f"Error in post-hook callback: {e}")
                print(e)

    def _build_entrypoint_args(self) -> Dict[str, Any]:
        """Builds the arguments for the entrypoint."""
        from inspect import signature
        entrypoint_args = {}
        # Check if the entrypoint has an agent argument
        if "agent" in signature(self.function.entrypoint).parameters:
            entrypoint_args["agent"] = self.function._agent
        # Check if the entrypoint has an fc argument
        if "fc" in signature(self.function.entrypoint).parameters:
            entrypoint_args["fc"] = self
        return entrypoint_args

    def execute(self) -> bool:
        """Runs the function call."""
        from inspect import isgenerator
        if self.function.entrypoint is None:
            return False
        print(f"Running: {self.get_call_str()}")
        function_call_success = False
        # Execute pre-hook if it exists
        self._handle_pre_hook()
        entrypoint_args = self._build_entrypoint_args()
        # Check cache if enabled and not a generator function
        if self.function.cache_results and not isgenerator(self.function.entrypoint):
            cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
            cache_file = self.function._get_cache_file_path(cache_key)
            cached_result = self.function._get_cached_result(cache_file)
            if cached_result is not None:
                print(f"Cache hit for: {self.get_call_str()}")
                self.result = cached_result
                function_call_success = True
                return function_call_success
        # Execute function
        try:
            if self.arguments == {} or self.arguments is None:
                result = self.function.entrypoint(**entrypoint_args)
            else:
                result = self.function.entrypoint(**entrypoint_args, **self.arguments)
            # Handle generator case
            if isgenerator(result):
                self.result = result  # Store generator directly, can't cache
            else:
                self.result = result
                # Only cache non-generator results
                if self.function.cache_results:
                    cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
                    cache_file = self.function._get_cache_file_path(cache_key)
                    self.function._save_to_cache(cache_file, self.result)
            function_call_success = True
        except AgentRunException as e:
            print(f"{e.__class__.__name__}: {e}")
            self.error = str(e)
            raise
        except Exception as e:
            print(f"Could not run function {self.get_call_str()}")
            print(e)
            self.error = str(e)
            return function_call_success
        # Execute post-hook if it exists
        self._handle_post_hook()
        return function_call_success
    async def _handle_pre_hook_async(self):
        """Handles the async pre-hook for the function call."""
        if self.function.pre_hook is not None:
            try:
                from inspect import signature
                pre_hook_args = {}
                # Check if the pre-hook has an agent argument
                if "agent" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["agent"] = self.function._agent
                # Check if the pre-hook has an fc argument
                if "fc" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["fc"] = self
                await self.function.pre_hook(**pre_hook_args)
            except AgentRunException as e:
                print(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                print(f"Error in pre-hook callback: {e}")
                print(e)
    async def _handle_post_hook_async(self):
        """Handles the async post-hook for the function call."""
        if self.function.post_hook is not None:
            try:
                from inspect import signature
                post_hook_args = {}
                # Check if the post-hook has an agent argument
                if "agent" in signature(self.function.post_hook).parameters:
                    post_hook_args["agent"] = self.function._agent
                # Check if the post-hook has an fc argument
                if "fc" in signature(self.function.post_hook).parameters:
                    post_hook_args["fc"] = self
                await self.function.post_hook(**post_hook_args)
            except AgentRunException as e:
                print(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                print(f"Error in post-hook callback: {e}")
                print(e)
    async def aexecute(self) -> bool:
        """Runs the function call asynchronously."""
        from inspect import isasyncgen, isasyncgenfunction, iscoroutinefunction, isgenerator
        if self.function.entrypoint is None:
            return False
        print(f"Running: {self.get_call_str()}")
        function_call_success = False
        # Execute pre-hook if it exists
        if iscoroutinefunction(self.function.pre_hook):
            await self._handle_pre_hook_async()
        else:
            self._handle_pre_hook()
        entrypoint_args = self._build_entrypoint_args()
        # Check cache if enabled and not a generator function
        if self.function.cache_results and not (isasyncgen(self.function.entrypoint) or isgenerator(self.function.entrypoint)):
            cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
            cache_file = self.function._get_cache_file_path(cache_key)
            cached_result = self.function._get_cached_result(cache_file)
            if cached_result is not None:
                print(f"Cache hit for: {self.get_call_str()}")
                self.result = cached_result
                function_call_success = True
                return function_call_success
        # Execute function
        try:
            if self.arguments == {} or self.arguments is None:
                result = self.function.entrypoint(**entrypoint_args)
                if isasyncgen(self.function.entrypoint) or isasyncgenfunction(self.function.entrypoint):
                    self.result = result  # Store async generator directly
                else:
                    self.result = await result
            else:
                result = self.function.entrypoint(**entrypoint_args, **self.arguments)
                if isasyncgen(self.function.entrypoint) or isasyncgenfunction(self.function.entrypoint):
                    self.result = result  # Store async generator directly
                else:
                    self.result = await result
            # Only cache if not a generator
            if self.function.cache_results and not (isgenerator(self.result) or isasyncgen(self.result)):
                cache_key = self.function._get_cache_key(entrypoint_args, self.arguments)
                cache_file = self.function._get_cache_file_path(cache_key)
                self.function._save_to_cache(cache_file, self.result)
            function_call_success = True
        except AgentRunException as e:
            print(f"{e.__class__.__name__}: {e}")
            self.error = str(e)
            raise
        except Exception as e:
            print(f"Could not run function {self.get_call_str()}")
            print(e)
            self.error = str(e)
            return function_call_success
        # Execute post-hook if it exists
        if iscoroutinefunction(self.function.post_hook):
            await self._handle_post_hook_async()
        else:
            self._handle_post_hook()
        return function_call_success

@overload
def tool() -> Callable[[F], Function]: ...

@overload
def tool(*,
    name: Optional[str] = None,

    description: Optional[str] = None,
    strict: Optional[bool] = None,
    sanitize_arguments: Optional[bool] = None,
    show_result: Optional[bool] = None,
    stop_after_tool_call: Optional[bool] = None,
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
    cache_results: bool = False,
    cache_dir: Optional[str] = None,
    cache_ttl: int = 3600) -> Callable[[F], Function]: ...

@overload
def tool(func: F) -> Function: ...

def tool(*args, **kwargs) -> Union[Function, Callable[[F], Function]]:
    """Decorator to convert a function into a Function that can be used by an agent.
    Args:
        name: Optional[str] - Override for the function name
        description: Optional[str] - Override for the function description
        strict: Optional[bool] - Flag for strict parameter checking
        sanitize_arguments: Optional[bool] - If True, arguments are sanitized before passing to function
        show_result: Optional[bool] - If True, shows the result after function call
        stop_after_tool_call: Optional[bool] - If True, the agent will stop after the function call.
        pre_hook: Optional[Callable] - Hook that runs before the function is executed.
        post_hook: Optional[Callable] - Hook that runs after the function is executed.
        cache_results: bool - If True, enable caching of function results
        cache_dir: Optional[str] - Directory to store cache files
        cache_ttl: int - Time-to-live for cached results in seconds
    Returns:
        Union[Function, Callable[[F], Function]]: Decorated function or decorator
    Examples:
        @tool
        def my_function():
            pass
        @tool(name="custom_name", description="Custom description")
        def another_function():
            pass
        @tool
        async def my_async_function():
            pass
    """
    # Move valid kwargs to a frozen set at module level
    VALID_KWARGS = frozenset({
            "name",
            "description",
            "strict",
            "sanitize_arguments",
            "show_result",
            "stop_after_tool_call",
            "pre_hook",
            "post_hook",
            "cache_results",
            "cache_dir",
            "cache_ttl",
        })
    # Improve error message with more context
    invalid_kwargs = set(kwargs.keys()) - VALID_KWARGS
    if invalid_kwargs:
        raise ValueError(f"Invalid tool configuration arguments: {invalid_kwargs}. Valid arguments are: {sorted(VALID_KWARGS)}")

    def decorator(func: F) -> Function:
        from inspect import getdoc, isasyncgenfunction, iscoroutine, iscoroutinefunction
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in tool {func.__name__!r}: {e!r}",
                    exc_info=True)
                raise
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Error in async tool {func.__name__!r}: {e!r}",
                    exc_info=True)
                raise
        @wraps(func)
        async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in async generator tool {func.__name__!r}: {e!r}",
                    exc_info=True)
                raise
        # Choose appropriate wrapper based on function type
        if isasyncgenfunction(func):
            wrapper = async_gen_wrapper
        elif iscoroutinefunction(func) or iscoroutine(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        # Preserve the original signature and metadata
        update_wrapper(wrapper, func)
        # Create Function instance with any provided kwargs
        tool_config = {
            "name": kwargs.get("name", func.__name__),
            "description": kwargs.get("description", getdoc(func)),  # Get docstring if description not provided
            "entrypoint": wrapper,
            "cache_results": kwargs.get("cache_results", False),
            "cache_dir": kwargs.get("cache_dir"),
            "cache_ttl": kwargs.get("cache_ttl", 3600),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["name", "description", "cache_results", "cache_dir", "cache_ttl"] and v is not None
            },
        }
        return Function(**tool_config)
    # Handle both @tool and @tool() cases
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator
def get_entrypoint_docstring(entrypoint: Callable) -> str:
    from inspect import getdoc
    if isinstance(entrypoint, partial):
        return str(entrypoint)
    doc = getdoc(entrypoint)
    if not doc:
        return ""
    parsed = parse(doc)
    lines = []
    if parsed.short_description:
        lines.append(parsed.short_description)
    if parsed.long_description:
        lines.extend(parsed.long_description.split("\n"))
    return "\n".join(lines)

class Toolkit:

    def __init__(self,
        name: str = "toolkit",
        cache_results: bool = False,
        cache_ttl: int = 3600,
        cache_dir: Optional[str] = None):
        """Initialize a new Toolkit.
        Args:
            name: A descriptive name for the toolkit
            cache_results (bool): Enable in-memory caching of function results.
            cache_ttl (int): Time-to-live for cached results in seconds.
            cache_dir (Optional[str]): Directory to store cache files. Defaults to system temp dir.
        """
        self.name: str = name
        self.functions: Dict[str, Function] = OrderedDict()
        self.cache_results: bool = cache_results
        self.cache_ttl: int = cache_ttl
        self.cache_dir: Optional[str] = cache_dir

    def register(self, function: Callable[..., Any], sanitize_arguments: bool = True):
        """Register a function with the toolkit.
        Args:
            function: The callable to register
        Returns:
            The registered function
        """
        try:
            f = Function(name=function.__name__,
                entrypoint=function,
                sanitize_arguments=sanitize_arguments,
                cache_results=self.cache_results,
                cache_dir=self.cache_dir,
                cache_ttl=self.cache_ttl)
            self.functions[f.name] = f
            print(f"Function: {f.name} registered with {self.name}")
        except Exception as e:
            print(f"Failed to create Function for: {function.__name__}")
            raise e

    def instructions(self) -> str:
        return ""

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>"

    def __str__(self):
        return self.__repr__()

from agno.tools import Toolkit

try:
    from newspaper import Article
except ImportError:
    raise ImportError("`newspaper3k` not installed. Please run `pip install newspaper3k lxml_html_clean`.")
import json
from typing import Any, Optional
from agno.tools import Toolkit

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")
import json
from agno.tools import Toolkit

try:
    import yfinance as yf
except ImportError:
    raise ImportError("`yfinance` not installed. Please install using `pip install yfinance`.")

class YFinanceTools(Toolkit):
    """
    YFinanceTools is a toolkit for getting financial data from Yahoo Finance.
    Args:
        stock_price (bool): Whether to get the current stock price.
        company_info (bool): Whether to get company information.
        stock_fundamentals (bool): Whether to get stock fundamentals.
        income_statements (bool): Whether to get income statements.
        key_financial_ratios (bool): Whether to get key financial ratios.
        analyst_recommendations (bool): Whether to get analyst recommendations.
        company_news (bool): Whether to get company news.
        technical_indicators (bool): Whether to get technical indicators.
        historical_prices (bool): Whether to get historical prices.
        enable_all (bool): Whether to enable all tools.
    """

    def __init__(self,
        stock_price: bool = True,
        company_info: bool = False,
        stock_fundamentals: bool = False,
        income_statements: bool = False,
        key_financial_ratios: bool = False,
        analyst_recommendations: bool = False,
        company_news: bool = False,
        technical_indicators: bool = False,
        historical_prices: bool = False,
        enable_all: bool = False,
        **kwargs):
        super().__init__(name="yfinance_tools", **kwargs)
        if stock_price or enable_all:
            self.register(self.get_current_stock_price)
        if company_info or enable_all:
            self.register(self.get_company_info)
        if stock_fundamentals or enable_all:
            self.register(self.get_stock_fundamentals)
        if income_statements or enable_all:
            self.register(self.get_income_statements)
        if key_financial_ratios or enable_all:
            self.register(self.get_key_financial_ratios)
        if analyst_recommendations or enable_all:
            self.register(self.get_analyst_recommendations)
        if company_news or enable_all:
            self.register(self.get_company_news)
        if technical_indicators or enable_all:
            self.register(self.get_technical_indicators)
        if historical_prices or enable_all:
            self.register(self.get_historical_stock_prices)

    def get_current_stock_price(self, symbol: str) -> str:
        """
        Use this function to get the current stock price for a given symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: The current stock price or error message.
        """
        try:
            print(f"Fetching current price for {symbol}")
            stock = yf.Ticker(symbol)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    def get_company_info(self, symbol: str) -> str:
        """Use this function to get company information and overview for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: JSON containing company profile and overview.
        """
        try:
            company_info_full = yf.Ticker(symbol).info
            if company_info_full is None:
                return f"Could not fetch company info for {symbol}"
            print(f"Fetching company info for {symbol}")
            company_info_cleaned = {
                "Name": company_info_full.get("shortName"),
                "Symbol": company_info_full.get("symbol"),
                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                "Sector": company_info_full.get("sector"),
                "Industry": company_info_full.get("industry"),
                "Address": company_info_full.get("address1"),
                "City": company_info_full.get("city"),
                "State": company_info_full.get("state"),
                "Zip": company_info_full.get("zip"),
                "Country": company_info_full.get("country"),
                "EPS": company_info_full.get("trailingEps"),
                "P/E Ratio": company_info_full.get("trailingPE"),
                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                "Website": company_info_full.get("website"),
                "Summary": company_info_full.get("longBusinessSummary"),
                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                "Employees": company_info_full.get("fullTimeEmployees"),
                "Total Cash": company_info_full.get("totalCash"),
                "Free Cash flow": company_info_full.get("freeCashflow"),
                "Operating Cash flow": company_info_full.get("operatingCashflow"),
                "EBITDA": company_info_full.get("ebitda"),
                "Revenue Growth": company_info_full.get("revenueGrowth"),
                "Gross Margins": company_info_full.get("grossMargins"),
                "Ebitda Margins": company_info_full.get("ebitdaMargins"),
            }
            return json.dumps(company_info_cleaned, indent=2)
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """
        Use this function to get the historical stock price for a given symbol.
        Args:
            symbol (str): The stock symbol.
            period (str): The period for which to retrieve historical prices. Defaults to "1mo".
                        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): The interval between data points. Defaults to "1d".
                        Valid intervals: 1d,5d,1wk,1mo,3mo
        Returns:
          str: The current stock price or error message.
        """
        try:
            print(f"Fetching historical prices for {symbol}")
            stock = yf.Ticker(symbol)
            historical_price = stock.history(period=period, interval=interval)
            return historical_price.to_json(orient="index")
        except Exception as e:
            return f"Error fetching historical prices for {symbol}: {e}"

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Use this function to get fundamental data for a given stock symbol yfinance API.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: A JSON string containing fundamental data or an error message.
                Keys:
                    - 'symbol': The stock symbol.
                    - 'company_name': The long name of the company.
                    - 'sector': The sector to which the company belongs.
                    - 'industry': The industry to which the company belongs.
                    - 'market_cap': The market capitalization of the company.
                    - 'pe_ratio': The forward price-to-earnings ratio.
                    - 'pb_ratio': The price-to-book ratio.
                    - 'dividend_yield': The dividend yield.
                    - 'eps': The trailing earnings per share.
                    - 'beta': The beta value of the stock.
                    - '52_week_high': The 52-week high price of the stock.
                    - '52_week_low': The 52-week low price of the stock.
        """
        try:
            print(f"Fetching fundamentals for {symbol}")
            stock = yf.Ticker(symbol)
            info = stock.info
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            }
            return json.dumps(fundamentals, indent=2)
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {e}"

    def get_income_statements(self, symbol: str) -> str:
        """Use this function to get income statements for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            dict: JSON containing income statements or an empty dictionary.
        """
        try:
            print(f"Fetching income statements for {symbol}")
            stock = yf.Ticker(symbol)
            financials = stock.financials
            return financials.to_json(orient="index")
        except Exception as e:
            return f"Error fetching income statements for {symbol}: {e}"

    def get_key_financial_ratios(self, symbol: str) -> str:
        """Use this function to get key financial ratios for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            dict: JSON containing key financial ratios.
        """
        try:
            print(f"Fetching key financial ratios for {symbol}")
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Use this function to get analyst recommendations for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: JSON containing analyst recommendations.
        """
        try:
            print(f"Fetching analyst recommendations for {symbol}")
            stock = yf.Ticker(symbol)
            recommendations = stock.recommendations
            return recommendations.to_json(orient="index")
        except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Use this function to get company news and press releases for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
            num_stories (int): The number of news stories to return. Defaults to 3.
        Returns:
            str: JSON containing company news and press releases.
        """
        try:
            print(f"Fetching company news for {symbol}")
            news = yf.Ticker(symbol).news
            return json.dumps(news[:num_stories], indent=2)
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> str:
        """Use this function to get technical indicators for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
            period (str): The time period for which to retrieve technical indicators.
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo.
        Returns:
            str: JSON containing technical indicators.
        """
        try:
            print(f"Fetching technical indicators for {symbol}")
            indicators = yf.Ticker(symbol).history(period=period)
            return indicators.to_json(orient="index")
        except Exception as e:
            return f"Error fetching technical indicators for {symbol}: {e}"

class DuckDuckGoTools(Toolkit):
    """
    DuckDuckGo is a toolkit for searching DuckDuckGo easily.
    Args:
        search (bool): Enable DuckDuckGo search function.
        news (bool): Enable DuckDuckGo news function.
        modifier (Optional[str]): A modifier to be used in the search request.
        fixed_max_results (Optional[int]): A fixed number of maximum results.
        headers (Optional[Any]): Headers to be used in the search request.
        proxy (Optional[str]): Proxy to be used in the search request.
        proxies (Optional[Any]): A list of proxies to be used in the search request.
        timeout (Optional[int]): The maximum number of seconds to wait for a response.
    """

    def __init__(self,
        search: bool = True,
        news: bool = True,
        modifier: Optional[str] = None,
        fixed_max_results: Optional[int] = None,
        headers: Optional[Any] = None,
        proxy: Optional[str] = None,
        proxies: Optional[Any] = None,
        timeout: Optional[int] = 10,
        verify_ssl: bool = True,
        **kwargs):
        super().__init__(name="duckduckgo", **kwargs)
        self.headers: Optional[Any] = headers
        self.proxy: Optional[str] = proxy
        self.proxies: Optional[Any] = proxies
        self.timeout: Optional[int] = timeout
        self.fixed_max_results: Optional[int] = fixed_max_results
        self.modifier: Optional[str] = modifier
        self.verify_ssl: bool = verify_ssl
        if search:
            self.register(self.duckduckgo_search)
        if news:
            self.register(self.duckduckgo_news)

    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search DuckDuckGo for a query.
        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.
        Returns:
            The result from DuckDuckGo.
        """
        actual_max_results = self.fixed_max_results or max_results
        search_query = f"{self.modifier} {query}" if self.modifier else query
        print(f"Searching DDG for: {search_query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.text(keywords=search_query, max_results=actual_max_results), indent=2)

    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        """Use this function to get the latest news from DuckDuckGo.
        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.
        Returns:
            The latest news from DuckDuckGo.
        """
        actual_max_results = self.fixed_max_results or max_results
        print(f"Searching DDG news for: {query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.news(keywords=query, max_results=actual_max_results), indent=2)
class NewspaperTools(Toolkit):
    """
    Newspaper is a tool for getting the text of an article from a URL.
    Args:
        get_article_text (bool): Whether to get the text of an article from a URL.
    """

    def __init__(self, get_article_text: bool = True, **kwargs):
        super().__init__(name="newspaper_toolkit", **kwargs)
        if get_article_text:
            self.register(self.get_article_text)

    def get_article_text(self, url: str) -> str:
        """Get the text of an article from a URL.
        Args:
            url (str): The URL of the article.
        Returns:
            str: The text of the article.
        """
        try:
            print(f"Reading news: {url}")
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return f"Error getting article text from {url}: {e}"
