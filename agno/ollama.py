from dataclasses import dataclass, field
from textwrap import dedent
from agno.models import Message, MessageMetrics
from agno.tools import FunctionCall
from agno.models import Timer
import json
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, AsyncIterator
from pydantic import BaseModel
from agno.models import Model
from agno.models import Message
from agno.models import ModelResponse
from ollama import AsyncClient as AsyncOllamaClient
from ollama import Client as OllamaClient
from ollama._types import ChatResponse, Message as OllamaMessage


def extract_tool_call_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extracting the content between the tags
    return text[start_index:end_index].strip()


def remove_tool_calls_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    """Remove multiple tool calls from a string."""
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
    """
    A class for interacting with Ollama models.

    For more information, see: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    id: str = "llama3.1"
    name: str = "Ollama"
    provider: str = "Ollama"

    supports_native_structured_outputs: bool = True

    # Request parameters
    format: Optional[Any] = None
    options: Optional[Any] = None
    keep_alive: Optional[Union[float, str]] = None
    request_params: Optional[Dict[str, Any]] = None

    # Client parameters
    host: Optional[str] = None
    timeout: Optional[Any] = None
    client_params: Optional[Dict[str, Any]] = None

    # Ollama clients
    client: Optional[OllamaClient] = None
    async_client: Optional[AsyncOllamaClient] = None

    def _get_client_params(self) -> Dict[str, Any]:
        base_params = {
            "host": self.host,
            "timeout": self.timeout,
        }
        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}
        # Add additional client params if provided
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def get_client(self) -> OllamaClient:
        """
        Returns an Ollama client.

        Returns:
            OllamaClient: An instance of the Ollama client.
        """
        if self.client is not None:
            return self.client

        self.client = OllamaClient(**self._get_client_params())
        return self.client

    def get_async_client(self) -> AsyncOllamaClient:
        """
        Returns an asynchronous Ollama client.

        Returns:
            AsyncOllamaClient: An instance of the Ollama client.
        """
        if self.async_client is not None:
            return self.async_client

        return AsyncOllamaClient(**self._get_client_params())

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: The API kwargs for the model.
        """
        base_params = {
            "format": self.format,
            "options": self.options,
            "keep_alive": self.keep_alive,
            "request_params": self.request_params,
        }
        # Filter out None values
        request_params = {k: v for k, v in base_params.items() if v is not None}
        # Add tools
        if self._tools is not None and len(self._tools) > 0:
            request_params["tools"] = self._tools
            # Fix optional parameters where the "type" is [type, null]
            for tool in request_params["tools"]:  # type: ignore
                if "parameters" in tool["function"] and "properties" in tool["function"]["parameters"]:  # type: ignore
                    for _, obj in tool["function"]["parameters"].get("properties", {}).items():  # type: ignore
                        if "type" in obj and isinstance(obj["type"], list) and len(obj["type"]) > 1:
                            obj["type"] = obj["type"][0]

        # Add additional request params if provided
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the model.
        """
        model_dict = super().to_dict()
        model_dict.update(
            {
                "format": self.format,
                "options": self.options,
                "keep_alive": self.keep_alive,
                "request_params": self.request_params,
            }
        )
        if self._tools is not None:
            model_dict["tools"] = self._tools
        cleaned_dict = {k: v for k, v in model_dict.items() if v is not None}
        return cleaned_dict

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """
        Format a message into the format expected by Ollama.

        Args:
            message (Message): The message to format.

        Returns:
            Dict[str, Any]: The formatted message.
        """
        _message: Dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "user":
            if message.images is not None:
                message_images = []
                for image in message.images:
                    if image.url is not None:
                        message_images.append(image.image_url_content)
                    if image.filepath is not None:
                        message_images.append(image.filepath)  # type: ignore
                    if image.content is not None and isinstance(image.content, bytes):
                        message_images.append(image.content)
                if message_images:
                    _message["images"] = message_images
        return _message

    def _prepare_request_kwargs_for_invoke(self) -> Dict[str, Any]:
        request_kwargs = self.request_kwargs
        if self.response_format is not None and self.structured_outputs:
            if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                print("Using structured outputs")
                format_schema = self.response_format.model_json_schema()
                if "format" not in request_kwargs:
                    request_kwargs["format"] = format_schema
        return request_kwargs

    def invoke(self, messages: List[Message]) -> Mapping[str, Any]:
        """
        Send a chat request to the Ollama API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Mapping[str, Any]: The response from the API.
        """
        request_kwargs = self._prepare_request_kwargs_for_invoke()

        return self.get_client().chat(
            model=self.id.strip(),
            messages=[self._format_message(m) for m in messages],  # type: ignore
            **request_kwargs,
        )  # type: ignore

    async def ainvoke(self, messages: List[Message]) -> Mapping[str, Any]:
        """
        Sends an asynchronous chat request to the Ollama API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Mapping[str, Any]: The response from the API.
        """
        request_kwargs = self._prepare_request_kwargs_for_invoke()

        return await self.get_async_client().chat(
            model=self.id.strip(),
            messages=[self._format_message(m) for m in messages],  # type: ignore
            **request_kwargs,
        )  # type: ignore

    def invoke_stream(self, messages: List[Message]) -> Iterator[Mapping[str, Any]]:
        """
        Sends a streaming chat request to the Ollama API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Iterator[Mapping[str, Any]]: An iterator of chunks from the API.
        """
        yield from self.get_client().chat(
            model=self.id,
            messages=[self._format_message(m) for m in messages],  # type: ignore
            stream=True,
            **self.request_kwargs,
        )  # type: ignore

    async def ainvoke_stream(self, messages: List[Message]) -> Any:
        """
        Sends an asynchronous streaming chat completion request to the Ollama API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Any: An asynchronous iterator of chunks from the API.
        """
        async_stream = await self.get_async_client().chat(
            model=self.id.strip(),
            messages=[self._format_message(m) for m in messages],  # type: ignore
            stream=True,
            **self.request_kwargs,
        )
        async for chunk in async_stream:  # type: ignore
            yield chunk

    def parse_provider_response(self, response: ChatResponse) -> ModelResponse:
        """
        Parse the provider response.

        Args:
            response (ChatResponse): The response from the provider.

        Returns:
            ModelResponse: The model response.
        """
        model_response = ModelResponse()
        # Get response message
        response_message: OllamaMessage = response.get("message")

        # Parse structured outputs if enabled
        try:
            if (
                self.response_format is not None
                and self.structured_outputs
                and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.content  # type: ignore
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            print(f"Error retrieving structured outputs: {e}")

        if response_message.get("role") is not None:
            model_response.role = response_message.get("role")

        if response_message.get("content") is not None:
            model_response.content = response_message.get("content")

        if response_message.get("tool_calls") is not None:
            if model_response.tool_calls is None:
                model_response.tool_calls = []
            for block in response_message.get("tool_calls"):
                tool_call = block.get("function")
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments")

                function_def = {
                    "name": tool_name,
                    "arguments": (json.dumps(tool_args) if tool_args is not None else None),
                }
                model_response.tool_calls.append({"type": "function", "function": function_def})

        # if response_message.get("images") is not None:
        #     model_response.images = response_message.get("images")

        # Get response usage
        if response.get("done"):
            model_response.response_usage = {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
                "additional_metrics": {
                    "total_duration": response.get("total_duration", 0),
                    "load_duration": response.get("load_duration", 0),
                    "prompt_eval_duration": response.get("prompt_eval_duration", 0),
                    "eval_duration": response.get("eval_duration", 0),
                },
            }

        return model_response

    def parse_provider_response_delta(self, response_delta: ChatResponse) -> ModelResponse:
        """
        Parse the provider response delta.

        Args:
            response_delta (ChatResponse): The response from the provider.

        Returns:
            Iterator[ModelResponse]: An iterator of the model response.
        """
        model_response = ModelResponse()

        response_message = response_delta.get("message")

        if response_message is not None:
            content_delta = response_message.get("content")
            if content_delta is not None and content_delta != "":
                model_response.content = content_delta

            tool_calls = response_message.get("tool_calls")
            if tool_calls is not None:
                for tool_call in tool_calls:
                    tc = tool_call.get("function")
                    tool_name = tc.get("name")
                    tool_args = tc.get("arguments")
                    function_def = {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if tool_args is not None else None,
                    }
                    model_response.tool_calls.append({"type": "function", "function": function_def})

        if response_delta.get("done"):
            model_response.response_usage = {
                "input_tokens": response_delta.get("prompt_eval_count", 0),
                "output_tokens": response_delta.get("eval_count", 0),
                "total_tokens": response_delta.get("prompt_eval_count", 0) + response_delta.get("eval_count", 0),
                "additional_metrics": {
                    "total_duration": response_delta.get("total_duration", 0),
                    "load_duration": response_delta.get("load_duration", 0),
                    "prompt_eval_duration": response_delta.get("prompt_eval_duration", 0),
                    "eval_duration": response_delta.get("eval_duration", 0),
                },
            }

        return model_response


@dataclass
class ToolCall:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_usage: Optional[Mapping[str, Any]] = None
    response_is_tool_call: bool = field(default=False)
    is_closing_tool_call_tag: bool = field(default=False)
    tool_calls_counter: int = field(default=0)
    tool_call_content: str = field(default="")


@dataclass
class OllamaTools(Ollama):
    """
    An Ollama class that uses XML tags for tool calls.

    For more information, see: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    id: str = "llama3.2"
    name: str = "OllamaTools"
    provider: str = "Ollama"

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: The API kwargs for the model.
        """
        base_params: Dict[str, Any] = {
            "format": self.format,
            "options": self.options,
            "keep_alive": self.keep_alive,
            "request_params": self.request_params,
        }
        request_params: Dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}
        # Add additional request params if provided
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def parse_provider_response(self, response: ChatResponse) -> ModelResponse:
        """
        Parse the provider response.

        Args:
            response (ChatResponse): The response from the provider.

        Returns:
            ModelResponse: The model response.
        """
        model_response = ModelResponse()
        # Get response message
        response_message = response.get("message")

        if response_message.get("role") is not None:
            model_response.role = response_message.get("role")

        content = response_message.get("content")
        if content is not None:
            model_response.content = content
            # Check for tool calls in content
            if "<tool_call>" in content and "</tool_call>" in content:
                if model_response.tool_calls is None:
                    model_response.tool_calls = []

                # Break the response into tool calls
                tool_call_responses = content.split("</tool_call>")
                for tool_call_response in tool_call_responses:
                    # Add back the closing tag if this is not the last tool call
                    if tool_call_response != tool_call_responses[-1]:
                        tool_call_response += "</tool_call>"

                    if "<tool_call>" in tool_call_response and "</tool_call>" in tool_call_response:
                        # Extract tool call string from response
                        tool_call_content = extract_tool_call_from_string(tool_call_response)
                        # Convert the extracted string to a dictionary
                        try:
                            tool_call_dict = json.loads(tool_call_content)
                        except json.JSONDecodeError:
                            raise ValueError(f"Could not parse tool call from: {tool_call_content}")

                        tool_call_name = tool_call_dict.get("name")
                        tool_call_args = tool_call_dict.get("arguments")
                        function_def = {
                            "name": tool_call_name,
                            "arguments": json.dumps(tool_call_args) if tool_call_args is not None else None,
                        }
                        model_response.tool_calls.append({"type": "function", "function": function_def})

        # Get response usage
        if response.get("done"):
            model_response.response_usage = OllamaResponseUsage(
                input_tokens=response.get("prompt_eval_count", 0),
                output_tokens=response.get("eval_count", 0),
                total_duration=response.get("total_duration", 0),
                load_duration=response.get("load_duration", 0),
                prompt_eval_duration=response.get("prompt_eval_duration", 0),
                eval_duration=response.get("eval_duration", 0),
            )
            if model_response.response_usage.input_tokens or model_response.response_usage.output_tokens:
                model_response.response_usage.total_tokens = (
                    model_response.response_usage.input_tokens + model_response.response_usage.output_tokens
                )

        return model_response

    def _create_function_call_result(
        self, fc: FunctionCall, success: bool, output: Optional[Union[List[Any], str]], timer: Timer
    ) -> Message:
        """Create a function call result message."""
        content = (
            "<tool_response>\n"
            + json.dumps({"name": fc.function.name, "content": output if success else fc.error})
            + "\n</tool_response>"
        )

        return Message(
            role=self.tool_message_role,
            content=content,
            tool_call_id=fc.call_id,
            tool_name=fc.function.name,
            tool_args=fc.arguments,
            tool_call_error=not success,
            stop_after_tool_call=fc.function.stop_after_tool_call,
            metrics=MessageMetrics(time=timer.elapsed),
        )

    def format_function_call_results(self, function_call_results: List[Message], messages: List[Message]) -> None:
        """
        Format the function call results and append them to the messages.

        Args:
            function_call_results (List[Message]): The list of function call results.
            messages (List[Message]): The list of messages.
        """
        if len(function_call_results) > 0:
            for _fc_message in function_call_results:
                _fc_message.content = (
                    "<tool_response>\n"
                    + json.dumps({"name": _fc_message.tool_name, "content": _fc_message.content})
                    + "\n</tool_response>"
                )
                messages.append(_fc_message)

    def _prepare_function_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
    ) -> List[FunctionCall]:
        """
        Prepare function calls from tool calls in the assistant message.

        Args:
            assistant_message (Message): The assistant message containing tool calls
            messages (List[Message]): The list of messages to append tool responses to
            model_response (ModelResponse): The model response to update
        Returns:
            List[FunctionCall]: The function calls to run
        """
        if model_response.content is None:
            model_response.content = ""
        if model_response.tool_calls is None:
            model_response.tool_calls = []

        model_response.content = str(remove_tool_calls_from_string(assistant_message.get_content_string()))
        model_response.content += "\n\n"
        function_calls_to_run = self.get_function_calls_to_run(assistant_message, messages)

        return function_calls_to_run

    def process_response_stream(
        self, messages: List[Message], assistant_message: Message, stream_data
    ) -> Iterator[ModelResponse]:
        """
        Process a streaming response from the model.
        """
        tool_call_data = ToolCall()

        for response_delta in self.invoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta, tool_call_data)
            if model_response_delta:
                yield from self._populate_stream_data_and_assistant_message(
                    stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta
                )

    async def aprocess_response_stream(
        self, messages: List[Message], assistant_message: Message, stream_data
    ) -> AsyncIterator[ModelResponse]:
        """
        Process a streaming response from the model.
        """
        tool_call_data = ToolCall()

        async for response_delta in self.ainvoke_stream(messages=messages):
            model_response_delta = self.parse_provider_response_delta(response_delta, tool_call_data)
            if model_response_delta:
                for model_response in self._populate_stream_data_and_assistant_message(
                    stream_data=stream_data, assistant_message=assistant_message, model_response=model_response_delta
                ):
                    yield model_response

    def parse_provider_response_delta(self, response_delta, tool_call_data: ToolCall) -> ModelResponse:
        """
        Parse the provider response delta.

        Args:
            response_delta: The response from the provider.

        Returns:
            Iterator[ModelResponse]: An iterator of the model response.
        """
        model_response = ModelResponse()

        response_message = response_delta.get("message")

        # print(f"Response message: {response_delta}")

        if response_message is not None:
            content_delta = response_message.get("content", "")
            if content_delta is not None and content_delta != "":
                # Append content delta to tool call content
                tool_call_data.tool_call_content += content_delta

            # Log tool call data to help debug tool call processing

            # Detect if response is a tool call
            # If the response is a tool call, it will start a <tool token
            if not tool_call_data.response_is_tool_call and "<tool" in content_delta:
                tool_call_data.response_is_tool_call = True

            # If response is a tool call, count the number of tool calls
            if tool_call_data.response_is_tool_call:
                # If the response is an opening tool call tag, increment the tool call counter
                if "<tool" in content_delta:
                    tool_call_data.tool_calls_counter += 1

                # If the response is a closing tool call tag, decrement the tool call counter
                if tool_call_data.tool_call_content.strip().endswith("</tool_call>"):
                    tool_call_data.tool_calls_counter -= 1

                # If the response is a closing tool call tag and the tool call counter is 0,
                # tool call response is complete
                if tool_call_data.tool_calls_counter == 0 and content_delta.strip().endswith(">"):
                    tool_call_data.response_is_tool_call = False
                    tool_call_data.is_closing_tool_call_tag = True

                    try:
                        model_response.tool_calls = _parse_tool_calls_from_content(tool_call_data.tool_call_content)
                        tool_call_data = ToolCall()
                    except Exception as e:
                        print(e)
                        pass

            # Yield content if not a tool call and content is not None
            if not tool_call_data.response_is_tool_call and content_delta is not None:
                if tool_call_data.is_closing_tool_call_tag and content_delta.strip().endswith(">"):
                    tool_call_data.is_closing_tool_call_tag = False

                model_response.content = content_delta

        if response_delta.get("done"):
            model_response.response_usage = {
                "input_tokens": response_delta.get("prompt_eval_count", 0),
                "output_tokens": response_delta.get("eval_count", 0),
                "total_tokens": response_delta.get("prompt_eval_count", 0) + response_delta.get("eval_count", 0),
                "additional_metrics": {
                    "total_duration": response_delta.get("total_duration", 0),
                    "load_duration": response_delta.get("load_duration", 0),
                    "prompt_eval_duration": response_delta.get("prompt_eval_duration", 0),
                    "eval_duration": response_delta.get("eval_duration", 0),
                },
            }

        return model_response

    def get_instructions_to_generate_tool_calls(self) -> List[str]:
        if self._functions is not None:
            return [
                "At the very first turn you don't have <tool_results> so you shouldn't not make up the results.",
                "To respond to the users message, you can use only one tool at a time.",
                "When using a tool, only respond with the tool call. Nothing else. Do not add any additional notes, explanations or white space.",
                "Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.",
            ]
        return []

    def get_tool_call_prompt(self) -> Optional[str]:
        if self._functions is not None and len(self._functions) > 0:
            tool_call_prompt = dedent(
                """\
            You are a function calling with a language model.
            You are provided with function signatures within <tools></tools> XML tags.
            You may use agentic frameworks for reasoning and planning to help with user query.
            Please call a function and wait for function results to be provided to you in the next iteration.
            Don't make assumptions about what values to plug into functions.
            When you call a function, don't add any additional notes, explanations or white space.
            Once you have called a function, results will be provided to you within <tool_response></tool_response> XML tags.
            Do not make assumptions about tool results if <tool_response> XML tags are not present since the function is not yet executed.
            Analyze the results once you get them and call another function if needed.
            Your final response should directly answer the user query with an analysis or summary of the results of function calls.
            """
            )
            tool_call_prompt += "\nHere are the available tools:"
            tool_call_prompt += "\n<tools>\n"
            tool_definitions: List[str] = []
            for _f_name, _function in self._functions.items():
                _function_def = _function.get_definition_for_prompt()
                if _function_def:
                    tool_definitions.append(_function_def)
            tool_call_prompt += "\n".join(tool_definitions)
            tool_call_prompt += "\n</tools>\n\n"
            tool_call_prompt += dedent(
                """\
            Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
            For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
            <tool_call>
            {"arguments": <args-dict>, "name": <function-name>}
            </tool_call>\n
            """
            )
            return tool_call_prompt
        return None

    def get_system_message_for_model(self) -> Optional[str]:
        return self.get_tool_call_prompt()

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.get_instructions_to_generate_tool_calls()


def _parse_tool_calls_from_content(response_content: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from response content.

    Args:
        response_content (str): The response content containing tool calls

    Returns:
        List[Dict[str, Any]]: List of parsed tool calls

    Raises:
        ValueError: If tool call content cannot be parsed
    """
    tool_calls = []

    if "<tool_call>" in response_content and "</tool_call>" in response_content:
        # Break the response into tool calls
        tool_call_responses = response_content.split("</tool_call>")
        for tool_call_response in tool_call_responses:
            # Add back the closing tag if this is not the last tool call
            if tool_call_response != tool_call_responses[-1]:
                tool_call_response += "</tool_call>"

            if "<tool_call>" in tool_call_response and "</tool_call>" in tool_call_response:
                # Extract tool call string from response
                tool_call_content = extract_tool_call_from_string(tool_call_response)
                # Convert the extracted string to a dictionary
                try:
                    tool_call_dict = json.loads(tool_call_content)
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse tool call from: {tool_call_content}")

                tool_call_name = tool_call_dict.get("name")
                tool_call_args = tool_call_dict.get("arguments")
                function_def = {"name": tool_call_name}
                if tool_call_args is not None:
                    function_def["arguments"] = json.dumps(tool_call_args)
                tool_calls.append(
                    {
                        "type": "function",
                        "function": function_def,
                    }
                )

    return tool_calls
