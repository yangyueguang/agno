from typing import List, Optional, Union

from agno.models.message import Message


class AgentRunException(Exception):
    def __init__(
        self,
        exc,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
        stop_execution: bool = False,
    ):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution


class RunCancelledException(Exception):
    """Exception raised when a run is cancelled."""

    def __init__(self, message: str = "Operation cancelled by user"):
        super().__init__(message)


class ModelProviderError(Exception):
    """Exception raised when an internal error occurs."""

    def __init__(
        self, message: str, status_code: int = 502, model_name: Optional[str] = None, model_id: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return str(self.message)



