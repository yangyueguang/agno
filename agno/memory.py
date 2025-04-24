from dataclasses import dataclass, field
from agno.media import AudioArtifact, ImageArtifact, VideoArtifact
from agno.run import TeamRunResponse
from textwrap import dedent
import json
from datetime import datetime
from hashlib import md5
from pydantic import BaseModel, ConfigDict, model_validator, Field, ValidationError
from enum import Enum
from agno.tools import Function
from typing import Any, List, Optional, cast
from agno.models.base import Model
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict
from agno.models.message import Message
from agno.run import RunResponse
from pathlib import Path
from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker
from abc import ABC, abstractmethod
from typing import List, Optional


class AgentRun(BaseModel):
    message: Optional[Message] = None
    messages: Optional[List[Message]] = None
    response: Optional[RunResponse] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "message": self.message.to_dict() if self.message else None,
            "messages": [message.to_dict() for message in self.messages] if self.messages else None,
            "response": self.response.to_dict() if self.response else None,
        }
        return {k: v for k, v in response.items() if v is not None}


class MemoryRow(BaseModel):
    """Memory Row that is stored in the database"""

    memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # id for this memory, auto-generated from the memory
    id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

    def to_dict(self) -> Dict[str, Any]:
        return self.serializable_dict()

    @model_validator(mode="after")
    def generate_id(self) -> "MemoryRow":
        if self.id is None:
            memory_str = json.dumps(self.memory, sort_keys=True)
            cleaned_memory = memory_str.replace(" ", "").replace("\n", "").replace("\t", "")
            self.id = md5(cleaned_memory.encode()).hexdigest()
        return self


class MemoryDb(ABC):
    """Base class for the Memory Database."""

    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_memories(
        self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_table(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def table_exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> bool:
        raise NotImplementedError


class SqliteMemoryDb(MemoryDb):
    def __init__(
        self,
        table_name: str = "memory",
        db_url: Optional[str] = None,
        db_file: Optional[str] = None,
        db_engine: Optional[Engine] = None,
    ):
        """
        This class provides a memory store backed by a SQLite table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url
            3. Use the db_file
            4. Create a new in-memory database

        Args:
            table_name: The name of the table to store Agent sessions.
            db_url: The database URL to connect to.
            db_file: The database file to connect to.
            db_engine: The database engine to use.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            # Use the db_file to create the engine
            db_path = Path(db_file).resolve()
            # Ensure the directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{db_path}")
        else:
            _engine = create_engine("sqlite://")

        if _engine is None:
            raise ValueError("Must provide either db_url, db_file or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()
        self.inspector = inspect(self.db_engine)

        # Database session
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        # Database table for memories
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String),
            Column("memory", String),
            Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
            Column(
                "updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
            ),
            extend_existing=True,
        )

    def create(self) -> None:
        if not self.table_exists():
            try:
                print(f"Creating table: {self.table_name}")
                self.table.create(self.db_engine, checkfirst=True)
            except Exception as e:
                print(f"Error creating table '{self.table_name}': {e}")
                raise

    def memory_exists(self, memory: MemoryRow) -> bool:
        with self.Session() as session:
            stmt = select(self.table.c.id).where(self.table.c.id == memory.id)
            result = session.execute(stmt).first()
            return result is not None

    def read_memories(
        self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        try:
            with self.Session() as session:
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)

                if sort == "asc":
                    stmt = stmt.order_by(self.table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(self.table.c.created_at.desc())

                if limit is not None:
                    stmt = stmt.limit(limit)

                result = session.execute(stmt)
                for row in result:
                    memories.append(MemoryRow(id=row.id, user_id=row.user_id, memory=eval(row.memory)))
        except SQLAlchemyError as e:
            print(f"Exception reading from table: {e}")
            print(f"Table does not exist: {self.table_name}")
            print("Creating table for future transactions")
            self.create()
        return memories

    def upsert_memory(self, memory: MemoryRow, create_and_retry: bool = True) -> None:
        try:
            with self.Session() as session:
                # Check if the memory already exists
                existing = session.execute(select(self.table).where(self.table.c.id == memory.id)).first()

                if existing:
                    # Update existing memory
                    stmt = (
                        self.table.update()
                        .where(self.table.c.id == memory.id)
                        .values(user_id=memory.user_id, memory=str(memory.memory), updated_at=text("CURRENT_TIMESTAMP"))
                    )
                else:
                    # Insert new memory
                    stmt = self.table.insert().values(id=memory.id, user_id=memory.user_id, memory=str(memory.memory))  # type: ignore

                session.execute(stmt)
                session.commit()
        except SQLAlchemyError as e:
            print(f"Exception upserting into table: {e}")
            if not self.table_exists():
                print(f"Table does not exist: {self.table_name}")
                print("Creating table for future transactions")
                self.create()
                if create_and_retry:
                    return self.upsert_memory(memory, create_and_retry=False)
            else:
                raise

    def delete_memory(self, id: str) -> None:
        with self.Session() as session:
            stmt = delete(self.table).where(self.table.c.id == id)
            session.execute(stmt)
            session.commit()

    def drop_table(self) -> None:
        if self.table_exists():
            print(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

    def table_exists(self) -> bool:
        print(f"Checking if table exists: {self.table.name}")
        try:
            return self.inspector.has_table(self.table.name)
        except Exception as e:
            print(e)
            return False

    def clear(self) -> bool:
        with self.Session() as session:
            stmt = delete(self.table)
            session.execute(stmt)
            session.commit()
        return True

    def __del__(self):
        # self.Session.remove()
        pass


class MemoryManager(BaseModel):
    model: Optional[Model] = None
    user_id: Optional[str] = None
    limit: Optional[int] = None
    # Provide the system prompt for the manager as a string
    system_prompt: Optional[str] = None
    # Memory Database
    db: Optional[MemoryDb] = None

    # Do not set the input message here, it will be set by the run method
    input_message: Optional[str] = None
    _tools_for_model: Optional[List[Dict]] = None
    _functions_for_model: Optional[Dict[str, Function]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_model(self) -> None:
        # Use the default Model (OpenAIChat) if no model is provided
        if self.model is None:
            try:
                from agno.models.ollama import Ollama
            except ModuleNotFoundError as e:
                print(e)
                print(
                    "Agno uses `openai` as the default model provider. Please provide a `model` or install `openai`."
                )
                exit(1)
            self.model = Ollama()

        # Add tools to the Model
        self.add_tools_to_model(model=self.model)

    def add_tools_to_model(self, model: Model) -> None:
        if self._tools_for_model is None:
            self._tools_for_model = []
        if self._functions_for_model is None:
            self._functions_for_model = {}

        for tool in [
            self.add_memory,
            self.update_memory,
            self.delete_memory,
            self.clear_memory,
        ]:
            try:
                function_name = tool.__name__
                if function_name not in self._functions_for_model:
                    func = Function.from_callable(tool)  # type: ignore
                    self._functions_for_model[func.name] = func
                    self._tools_for_model.append({"type": "function", "function": func.to_dict()})
                    print(f"Included function {func.name}")
            except Exception as e:
                print(f"Could not add function {tool}: {e}")
        # Set tools on the model
        model.set_tools(tools=self._tools_for_model)
        # Set functions on the model
        model.set_functions(functions=self._functions_for_model)

    def get_existing_memories(self) -> Optional[List[MemoryRow]]:
        if self.db is None:
            return None

        return self.db.read_memories(user_id=self.user_id, limit=self.limit)

    def add_memory(self, memory: str) -> str:
        """Use this function to add a memory to the database.
        Args:
            memory (str): The memory to be stored.
        Returns:
            str: A message indicating if the memory was added successfully or not.
        """
        try:
            if self.db:
                self.db.upsert_memory(
                    MemoryRow(user_id=self.user_id, memory=Memory(memory=memory, input=self.input_message).to_dict())
                )
            return "Memory added successfully"
        except Exception as e:
            print(f"Error storing memory in db: {e}")
            return f"Error adding memory: {e}"

    def delete_memory(self, id: str) -> str:
        """Use this function to delete a memory from the database.
        Args:
            id (str): The id of the memory to be deleted.
        Returns:
            str: A message indicating if the memory was deleted successfully or not.
        """
        try:
            if self.db:
                self.db.delete_memory(id=id)
            return "Memory deleted successfully"
        except Exception as e:
            print(f"Error deleting memory in db: {e}")
            return f"Error deleting memory: {e}"

    def update_memory(self, id: str, memory: str) -> str:
        """Use this function to update a memory in the database.
        Args:
            id (str): The id of the memory to be updated.
            memory (str): The updated memory.
        Returns:
            str: A message indicating if the memory was updated successfully or not.
        """
        try:
            if self.db:
                self.db.upsert_memory(
                    MemoryRow(
                        id=id, user_id=self.user_id, memory=Memory(memory=memory, input=self.input_message).to_dict()
                    )
                )
            return "Memory updated successfully"
        except Exception as e:
            print(f"Error updating memory in db: {e}")
            return f"Error updating memory: {e}"

    def clear_memory(self) -> str:
        """Use this function to clear all memories from the database.

        Returns:
            str: A message indicating if the memory was cleared successfully or not.
        """
        try:
            if self.db:
                self.db.clear()
            return "Memory cleared successfully"
        except Exception as e:
            print(f"Error clearing memory in db: {e}")
            return f"Error clearing memory: {e}"

    def get_system_message(self) -> Message:
        # -*- Return a system message for the memory manager
        system_prompt_lines = [
            "Your task is to generate a concise memory for the user's message. "
            "Create a memory that captures the key information provided by the user, as if you were storing it for future reference. "
            "The memory should be a brief, third-person statement that encapsulates the most important aspect of the user's input, without adding any extraneous details. "
            "This memory will be used to enhance the user's experience in subsequent conversations.",
            "You will also be provided with a list of existing memories. You may:",
            "  1. Add a new memory using the `add_memory` tool.",
            "  2. Update a memory using the `update_memory` tool.",
            "  3. Delete a memory using the `delete_memory` tool.",
            "  4. Clear all memories using the `clear_memory` tool. Use this with extreme caution, as it will remove all memories from the database.",
        ]
        existing_memories = self.get_existing_memories()
        if existing_memories and len(existing_memories) > 0:
            system_prompt_lines.extend(
                [
                    "\nExisting memories:",
                    "<existing_memories>\n"
                    + "\n".join([f"  - id: {m.id} | memory: {m.memory}" for m in existing_memories])
                    + "\n</existing_memories>",
                ]
            )
        return Message(role="system", content="\n".join(system_prompt_lines))

    def run(
        self,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        print("*********** MemoryManager Start ***********")

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message()]

        # Add the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]

        # Set input message added with the memory
        self.input_message = message

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print("*********** MemoryManager End ***********")
        return response.content

    async def arun(
        self,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        print("*********** Async MemoryManager Start ***********")

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message()]
        # Add the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]

        # Set input message added with the memory
        self.input_message = message

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print("*********** Async MemoryManager End ***********")
        return response.content


class MemoryRetrieval(str, Enum):
    last_n = "last_n"
    first_n = "first_n"
    semantic = "semantic"


class Memory(BaseModel):
    """Model for Agent Memories"""

    memory: str
    id: Optional[str] = None
    topic: Optional[str] = None
    input: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class MemoryClassifier(BaseModel):
    model: Optional[Model] = None

    # Provide the system prompt for the classifier as a string
    system_prompt: Optional[str] = None
    # Existing Memories
    existing_memories: Optional[List[Memory]] = None

    def update_model(self) -> None:
        if self.model is None:
            try:
                from agno.models.ollama import Ollama
            except ModuleNotFoundError as e:
                print(e)
                print(
                    "Agno uses `openai` as the default model provider. Please provide a `model` or install `openai`."
                )
                exit(1)
            self.model = Ollama()

    def get_system_message(self) -> Message:
        # -*- Return a system message for classification
        system_prompt_lines = [
            "Your task is to identify if the user's message contains information that is worth remembering for future conversations.",
            "This includes details that could personalize ongoing interactions with the user, such as:\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - Significant life events or experiences shared by the user\n"
            "  - Important context about the user's current situation, challenges or goals\n"
            "  - What the user likes or dislikes, their opinions, beliefs, values, etc.\n"
            "  - Any other details that provide valuable insights into the user's personality, perspective or needs",
            "Your task is to decide whether the user input contains any of the above information worth remembering.",
            "If the user input contains any information worth remembering for future conversations, respond with 'yes'.",
            "If the input does not contain any important details worth saving, respond with 'no' to disregard it.",
            "You will also be provided with a list of existing memories to help you decide if the input is new or already known.",
            "If the memory already exists that matches the input, respond with 'no' to keep it as is.",
            "If a memory exists that needs to be updated or deleted, respond with 'yes' to update/delete it.",
            "You must only respond with 'yes' or 'no'. Nothing else will be considered as a valid response.",
        ]
        if self.existing_memories and len(self.existing_memories) > 0:
            system_prompt_lines.extend(
                [
                    "\nExisting memories:",
                    "<existing_memories>\n"
                    + "\n".join([f"  - {m.memory}" for m in self.existing_memories])
                    + "\n</existing_memories>",
                ]
            )
        return Message(role="system", content="\n".join(system_prompt_lines))

    def run(
        self,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        print("*********** MemoryClassifier Start ***********")

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message()]
        # Add the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print("*********** MemoryClassifier End ***********")
        return response.content

    async def arun(
        self,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        print("*********** Async MemoryClassifier Start ***********")

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message()]
        # Add the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print("*********** Async MemoryClassifier End ***********")
        return response.content


class SessionSummary(BaseModel):
    """Model for Session Summary."""

    summary: str = Field(
        ...,
        description="Summary of the session. Be concise and focus on only important information. Do not make anything up.",
    )
    topics: Optional[List[str]] = Field(None, description="Topics discussed in the session.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


class MemorySummarizer(BaseModel):
    model: Optional[Model] = None
    use_structured_outputs: bool = False

    def update_model(self) -> None:
        if self.model is None:
            try:
                from agno.models.ollama import Ollama
            except ModuleNotFoundError as e:
                print(e)
                print(
                    "Agno uses `openai` as the default model provider. Please provide a `model` or install `openai`."
                )
                exit(1)
            self.model = Ollama()

        # Set response_format if it is not set on the Model
        if self.use_structured_outputs:
            self.model.response_format = SessionSummary
            self.model.structured_outputs = True
        else:
            self.model.response_format = {"type": "json_object"}

    def get_system_message(self, messages_for_summarization: List[Dict[str, str]]) -> Message:
        # -*- Return a system message for summarization
        system_prompt = dedent("""\
        Analyze the following conversation between a user and an assistant, and extract the following details:
          - Summary (str): Provide a concise summary of the session, focusing on important information that would be helpful for future interactions.
          - Topics (Optional[List[str]]): List the topics discussed in the session.
        Please ignore any frivolous information.

        Conversation:
        """)
        conversation = []
        for message_pair in messages_for_summarization:
            conversation.append(f"User: {message_pair['user']}")
            if "assistant" in message_pair:
                conversation.append(f"Assistant: {message_pair['assistant']}")
            elif "model" in message_pair:
                conversation.append(f"Assistant: {message_pair['model']}")

        system_prompt += "\n".join(conversation)

        if not self.use_structured_outputs:
            system_prompt += "\n\nProvide your output as a JSON containing the following fields:"
            json_schema = SessionSummary.model_json_schema()
            response_model_properties = {}
            json_schema_properties = json_schema.get("properties")
            if json_schema_properties is not None:
                for field_name, field_properties in json_schema_properties.items():
                    formatted_field_properties = {
                        prop_name: prop_value
                        for prop_name, prop_value in field_properties.items()
                        if prop_name != "title"
                    }
                    response_model_properties[field_name] = formatted_field_properties

            if len(response_model_properties) > 0:
                system_prompt += "\n<json_fields>"
                system_prompt += f"\n{json.dumps([key for key in response_model_properties.keys() if key != '$defs'])}"
                system_prompt += "\n</json_fields>"
                system_prompt += "\nHere are the properties for each field:"
                system_prompt += "\n<json_field_properties>"
                system_prompt += f"\n{json.dumps(response_model_properties, indent=2)}"
                system_prompt += "\n</json_field_properties>"

            system_prompt += "\nStart your response with `{` and end it with `}`."
            system_prompt += "\nYour output will be passed to json.loads() to convert it to a Python object."
            system_prompt += "\nMake sure it only contains valid JSON."
        return Message(role="system", content=system_prompt)

    def run(
        self,
        message_pairs: List[Tuple[Message, Message]],
        **kwargs: Any,
    ) -> Optional[SessionSummary]:
        print("*********** MemorySummarizer Start ***********")

        if message_pairs is None or len(message_pairs) == 0:
            print("No message pairs provided for summarization.")
            return None

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Convert the message pairs to a list of dictionaries
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append(
                {
                    user_message.role: user_message.get_content_string(),
                    assistant_message.role: assistant_message.get_content_string(),
                }
            )

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [
            self.get_system_message(messages_for_summarization),
            # For models that require a non-system message
            Message(role="user", content="Provide the summary of the conversation."),
        ]

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print("*********** MemorySummarizer End ***********")

        # If the model natively supports structured outputs, the parsed value is already in the structured format
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed

        # Otherwise convert the response to the structured format
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    # Check if response starts with ```json
                    if response.content.startswith("```json"):
                        response.content = response.content.replace("```json\n", "").replace("\n```", "")
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            print(f"Failed to validate session_summary response: {exc}")
                return session_summary
            except Exception as e:
                print(f"Failed to convert response to session_summary: {e}")
        return None

    async def arun(
        self,
        message_pairs: List[Tuple[Message, Message]],
        **kwargs: Any,
    ) -> Optional[SessionSummary]:
        print("*********** Async MemorySummarizer Start ***********")

        if message_pairs is None or len(message_pairs) == 0:
            print("No message pairs provided for summarization.")
            return None

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Convert the message pairs to a list of dictionaries
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append(
                {
                    user_message.role: user_message.get_content_string(),
                    assistant_message.role: assistant_message.get_content_string(),
                }
            )

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [
            self.get_system_message(messages_for_summarization),
            # For models that require a non-system message
            Message(role="user", content="Provide the summary of the conversation."),
        ]

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print("*********** Async MemorySummarizer End ***********")

        # If the model natively supports structured outputs, the parsed value is already in the structured format
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed

        # Otherwise convert the response to the structured format
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    # Check if response starts with ```json
                    if response.content.startswith("```json"):
                        response.content = response.content.replace("```json\n", "").replace("\n```", "")
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            print(f"Failed to validate session_summary response: {exc}")
                return session_summary
            except Exception as e:
                print(f"Failed to convert response to session_summary: {e}")
        return None


class AgentMemory(BaseModel):
    # Runs between the user and agent
    runs: List[AgentRun] = []
    # List of messages sent to the model
    messages: List[Message] = []
    update_system_message_on_change: bool = False

    # Summary of the session
    summary: Optional[SessionSummary] = None
    # Create and store session summaries
    create_session_summary: bool = False
    # Update session summaries after each run
    update_session_summary_after_run: bool = True
    # Summarizer to generate session summaries
    summarizer: Optional[MemorySummarizer] = None

    # Create and store personalized memories for this user
    create_user_memories: bool = False
    # Update memories for the user after each run
    update_user_memories_after_run: bool = True

    # MemoryDb to store personalized memories
    db: Optional[MemoryDb] = None
    # User ID for the personalized memories
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    num_memories: Optional[int] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None

    # True when memory is being updated
    updating_memory: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(
            exclude_none=True,
            include={
                "update_system_message_on_change",
                "create_session_summary",
                "update_session_summary_after_run",
                "create_user_memories",
                "update_user_memories_after_run",
                "user_id",
                "num_memories",
            },
        )
        # Add summary if it exists
        if self.summary is not None:
            _memory_dict["summary"] = self.summary.to_dict()
        # Add memories if they exist
        if self.memories is not None:
            _memory_dict["memories"] = [memory.to_dict() for memory in self.memories]
        # Add messages if they exist
        if self.messages is not None:
            _memory_dict["messages"] = [message.to_dict() for message in self.messages]
        # Add runs if they exist
        if self.runs is not None:
            _memory_dict["runs"] = [run.to_dict() for run in self.runs]
        return _memory_dict

    def add_run(self, agent_run: AgentRun) -> None:
        """Adds an AgentRun to the runs list."""
        self.runs.append(agent_run)
        print("Added AgentRun to AgentMemory")

    def add_system_message(self, message: Message, system_message_role: str = "system") -> None:
        """Add the system messages to the messages list"""
        # If this is the first run in the session, add the system message to the messages list
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        # If there are messages in the memory, check if the system message is already in the memory
        # If it is not, add the system message to the messages list
        # If it is, update the system message if content has changed and update_system_message_on_change is True
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            # Update the system message in memory if content has changed
            if system_message_index is not None:
                if (
                    self.messages[system_message_index].content != message.content
                    and self.update_system_message_on_change
                ):
                    print("Updating system message in memory with new content")
                    self.messages[system_message_index] = message
            else:
                # Add the system message to the messages list
                self.messages.insert(0, message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add a list of messages to the messages list."""
        self.messages.extend(messages)
        print(f"Added {len(messages)} Messages to AgentMemory")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the messages list as a list of dictionaries."""
        return [message.model_dump() for message in self.messages]

    def get_messages_from_last_n_runs(
        self, last_n: Optional[int] = None, skip_role: Optional[str] = None
    ) -> List[Message]:
        """Returns the messages from the last_n runs, excluding previously tagged history messages.

        Args:
            last_n: The number of runs to return from the end of the conversation.
            skip_role: Skip messages with this role.

        Returns:
            A list of Messages from the specified runs, excluding history messages.
        """
        if not self.runs:
            return []

        runs_to_process = self.runs if last_n is None else self.runs[-last_n:]
        messages_from_history = []

        for run in runs_to_process:
            if not (run.response and run.response.messages):
                continue

            for message in run.response.messages:
                # Skip messages with specified role
                if skip_role and message.role == skip_role:
                    continue
                # Skip messages that were tagged as history in previous runs
                if hasattr(message, "from_history") and message.from_history:
                    continue

                messages_from_history.append(message)

        print(f"Getting messages from previous runs: {len(messages_from_history)}")
        return messages_from_history

    def get_message_pairs(
        self, user_role: str = "user", assistant_role: Optional[List[str]] = None
    ) -> List[Tuple[Message, Message]]:
        """Returns a list of tuples of (user message, assistant response)."""

        if assistant_role is None:
            assistant_role = ["assistant", "model", "CHATBOT"]

        runs_as_message_pairs: List[Tuple[Message, Message]] = []
        for run in self.runs:
            if run.response and run.response.messages:
                user_messages_from_run = None
                assistant_messages_from_run = None

                # Start from the beginning to look for the user message
                for message in run.response.messages:
                    if hasattr(message, "from_history") and message.from_history:
                        continue
                    if message.role == user_role:
                        user_messages_from_run = message
                        break

                # Start from the end to look for the assistant response
                for message in run.response.messages[::-1]:
                    if hasattr(message, "from_history") and message.from_history:
                        continue
                    if message.role in assistant_role:
                        assistant_messages_from_run = message
                        break

                if user_messages_from_run and assistant_messages_from_run:
                    runs_as_message_pairs.append((user_messages_from_run, assistant_messages_from_run))
        return runs_as_message_pairs

    def get_tool_calls(self, num_calls: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns a list of tool calls from the messages"""

        tool_calls = []
        for message in self.messages[::-1]:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(tool_call)
                    if num_calls and len(tool_calls) >= num_calls:
                        return tool_calls
        return tool_calls

    def load_user_memories(self) -> None:
        """Load memories from memory db for this user."""

        if self.db is None:
            return

        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="asc" if self.retrieval == MemoryRetrieval.first_n else "desc",
                )
            else:
                raise NotImplementedError("Semantic retrieval not yet supported.")
        except Exception as e:
            print(f"Error reading memory: {e}")
            return

        # Clear the existing memories
        self.memories = []

        # No memories to load
        if memory_rows is None or len(memory_rows) == 0:
            return

        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                print(f"Error loading memory: {e}")
                continue

    def should_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input)
        if classifier_response == "yes":
            return True
        return False

    async def ashould_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = await self.classifier.arun(input)
        if classifier_response == "yes":
            return True
        return False

    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            print("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or self.should_update_memory(input=input)
        print(f"Update memory: {should_update_memory}")

        if not should_update_memory:
            print("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = self.manager.run(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    async def aupdate_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            print("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or await self.ashould_update_memory(input=input)
        print(f"Async update memory: {should_update_memory}")

        if not should_update_memory:
            print("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = await self.manager.arun(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    def update_summary(self) -> Optional[SessionSummary]:
        """Creates a summary of the session"""

        self.updating_memory = True

        if self.summarizer is None:
            self.summarizer = MemorySummarizer()

        self.summary = self.summarizer.run(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    async def aupdate_summary(self) -> Optional[SessionSummary]:
        """Creates a summary of the session"""

        self.updating_memory = True

        if self.summarizer is None:
            self.summarizer = MemorySummarizer()

        self.summary = await self.summarizer.arun(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    def clear(self) -> None:
        """Clear the AgentMemory"""

        self.runs = []
        self.messages = []
        self.summary = None
        self.memories = None

    def deep_copy(self) -> "AgentMemory":
        from copy import deepcopy

        # Create a shallow copy of the object
        copied_obj = self.__class__(**self.to_dict())

        # Manually deepcopy fields that are known to be safe
        for field_name, field_value in self.__dict__.items():
            if field_name not in ["db", "classifier", "manager", "summarizer"]:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    print(f"Failed to deepcopy field: {field_name} - {e}")
                    setattr(copied_obj, field_name, field_value)

        copied_obj.db = self.db
        copied_obj.classifier = self.classifier
        copied_obj.manager = self.manager
        copied_obj.summarizer = self.summarizer

        return copied_obj


@dataclass
class TeamRun:
    message: Optional[Message] = None
    member_runs: Optional[List[AgentRun]] = None
    response: Optional[TeamRunResponse] = None

    def to_dict(self) -> Dict[str, Any]:
        message = self.message.to_dict() if self.message else None
        member_responses = [run.to_dict() for run in self.member_runs] if self.member_runs else None
        response = self.response.to_dict() if self.response else None
        return {
            "message": message,
            "member_responses": member_responses,
            "response": response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamRun":
        message = Message.model_validate(data.get("message")) if data.get("message") else None
        member_runs = (
            [AgentRun.model_validate(run) for run in data.get("member_runs", [])] if data.get("member_runs") else None
        )
        response = TeamRunResponse.from_dict(data.get("response", {})) if data.get("response") else None
        return cls(message=message, member_runs=member_runs, response=response)


@dataclass
class TeamMemberInteraction:
    member_name: str
    task: str
    response: RunResponse


@dataclass
class TeamContext:
    # List of team member interaction, represented as a request and a response
    member_interactions: List[TeamMemberInteraction] = field(default_factory=list)
    text: Optional[str] = None


@dataclass
class TeamMemory:
    # Runs between the user and agent
    runs: List[TeamRun] = field(default_factory=list)
    # List of messages sent to the model
    messages: List[Message] = field(default_factory=list)
    # If True, update the system message when it changes
    update_system_message_on_change: bool = True

    team_context: Optional[TeamContext] = None

    # Create and store personalized memories for this user
    create_user_memories: bool = False
    # Update memories for the user after each run
    update_user_memories_after_run: bool = True

    # MemoryDb to store personalized memories
    db: Optional[MemoryDb] = None
    # User ID for the personalized memories
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None

    num_memories: Optional[int] = None

    # True when memory is being updated
    updating_memory: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = {}
        for key, value in self.__dict__.items():
            if value is not None and key in [
                "update_system_message_on_change",
                "create_user_memories",
                "update_user_memories_after_run",
                "user_id",
                "num_memories",
            ]:
                _memory_dict[key] = value

        # Add messages if they exist
        if self.messages is not None:
            _memory_dict["messages"] = [message.to_dict() for message in self.messages]
        # Add memories if they exist
        if self.memories is not None:
            _memory_dict["memories"] = [memory.to_dict() for memory in self.memories]
        # Add runs if they exist
        if self.runs is not None:
            _memory_dict["runs"] = [run.to_dict() for run in self.runs]
        return _memory_dict

    def add_interaction_to_team_context(self, member_name: str, task: str, run_response: RunResponse) -> None:
        if self.team_context is None:
            self.team_context = TeamContext()
        self.team_context.member_interactions.append(
            TeamMemberInteraction(
                member_name=member_name,
                task=task,
                response=run_response,
            )
        )
        print(f"Updated team context with member name: {member_name}")

    def set_team_context_text(self, text: str) -> None:
        if self.team_context:
            self.team_context.text = text
        else:
            self.team_context = TeamContext(text=text)

    def get_team_context_str(self) -> str:
        if self.team_context and self.team_context.text:
            return f"<team context>\n{self.team_context.text}\n</team context>\n"
        return ""

    def get_team_member_interactions_str(self) -> str:
        team_member_interactions_str = ""
        if self.team_context and self.team_context.member_interactions:
            team_member_interactions_str += "<member interactions>\n"

            for interaction in self.team_context.member_interactions:
                team_member_interactions_str += f"Member: {interaction.member_name}\n"
                team_member_interactions_str += f"Task: {interaction.task}\n"
                team_member_interactions_str += f"Response: {interaction.response.to_dict().get('content', '')}\n"
                team_member_interactions_str += "\n"
            team_member_interactions_str += "</member interactions>\n"
        return team_member_interactions_str

    def get_team_context_images(self) -> List[ImageArtifact]:
        images = []
        if self.team_context and self.team_context.member_interactions:
            for interaction in self.team_context.member_interactions:
                if interaction.response.images:
                    images.extend(interaction.response.images)
        return images

    def get_team_context_videos(self) -> List[VideoArtifact]:
        videos = []
        if self.team_context and self.team_context.member_interactions:
            for interaction in self.team_context.member_interactions:
                if interaction.response.videos:
                    videos.extend(interaction.response.videos)
        return videos

    def get_team_context_audio(self) -> List[AudioArtifact]:
        audio = []
        if self.team_context and self.team_context.member_interactions:
            for interaction in self.team_context.member_interactions:
                if interaction.response.audio:
                    audio.extend(interaction.response.audio)
        return audio

    def add_team_run(self, team_run: TeamRun) -> None:
        """Adds an TeamRun to the runs list."""
        self.runs.append(team_run)
        print("Added TeamRun to TeamMemory")

    def add_system_message(self, message: Message, system_message_role: str = "system") -> None:
        """Add the system messages to the messages list"""
        # If this is the first run in the session, add the system message to the messages list
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        # If there are messages in the memory, check if the system message is already in the memory
        # If it is not, add the system message to the messages list
        # If it is, update the system message if content has changed and update_system_message_on_change is True
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            # Update the system message in memory if content has changed
            if system_message_index is not None:
                if (
                    self.messages[system_message_index].content != message.content
                    and self.update_system_message_on_change
                ):
                    print("Updating system message in memory with new content")
                    self.messages[system_message_index] = message
            else:
                # Add the system message to the messages list
                self.messages.insert(0, message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add a list of messages to the messages list."""
        self.messages.extend(messages)
        print(f"Added {len(messages)} Messages to TeamMemory")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the messages list as a list of dictionaries."""
        return [message.model_dump() for message in self.messages]

    def get_messages_from_last_n_runs(
        self, last_n: Optional[int] = None, skip_role: Optional[str] = None
    ) -> List[Message]:
        """Returns the messages from the last_n runs, excluding previously tagged history messages.

        Args:
            last_n: The number of runs to return from the end of the conversation.
            skip_role: Skip messages with this role.

        Returns:
            A list of Messages from the specified runs, excluding history messages.
        """
        if not self.runs:
            return []

        runs_to_process = self.runs if last_n is None else self.runs[-last_n:]
        messages_from_history = []

        for run in runs_to_process:
            if not (run.response and run.response.messages):
                continue

            for message in run.response.messages:
                # Skip messages with specified role
                if skip_role and message.role == skip_role:
                    continue
                # Skip messages that were tagged as history in previous runs
                if hasattr(message, "from_history") and message.from_history:
                    continue

                messages_from_history.append(message)

        print(f"Getting messages from previous runs: {len(messages_from_history)}")
        return messages_from_history

    def get_all_messages(self) -> List[Tuple[Message, Message]]:
        """Returns a list of tuples of (user message, assistant response)."""

        assistant_role = ["assistant", "model", "CHATBOT"]

        runs_as_message_pairs: List[Tuple[Message, Message]] = []
        for run in self.runs:
            if run.response and run.response.messages:
                user_message_from_run = None
                assistant_message_from_run = None

                # Start from the beginning to look for the user message
                for message in run.response.messages:
                    if message.role == "user":
                        user_message_from_run = message
                        break

                # Start from the end to look for the assistant response
                for message in run.response.messages[::-1]:
                    if message.role in assistant_role:
                        assistant_message_from_run = message
                        break

                if user_message_from_run and assistant_message_from_run:
                    runs_as_message_pairs.append((user_message_from_run, assistant_message_from_run))
        return runs_as_message_pairs

    def load_user_memories(self) -> None:
        """Load memories from memory db for this user."""

        if self.db is None:
            return

        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="asc" if self.retrieval == MemoryRetrieval.first_n else "desc",
                )
            else:
                raise NotImplementedError("Semantic retrieval not yet supported.")
        except Exception as e:
            print(f"Error reading memory: {e}")
            return

        # Clear the existing memories
        self.memories = []

        # No memories to load
        if memory_rows is None or len(memory_rows) == 0:
            return

        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                print(f"Error loading memory: {e}")
                continue

    def should_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input)
        if classifier_response == "yes":
            return True
        return False

    async def ashould_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = await self.classifier.arun(input)
        if classifier_response == "yes":
            return True
        return False

    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            print("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or self.should_update_memory(input=input)
        print(f"Update memory: {should_update_memory}")

        if not should_update_memory:
            print("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = self.manager.run(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    async def aupdate_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""
        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            print("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or await self.ashould_update_memory(input=input)
        print(f"Async update memory: {should_update_memory}")

        if not should_update_memory:
            print("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = await self.manager.arun(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    def deep_copy(self) -> "TeamMemory":
        from copy import deepcopy

        # Create a shallow copy of the object
        copied_obj = self.__class__(**self.to_dict())

        # Manually deepcopy fields that are known to be safe
        for field_name, field_value in self.__dict__.items():
            if field_name not in ["db", "classifier", "manager"]:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    print(f"Failed to deepcopy field: {field_name} - {e}")
                    setattr(copied_obj, field_name, field_value)

        copied_obj.db = self.db
        copied_obj.classifier = self.classifier
        copied_obj.manager = self.manager

        return copied_obj


class WorkflowRun(BaseModel):
    input: Optional[Dict[str, Any]] = None
    response: Optional[RunResponse] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowMemory(BaseModel):
    runs: List[WorkflowRun] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def add_run(self, workflow_run: WorkflowRun) -> None:
        """Adds a WorkflowRun to the runs list."""
        self.runs.append(workflow_run)
        print("Added WorkflowRun to WorkflowMemory")

    def clear(self) -> None:
        """Clear the WorkflowMemory"""

        self.runs = []

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> "WorkflowMemory":
        new_memory = self.model_copy(deep=True, update=update)
        # clear the new memory to remove any references to the old memory
        new_memory.clear()
        return new_memory
