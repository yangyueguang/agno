import json
from textwrap import dedent
from datetime import datetime
from hashlib import md5
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from enum import Enum
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, cast, Union
from pathlib import Path
from sqlalchemy import Column, DateTime, Engine, MetaData, String, Table, create_engine, delete, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker
from abc import ABC, abstractmethod
from time import time
from agno.models import Ollama, Model, AudioArtifact, AudioResponse, ImageArtifact, VideoArtifact, Citations, Message, MessageReferences, Function


class RunMessages:
    def __init__(self, messages: List[Message] = None, system_message: Message = None, user_message: Message = None, extra_messages: List[Message] = None):
        self.messages = messages or []
        self.system_message = system_message
        self.user_message = user_message
        self.extra_messages = extra_messages or []

    def get_input_messages(self) -> List[Message]:
        input_messages = []
        if self.system_message is not None:
            input_messages.append(self.system_message)
        if self.user_message is not None:
            input_messages.append(self.user_message)
        if self.extra_messages is not None:
            input_messages.extend(self.extra_messages)
        return input_messages


class NextAction(str, Enum):
    CONTINUE = 'continue'
    VALIDATE = 'validate'
    FINAL_ANSWER = 'final_answer'
    RESET = 'reset'


class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description='A concise title summarizing the step"s purpose')
    action: Optional[str] = Field(None, description='The action derived from this step. Talk in first person like I will ... ')
    result: Optional[str] = Field(None, description='The result of executing the action. Talk in first person like I did this and got ... ')
    reasoning: Optional[str] = Field(None, description='The thought process and considerations behind this step')
    next_action: Optional[NextAction] = Field(None, description='Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer')
    confidence: Optional[float] = Field(None, description='Confidence score for this step (0.0 to 1.0)')


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description='A list of reasoning steps')


class RunEvent(str, Enum):
    run_started = 'RunStarted'
    run_response = 'RunResponse'
    run_completed = 'RunCompleted'
    run_error = 'RunError'
    run_cancelled = 'RunCancelled'
    tool_call_started = 'ToolCallStarted'
    tool_call_completed = 'ToolCallCompleted'
    reasoning_started = 'ReasoningStarted'
    reasoning_step = 'ReasoningStep'
    reasoning_completed = 'ReasoningCompleted'
    updating_memory = 'UpdatingMemory'
    workflow_started = 'WorkflowStarted'
    workflow_completed = 'WorkflowCompleted'


class RunResponseExtraData:
    def __init__(self, references: Optional[List[MessageReferences]] = None, add_messages: Optional[List[Message]] = None, reasoning_steps: Optional[List['ReasoningStep']] = None, reasoning_messages: Optional[List[Message]] = None):
        self.references = references
        self.add_messages = add_messages
        self.reasoning_steps = reasoning_steps
        self.reasoning_messages = reasoning_messages

    def to_dict(self) -> Dict[str, Any]:
        _dict = {}
        if self.add_messages is not None:
            _dict['add_messages'] = [m.to_dict() for m in self.add_messages]
        if self.reasoning_messages is not None:
            _dict['reasoning_messages'] = [m.to_dict() for m in self.reasoning_messages]
        if self.reasoning_steps is not None:
            _dict['reasoning_steps'] = [rs.model_dump() for rs in self.reasoning_steps]
        if self.references is not None:
            _dict['references'] = [r.model_dump() for r in self.references]
        return _dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResponseExtraData':
        add_messages = data.pop('add_messages', None)
        add_messages = [Message.model_validate(message) for message in add_messages] if add_messages else None
        history = data.pop('history', None)
        history = [Message.model_validate(message) for message in history] if history else None
        reasoning_steps = data.pop('reasoning_steps', None)
        reasoning_steps = [ReasoningStep.model_validate(step) for step in reasoning_steps] if reasoning_steps else None
        reasoning_messages = data.pop('reasoning_messages', None)
        reasoning_messages = ([Message.model_validate(message) for message in reasoning_messages] if reasoning_messages else None)
        references = data.pop('references', None)
        references = [MessageReferences.model_validate(reference) for reference in references] if references else None
        return cls(add_messages=add_messages, reasoning_steps=reasoning_steps, reasoning_messages=reasoning_messages, references=references)


class RunResponse:
    def __init__(self, content=None, content_type: str = 'str', thinking: str = None, event: str = RunEvent.run_response.value,
            messages: List[Message] = None, metrics: Dict[str, Any] = None,
            model: str = None, run_id: str = None, agent_id: str = None, session_id: str = None,
            workflow_id: str = None, tools: Optional[List[Dict[str, Any]]] = None,
            formatted_tool_calls: List[str] = None, images: List[ImageArtifact] = None,
            videos: List[VideoArtifact] = None, audio: List[AudioArtifact] = None,
            response_audio: AudioResponse = None, citations: Citations = None,
            extra_data: RunResponseExtraData = None, created_at: int = 0):
        self.content = content
        self.content_type = content_type
        self.thinking = thinking
        self.event = event
        self.messages = messages
        self.metrics = metrics
        self.model = model
        self.run_id = run_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.workflow_id = workflow_id
        self.tools = tools
        self.formatted_tool_calls = formatted_tool_calls
        self.images = images
        self.videos = videos
        self.audio = audio
        self.response_audio = response_audio
        self.citations = citations
        self.extra_data = extra_data
        self.created_at = created_at or int(time())

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in self.__dict__.items()
            if v is not None and k not in ['messages', 'extra_data', 'images', 'videos', 'audio', 'response_audio']}
        if self.messages is not None:
            _dict['messages'] = [m.to_dict() for m in self.messages]
        if self.extra_data is not None:
            _dict['extra_data'] = (self.extra_data.to_dict() if isinstance(self.extra_data, RunResponseExtraData) else self.extra_data)
        if self.images is not None:
            _dict['images'] = [img.model_dump(exclude_none=True) for img in self.images]
        if self.videos is not None:
            _dict['videos'] = [vid.model_dump(exclude_none=True) for vid in self.videos]
        if self.audio is not None:
            _dict['audio'] = [aud.model_dump(exclude_none=True) for aud in self.audio]
        if self.response_audio is not None:
            _dict['response_audio'] = (self.response_audio.to_dict() if isinstance(self.response_audio, AudioResponse) else self.response_audio)
        if isinstance(self.content, BaseModel):
            _dict['content'] = self.content.model_dump(exclude_none=True)
        return _dict

    def to_json(self) -> str:
        _dict = self.to_dict()
        return json.dumps(_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResponse':
        messages = data.pop('messages', None)
        messages = [Message.model_validate(message) for message in messages] if messages else None
        return cls(messages=messages, **data)

    def get_content_as_string(self, **kwargs) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, **kwargs)


class TeamRunResponse(RunResponse):
    def __init__(self, member_responses: List[Union['TeamRunResponse', RunResponse]] = None, team_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.member_responses = member_responses or []
        self.team_id = team_id

    def to_dict(self) -> Dict[str, Any]:
        _dict = {k: v
            for k, v in self.__dict__.items()
            if v is not None and k not in ['messages', 'extra_data', 'images', 'videos', 'audio', 'response_audio']}
        if self.messages is not None:
            _dict['messages'] = [m.to_dict() for m in self.messages]
        if self.extra_data is not None:
            _dict['extra_data'] = self.extra_data.to_dict()
        if self.images is not None:
            _dict['images'] = [img.model_dump(exclude_none=True) for img in self.images]
        if self.videos is not None:
            _dict['videos'] = [vid.model_dump(exclude_none=True) for vid in self.videos]
        if self.audio is not None:
            _dict['audio'] = [aud.model_dump(exclude_none=True) for aud in self.audio]
        if self.response_audio is not None:
            _dict['response_audio'] = self.response_audio.to_dict()
        if self.member_responses:
            _dict['member_responses'] = [response.to_dict() for response in self.member_responses]
        if isinstance(self.content, BaseModel):
            _dict['content'] = self.content.model_dump(exclude_none=True)
        return _dict

    def to_json(self) -> str:
        _dict = self.to_dict()
        return json.dumps(_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamRunResponse':
        messages = data.pop('messages', None)
        messages = [Message.model_validate(message) for message in messages] if messages else None
        member_responses = data.pop('member_responses', None)
        parsed_member_responses: List[Union['TeamRunResponse', RunResponse]] = []
        if member_responses is not None:
            for response in member_responses:
                if 'agent_id' in response:
                    parsed_member_responses.append(RunResponse.from_dict(response))
                else:
                    parsed_member_responses.append(cls.from_dict(response))
        extra_data = data.pop('extra_data', None)
        if extra_data is not None:
            extra_data = RunResponseExtraData.from_dict(extra_data)
        images = data.pop('images', None)
        images = [ImageArtifact.model_validate(image) for image in images] if images else None
        videos = data.pop('videos', None)
        videos = [VideoArtifact.model_validate(video) for video in videos] if videos else None
        audio = data.pop('audio', None)
        audio = [AudioArtifact.model_validate(audio) for audio in audio] if audio else None
        response_audio = data.pop('response_audio', None)
        response_audio = AudioResponse.model_validate(response_audio) if response_audio else None
        return cls(messages=messages, member_responses=parsed_member_responses, extra_data=extra_data, images=images, videos=videos, audio=audio, response_audio=response_audio, **data)

    def get_content_as_string(self, **kwargs) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, **kwargs)

    def add_member_run(self, run_response: Union['TeamRunResponse', RunResponse]) -> None:
        self.member_responses.append(run_response)
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


class AgentRun(BaseModel):
    message: Optional[Message] = None
    messages: Optional[List[Message]] = None
    response: Optional[RunResponse] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        response = {'message': self.message.to_dict() if self.message else None, 'messages': [message.to_dict() for message in self.messages] if self.messages else None, 'response': self.response.to_dict() if self.response else None}
        return {k: v for k, v in response.items() if v is not None}


class MemoryRow(BaseModel):
    memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={'created_at', 'updated_at'})
        _dict['created_at'] = self.created_at.isoformat() if self.created_at else None
        _dict['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        if self.id is None:
            memory_str = json.dumps(self.memory, sort_keys=True)
            cleaned_memory = memory_str.replace(' ', '').replace('\n', '').replace('\t', '')
            self.id = md5(cleaned_memory.encode()).hexdigest()
        return _dict

    def to_dict(self) -> Dict[str, Any]:
        return self.serializable_dict()


class MemoryDb(ABC):
    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_memories(self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None) -> List[MemoryRow]:
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
    def __init__(self, table_name: str = 'memory', db_url: Optional[str] = None, db_file: Optional[str] = None, db_engine: Optional[Engine] = None):
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            db_path = Path(db_file).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f'sqlite:///{db_path}')
        else:
            _engine = create_engine('sqlite://')
        if _engine is None:
            raise ValueError('Must provide either db_url, db_file or db_engine')
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()
        self.inspector = inspect(self.db_engine)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(self.table_name, self.metadata, Column('id', String, primary_key=True), Column('user_id', String), Column('memory', String), Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')), Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP')), extend_existing=True)

    def create(self) -> None:
        if not self.table_exists():
            try:
                print(f'Creating table: {self.table_name}')
                self.table.create(self.db_engine, checkfirst=True)
            except Exception as e:
                print(f'Error creating table "{self.table_name}": {e}')
                raise

    def memory_exists(self, memory: MemoryRow) -> bool:
        with self.Session() as session:
            stmt = select(self.table.c.id).where(self.table.c.id == memory.id)
            result = session.execute(stmt).first()
            return result is not None

    def read_memories(self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        try:
            with self.Session() as session:
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if sort == 'asc':
                    stmt = stmt.order_by(self.table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(self.table.c.created_at.desc())
                if limit is not None:
                    stmt = stmt.limit(limit)
                result = session.execute(stmt)
                for row in result:
                    memories.append(MemoryRow(id=row.id, user_id=row.user_id, memory=eval(row.memory)))
        except SQLAlchemyError as e:
            print(f'Exception reading from table: {e}')
            print(f'Table does not exist: {self.table_name}')
            print('Creating table for future transactions')
            self.create()
        return memories

    def upsert_memory(self, memory: MemoryRow, create_and_retry: bool = True) -> None:
        try:
            with self.Session() as session:
                existing = session.execute(select(self.table).where(self.table.c.id == memory.id)).first()
                if existing:
                    stmt = (self.table.update()
                        .where(self.table.c.id == memory.id)
                        .values(user_id=memory.user_id, memory=str(memory.memory), updated_at=text('CURRENT_TIMESTAMP')))
                else:
                    stmt = self.table.insert().values(id=memory.id, user_id=memory.user_id, memory=str(memory.memory))
                session.execute(stmt)
                session.commit()
        except SQLAlchemyError as e:
            print(f'Exception upserting into table: {e}')
            if not self.table_exists():
                print(f'Table does not exist: {self.table_name}')
                print('Creating table for future transactions')
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
            print(f'Deleting table: {self.table_name}')
            self.table.drop(self.db_engine)

    def table_exists(self) -> bool:
        print(f'Checking if table exists: {self.table.name}')
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
        pass


class MemoryManager(BaseModel):
    model: Optional[Model] = None
    user_id: Optional[str] = None
    limit: Optional[int] = None
    system_prompt: Optional[str] = None
    db: Optional[MemoryDb] = None
    input_message: Optional[str] = None
    _tools_for_model: Optional[List[Dict]] = None
    _functions_for_model: Optional[Dict[str, Function]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_model(self) -> None:
        if self.model is None:
            self.model = Ollama()
        self.add_tools_to_model(model=self.model)

    def add_tools_to_model(self, model: Model) -> None:
        if self._tools_for_model is None:
            self._tools_for_model = []
        if self._functions_for_model is None:
            self._functions_for_model = {}
        for tool in [
            self.add_memory, self.update_memory, self.delete_memory, self.clear_memory, ]:
            try:
                function_name = tool.__name__
                if function_name not in self._functions_for_model:
                    func = Function.from_callable(tool)
                    self._functions_for_model[func.name] = func
                    self._tools_for_model.append({'type': 'function', 'function': func.to_dict()})
                    print(f'Included function {func.name}')
            except Exception as e:
                print(f'Could not add function {tool}: {e}')
        model.set_tools(tools=self._tools_for_model)
        model.set_functions(functions=self._functions_for_model)

    def get_existing_memories(self) -> Optional[List[MemoryRow]]:
        if self.db is None:
            return None
        return self.db.read_memories(user_id=self.user_id, limit=self.limit)

    def add_memory(self, memory: str) -> str:
        try:
            if self.db:
                self.db.upsert_memory(MemoryRow(user_id=self.user_id, memory=Memory(memory=memory, input=self.input_message).to_dict()))
            return 'Memory added successfully'
        except Exception as e:
            print(f'Error storing memory in db: {e}')
            return f'Error adding memory: {e}'

    def delete_memory(self, id: str) -> str:
        try:
            if self.db:
                self.db.delete_memory(id=id)
            return 'Memory deleted successfully'
        except Exception as e:
            print(f'Error deleting memory in db: {e}')
            return f'Error deleting memory: {e}'

    def update_memory(self, id: str, memory: str) -> str:
        try:
            if self.db:
                self.db.upsert_memory(MemoryRow(id=id, user_id=self.user_id, memory=Memory(memory=memory, input=self.input_message).to_dict()))
            return 'Memory updated successfully'
        except Exception as e:
            print(f'Error updating memory in db: {e}')
            return f'Error updating memory: {e}'

    def clear_memory(self) -> str:
        try:
            if self.db:
                self.db.clear()
            return 'Memory cleared successfully'
        except Exception as e:
            print(f'Error clearing memory in db: {e}')
            return f'Error clearing memory: {e}'

    def get_system_message(self) -> Message:
        system_prompt_lines = [
            '你的任务是为用户的消息生成一个简洁的记忆。',
            '创建一个能够捕获用户提供的关键信息的内存，就像您正在存储它以供将来参考一样',
            '记忆应该是一个简短的第三人称陈述，概括了用户输入中最重要的方面，没有添加任何无关的细节。',
            '这些记忆将用于增强用户在后续对话中的体验。',
            '您还将收到一份现有记忆列表。您可以：\n1.使用`Add_memory`工具添加新内存。\n2.使用`Update_memory`工具更新内存。\n3.使用`Delete_memory`工具删除内存。\n4.使用“Clear_memory”工具清除所有记忆。使用时要格外小心，因为它会从数据库中删除所有内存。'
        ]
        existing_memories = self.get_existing_memories()
        if existing_memories and len(existing_memories) > 0:
            system_prompt_lines.extend([
                    '\nExisting memories:', '<existing_memories>\n'
                    + '\n'.join([f'  - id: {m.id} | memory: {m.memory}' for m in existing_memories])
                    + '\n</existing_memories>', ])
        return Message(role='system', content='\n'.join(system_prompt_lines))

    def run(self, message: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        print('*********** MemoryManager Start ***********')
        self.update_model()
        messages_for_model: List[Message] = [self.get_system_message()]
        user_prompt_message = Message(role='user', content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]
        self.input_message = message
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print('*********** MemoryManager End ***********')
        return response.content

    async def arun(self, message: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        print('*********** Async MemoryManager Start ***********')
        self.update_model()
        messages_for_model: List[Message] = [self.get_system_message()]
        user_prompt_message = Message(role='user', content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]
        self.input_message = message
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print('*********** Async MemoryManager End ***********')
        return response.content


class MemoryRetrieval(str, Enum):
    last_n = 'last_n'
    first_n = 'first_n'
    semantic = 'semantic'


class Memory(BaseModel):
    memory: str
    id: Optional[str] = None
    topic: Optional[str] = None
    input: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class MemoryClassifier(BaseModel):
    model: Optional[Model] = None
    system_prompt: Optional[str] = None
    existing_memories: Optional[List[Memory]] = None

    def update_model(self) -> None:
        if self.model is None:
            self.model = Ollama()

    def get_system_message(self) -> Message:
        system_prompt_lines = [
            '你的任务是确定用户的消息是否包含值得记住的信息，以便将来进行对话。',
            '这包括可以个性化与用户正在进行的交互的详细信息，例如：\n- 个人事实：姓名、年龄、职业、地点、兴趣、偏好等。\n',
            '-用户分享的重大生活事件或经历\n-有关用户当前状况、挑战或目标的重要上下文\n- 用户喜欢或不喜欢什么，他们的观点、信仰、价值观等。\n',
            '-提供对用户个性、观点或需求有价值见解的任何其他细节', '你的任务是决定用户输入是否包含任何值得记住的上述信息。',
            '如果用户输入包含任何值得记住以备将来对话的信息，请回答‘是’。', '如果输入不包含任何值得保存的重要细节，请回答“否”以忽略它。',
            '您还将收到一份现有记忆列表，以帮助您决定输入是新的还是已知的。', '如果与输入匹配的内存已经存在，请使用‘否’进行响应以保持原样。',
            '如果存在需要更新或删除的内存，请回答“是”以更新 / 删除它。', '您只能回答“是”或“否”。其他任何内容都不会被视为有效的回答。'
        ]
        if self.existing_memories and len(self.existing_memories) > 0:
            system_prompt_lines.extend([
                    '\nExisting memories:', '<existing_memories>\n'
                    + '\n'.join([f'  - {m.memory}' for m in self.existing_memories])
                    + '\n</existing_memories>'])
        return Message(role='system', content='\n'.join(system_prompt_lines))

    def run(self, message: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        print('*********** MemoryClassifier Start ***********')
        self.update_model()
        messages_for_model: List[Message] = [self.get_system_message()]
        user_prompt_message = Message(role='user', content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print('*********** MemoryClassifier End ***********')
        return response.content

    async def arun(self, message: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        print('*********** Async MemoryClassifier Start ***********')
        self.update_model()
        messages_for_model: List[Message] = [self.get_system_message()]
        user_prompt_message = Message(role='user', content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print('*********** Async MemoryClassifier End ***********')
        return response.content


class SessionSummary(BaseModel):
    summary: str = Field(..., description='会议总结。简明扼要，只关注重要信息。不要编造任何东西。')
    topics: Optional[List[str]] = Field(None, description='会议讨论的主题')

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


class MemorySummarizer(BaseModel):
    model: Optional[Model] = None
    use_structured_outputs: bool = False

    def update_model(self) -> None:
        if self.model is None:
            self.model = Ollama()
        if self.use_structured_outputs:
            self.model.response_format = SessionSummary
            self.model.structured_outputs = True
        else:
            self.model.response_format = {'type': 'json_object'}

    def get_system_message(self, messages_for_summarization: List[Dict[str, str]]) -> Message:
        system_prompt = dedent('分析用户和助手之间的以下对话，并提取以下详细信息：\n-摘要（str）：提供会议的简明摘要，重点介绍有助于未来互动的重要信息。\n-Topics（可选[List[str]]）：列出会话中讨论的主题。\n请忽略任何琐碎的信息。\n对话：')
        conversation = []
        for message_pair in messages_for_summarization:
            conversation.append(f'User: {message_pair["user"]}')
            if 'assistant' in message_pair:
                conversation.append(f'Assistant: {message_pair["assistant"]}')
            elif 'model' in message_pair:
                conversation.append(f'Assistant: {message_pair["model"]}')
        system_prompt += '\n'.join(conversation)
        if not self.use_structured_outputs:
            system_prompt += '\n\nProvide your output as a JSON containing the following fields:'
            json_schema = SessionSummary.model_json_schema()
            response_model_properties = {}
            json_schema_properties = json_schema.get('properties')
            if json_schema_properties is not None:
                for field_name, field_properties in json_schema_properties.items():
                    formatted_field_properties = {prop_name: prop_value
                        for prop_name, prop_value in field_properties.items()
                        if prop_name != 'title'}
                    response_model_properties[field_name] = formatted_field_properties
            if len(response_model_properties) > 0:
                system_prompt += '\n<json_fields>'
                system_prompt += f'\n{json.dumps([key for key in response_model_properties.keys() if key != "$defs"])}'
                system_prompt += '\n</json_fields>'
                system_prompt += '\nHere are the properties for each field:'
                system_prompt += '\n<json_field_properties>'
                system_prompt += f'\n{json.dumps(response_model_properties, indent=2)}'
                system_prompt += '\n</json_field_properties>'
            system_prompt += '\nStart your response with `{` and end it with `}`.'
            system_prompt += '\nYour output will be passed to json.loads() to convert it to a Python object.'
            system_prompt += '\nMake sure it only contains valid JSON.'
        return Message(role='system', content=system_prompt)

    def run(self, message_pairs: List[Tuple[Message, Message]], **kwargs: Any) -> Optional[SessionSummary]:
        print('*********** MemorySummarizer Start ***********')
        if message_pairs is None or len(message_pairs) == 0:
            print('No message pairs provided for summarization.')
            return None
        self.update_model()
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append({user_message.role: user_message.get_content_string(), assistant_message.role: assistant_message.get_content_string()})
        messages_for_model: List[Message] = [
            self.get_system_message(messages_for_summarization), Message(role='user', content='Provide the summary of the conversation.'), ]
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        print('*********** MemorySummarizer End ***********')
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    if response.content.startswith('```json'):
                        response.content = response.content.replace('```json\n', '').replace('\n```', '')
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            print(f'Failed to validate session_summary response: {exc}')
                return session_summary
            except Exception as e:
                print(f'Failed to convert response to session_summary: {e}')
        return None

    async def arun(self, message_pairs: List[Tuple[Message, Message]], **kwargs: Any) -> Optional[SessionSummary]:
        print('*********** Async MemorySummarizer Start ***********')
        if message_pairs is None or len(message_pairs) == 0:
            print('No message pairs provided for summarization.')
            return None
        self.update_model()
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append({user_message.role: user_message.get_content_string(), assistant_message.role: assistant_message.get_content_string()})
        messages_for_model: List[Message] = [
            self.get_system_message(messages_for_summarization), Message(role='user', content='Provide the summary of the conversation.'), ]
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        print('*********** Async MemorySummarizer End ***********')
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    if response.content.startswith('```json'):
                        response.content = response.content.replace('```json\n', '').replace('\n```', '')
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            print(f'Failed to validate session_summary response: {exc}')
                return session_summary
            except Exception as e:
                print(f'Failed to convert response to session_summary: {e}')
        return None


class AgentMemory(BaseModel):
    runs: List[AgentRun] = []
    messages: List[Message] = []
    update_system_message_on_change: bool = False
    summary: Optional[SessionSummary] = None
    create_session_summary: bool = False
    update_session_summary_after_run: bool = True
    summarizer: Optional[MemorySummarizer] = None
    create_user_memories: bool = False
    update_user_memories_after_run: bool = True
    db: Optional[MemoryDb] = None
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    num_memories: Optional[int] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None
    updating_memory: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(exclude_none=True, include={'update_system_message_on_change', 'create_session_summary', 'update_session_summary_after_run', 'create_user_memories', 'update_user_memories_after_run', 'user_id', 'num_memories'})
        if self.summary is not None:
            _memory_dict['summary'] = self.summary.to_dict()
        if self.memories is not None:
            _memory_dict['memories'] = [memory.to_dict() for memory in self.memories]
        if self.messages is not None:
            _memory_dict['messages'] = [message.to_dict() for message in self.messages]
        if self.runs is not None:
            _memory_dict['runs'] = [run.to_dict() for run in self.runs]
        return _memory_dict

    def add_run(self, agent_run: AgentRun) -> None:
        self.runs.append(agent_run)
        print('Added AgentRun to AgentMemory')

    def add_system_message(self, message: Message, system_message_role: str = 'system') -> None:
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            if system_message_index is not None:
                if self.messages[system_message_index].content != message.content and self.update_system_message_on_change:
                    print('Updating system message in memory with new content')
                    self.messages[system_message_index] = message
            else:
                self.messages.insert(0, message)

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)
        print(f'Added {len(messages)} Messages to AgentMemory')

    def get_messages(self) -> List[Dict[str, Any]]:
        return [message.model_dump() for message in self.messages]

    def get_messages_from_last_n_runs(self, last_n: Optional[int] = None, skip_role: Optional[str] = None) -> List[Message]:
        if not self.runs:
            return []
        runs_to_process = self.runs if last_n is None else self.runs[-last_n:]
        messages_from_history = []
        for run in runs_to_process:
            if not (run.response and run.response.messages):
                continue
            for message in run.response.messages:
                if skip_role and message.role == skip_role:
                    continue
                if hasattr(message, 'from_history') and message.from_history:
                    continue
                messages_from_history.append(message)
        print(f'Getting messages from previous runs: {len(messages_from_history)}')
        return messages_from_history

    def get_message_pairs(self, user_role: str = 'user', assistant_role: Optional[List[str]] = None) -> List[Tuple[Message, Message]]:
        if assistant_role is None:
            assistant_role = ['assistant', 'model', 'CHATBOT']
        runs_as_message_pairs: List[Tuple[Message, Message]] = []
        for run in self.runs:
            if run.response and run.response.messages:
                user_messages_from_run = None
                assistant_messages_from_run = None
                for message in run.response.messages:
                    if hasattr(message, 'from_history') and message.from_history:
                        continue
                    if message.role == user_role:
                        user_messages_from_run = message
                        break
                for message in run.response.messages[::-1]:
                    if hasattr(message, 'from_history') and message.from_history:
                        continue
                    if message.role in assistant_role:
                        assistant_messages_from_run = message
                        break
                if user_messages_from_run and assistant_messages_from_run:
                    runs_as_message_pairs.append((user_messages_from_run, assistant_messages_from_run))
        return runs_as_message_pairs

    def get_tool_calls(self, num_calls: Optional[int] = None) -> List[Dict[str, Any]]:
        tool_calls = []
        for message in self.messages[::-1]:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(tool_call)
                    if num_calls and len(tool_calls) >= num_calls:
                        return tool_calls
        return tool_calls

    def load_user_memories(self) -> None:
        if self.db is None:
            return
        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(user_id=self.user_id, limit=self.num_memories, sort='asc' if self.retrieval == MemoryRetrieval.first_n else 'desc')
            else:
                raise NotImplementedError('Semantic retrieval not yet supported.')
        except Exception as e:
            print(f'Error reading memory: {e}')
            return
        self.memories = []
        if memory_rows is None or len(memory_rows) == 0:
            return
        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                print(f'Error loading memory: {e}')
                continue

    def should_update_memory(self, input: str) -> bool:
        if self.classifier is None:
            self.classifier = MemoryClassifier()
        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input)
        if classifier_response == 'yes':
            return True
        return False

    async def ashould_update_memory(self, input: str) -> bool:
        if self.classifier is None:
            self.classifier = MemoryClassifier()
        self.classifier.existing_memories = self.memories
        classifier_response = await self.classifier.arun(input)
        if classifier_response == 'yes':
            return True
        return False

    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        if input is None or not isinstance(input, str):
            return 'Invalid message content'
        if self.db is None:
            print('MemoryDb not provided.')
            return 'Please provide a db to store memories'
        self.updating_memory = True
        should_update_memory = force or self.should_update_memory(input=input)
        print(f'Update memory: {should_update_memory}')
        if not should_update_memory:
            print('Memory update not required')
            return 'Memory update not required'
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
        if input is None or not isinstance(input, str):
            return 'Invalid message content'
        if self.db is None:
            print('MemoryDb not provided.')
            return 'Please provide a db to store memories'
        self.updating_memory = True
        should_update_memory = force or await self.ashould_update_memory(input=input)
        print(f'Async update memory: {should_update_memory}')
        if not should_update_memory:
            print('Memory update not required')
            return 'Memory update not required'
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
        self.updating_memory = True
        if self.summarizer is None:
            self.summarizer = MemorySummarizer()
        self.summary = self.summarizer.run(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    async def aupdate_summary(self) -> Optional[SessionSummary]:
        self.updating_memory = True
        if self.summarizer is None:
            self.summarizer = MemorySummarizer()
        self.summary = await self.summarizer.arun(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    def clear(self) -> None:
        self.runs = []
        self.messages = []
        self.summary = None
        self.memories = None

    def deep_copy(self) -> 'AgentMemory':
        copied_obj = self.__class__(**self.to_dict())
        for field_name, field_value in self.__dict__.items():
            if field_name not in ['db', 'classifier', 'manager', 'summarizer']:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    print(f'Failed to deepcopy field: {field_name} - {e}')
                    setattr(copied_obj, field_name, field_value)
        copied_obj.db = self.db
        copied_obj.classifier = self.classifier
        copied_obj.manager = self.manager
        copied_obj.summarizer = self.summarizer
        return copied_obj


class TeamRun:
    def __init__(self,  message: Message = None, member_runs: List[AgentRun] = None, response: TeamRunResponse = None):
        self.message = message
        self.member_runs = member_runs
        self.response = response

    def to_dict(self) -> Dict[str, Any]:
        message = self.message.to_dict() if self.message else None
        member_responses = [run.to_dict() for run in self.member_runs] if self.member_runs else None
        response = self.response.to_dict() if self.response else None
        return {'message': message, 'member_responses': member_responses, 'response': response}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamRun':
        message = Message.model_validate(data.get('message')) if data.get('message') else None
        member_runs = ([AgentRun.model_validate(run) for run in data.get('member_runs', [])] if data.get('member_runs') else None)
        response = TeamRunResponse.from_dict(data.get('response', {})) if data.get('response') else None
        return cls(message=message, member_runs=member_runs, response=response)


class TeamMemberInteraction:
    def __init__(self, member_name='', task='', response: RunResponse = None):
        self.member_name = member_name
        self.task = task
        self.response = response


class TeamContext:
    def __init__(self, member_interactions: List[TeamMemberInteraction] = None, text: str = None):
        self.member_interactions = member_interactions or []
        self.text = text


class TeamMemory:
    def __init__(self,
    runs: List[TeamRun] = None,
    messages: List[Message] = None,
    update_system_message_on_change=True,
    team_context: TeamContext = None,
    create_user_memories=False,
    update_user_memories_after_run=True,
    db: MemoryDb = None,
    user_id: str = None,
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n,
    memories: List[Memory] = None,
    classifier: MemoryClassifier = None,
    manager: MemoryManager = None,
    num_memories: int = None,
    updating_memory=False,
    model_config: dict = None):
        self.runs = runs or []
        self.messages = messages or []
        self.update_system_message_on_change = update_system_message_on_change
        self.team_context = team_context
        self.create_user_memories = create_user_memories
        self.update_user_memories_after_run = update_user_memories_after_run
        self.db = db
        self.user_id = user_id
        self.retrieval = retrieval
        self.memories = memories
        self.classifier = classifier
        self.manager = manager
        self.num_memories = num_memories
        self.updating_memory = updating_memory
        self.model_config = model_config or {'arbitrary_types_allowed': True}

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = {}
        for key, value in self.__dict__.items():
            if value is not None and key in ['update_system_message_on_change', 'create_user_memories', 'update_user_memories_after_run', 'user_id', 'num_memories']:
                _memory_dict[key] = value
        if self.messages is not None:
            _memory_dict['messages'] = [message.to_dict() for message in self.messages]
        if self.memories is not None:
            _memory_dict['memories'] = [memory.to_dict() for memory in self.memories]
        if self.runs is not None:
            _memory_dict['runs'] = [run.to_dict() for run in self.runs]
        return _memory_dict

    def add_interaction_to_team_context(self, member_name: str, task: str, run_response: RunResponse) -> None:
        if self.team_context is None:
            self.team_context = TeamContext()
        self.team_context.member_interactions.append(TeamMemberInteraction(member_name=member_name, task=task, response=run_response))
        print(f'Updated team context with member name: {member_name}')

    def set_team_context_text(self, text: str) -> None:
        if self.team_context:
            self.team_context.text = text
        else:
            self.team_context = TeamContext(text=text)

    def get_team_context_str(self) -> str:
        if self.team_context and self.team_context.text:
            return f'<team context>\n{self.team_context.text}\n</team context>\n'
        return ''

    def get_team_member_interactions_str(self) -> str:
        team_member_interactions_str = ''
        if self.team_context and self.team_context.member_interactions:
            team_member_interactions_str += '<member interactions>\n'
            for interaction in self.team_context.member_interactions:
                team_member_interactions_str += f'Member: {interaction.member_name}\n'
                team_member_interactions_str += f'Task: {interaction.task}\n'
                team_member_interactions_str += f"Response: {interaction.response.to_dict().get('content', '')}\n"
                team_member_interactions_str += '\n'
            team_member_interactions_str += '</member interactions>\n'
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
        self.runs.append(team_run)
        print('Added TeamRun to TeamMemory')

    def add_system_message(self, message: Message, system_message_role: str = 'system') -> None:
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            if system_message_index is not None:
                if self.messages[system_message_index].content != message.content and self.update_system_message_on_change:
                    print('Updating system message in memory with new content')
                    self.messages[system_message_index] = message
            else:
                self.messages.insert(0, message)

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)
        print(f'Added {len(messages)} Messages to TeamMemory')

    def get_messages(self) -> List[Dict[str, Any]]:
        return [message.model_dump() for message in self.messages]

    def get_messages_from_last_n_runs(self, last_n: Optional[int] = None, skip_role: Optional[str] = None) -> List[Message]:
        if not self.runs:
            return []
        runs_to_process = self.runs if last_n is None else self.runs[-last_n:]
        messages_from_history = []
        for run in runs_to_process:
            if not (run.response and run.response.messages):
                continue
            for message in run.response.messages:
                if skip_role and message.role == skip_role:
                    continue
                if hasattr(message, 'from_history') and message.from_history:
                    continue
                messages_from_history.append(message)
        print(f'Getting messages from previous runs: {len(messages_from_history)}')
        return messages_from_history

    def get_all_messages(self) -> List[Tuple[Message, Message]]:
        assistant_role = ['assistant', 'model', 'CHATBOT']
        runs_as_message_pairs: List[Tuple[Message, Message]] = []
        for run in self.runs:
            if run.response and run.response.messages:
                user_message_from_run = None
                assistant_message_from_run = None
                for message in run.response.messages:
                    if message.role == 'user':
                        user_message_from_run = message
                        break
                for message in run.response.messages[::-1]:
                    if message.role in assistant_role:
                        assistant_message_from_run = message
                        break
                if user_message_from_run and assistant_message_from_run:
                    runs_as_message_pairs.append((user_message_from_run, assistant_message_from_run))
        return runs_as_message_pairs

    def load_user_memories(self) -> None:
        if self.db is None:
            return
        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(user_id=self.user_id, limit=self.num_memories, sort='asc' if self.retrieval == MemoryRetrieval.first_n else 'desc')
            else:
                raise NotImplementedError('Semantic retrieval not yet supported.')
        except Exception as e:
            print(f'Error reading memory: {e}')
            return
        self.memories = []
        if memory_rows is None or len(memory_rows) == 0:
            return
        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                print(f'Error loading memory: {e}')
                continue

    def should_update_memory(self, input: str) -> bool:
        if self.classifier is None:
            self.classifier = MemoryClassifier()
        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input)
        if classifier_response == 'yes':
            return True
        return False

    async def ashould_update_memory(self, input: str) -> bool:
        if self.classifier is None:
            self.classifier = MemoryClassifier()
        self.classifier.existing_memories = self.memories
        classifier_response = await self.classifier.arun(input)
        if classifier_response == 'yes':
            return True
        return False

    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        if input is None or not isinstance(input, str):
            return 'Invalid message content'
        if self.db is None:
            print('MemoryDb not provided.')
            return 'Please provide a db to store memories'
        self.updating_memory = True
        should_update_memory = force or self.should_update_memory(input=input)
        print(f'Update memory: {should_update_memory}')
        if not should_update_memory:
            print('Memory update not required')
            return 'Memory update not required'
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
        if input is None or not isinstance(input, str):
            return 'Invalid message content'
        if self.db is None:
            print('MemoryDb not provided.')
            return 'Please provide a db to store memories'
        self.updating_memory = True
        should_update_memory = force or await self.ashould_update_memory(input=input)
        print(f'Async update memory: {should_update_memory}')
        if not should_update_memory:
            print('Memory update not required')
            return 'Memory update not required'
        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)
        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id
        response = await self.manager.arun(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    def deep_copy(self) -> 'TeamMemory':
        copied_obj = self.__class__(**self.to_dict())
        for field_name, field_value in self.__dict__.items():
            if field_name not in ['db', 'classifier', 'manager']:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    print(f'Failed to deepcopy field: {field_name} - {e}')
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
        self.runs.append(workflow_run)
        print('已将WorkflowRun添加到WorkflowMemory')

    def clear(self) -> None:
        self.runs = []

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> 'WorkflowMemory':
        new_memory = self.model_copy(deep=True, update=update)
        new_memory.clear()
        return new_memory
