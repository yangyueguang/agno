import os
import collections.abc
import inspect
from types import GeneratorType
from typing import Any, Callable, Dict, List, Optional, cast
import uuid
from pydantic import BaseModel
from agno.agent import Agent
from agno.media import AudioArtifact, ImageArtifact, VideoArtifact
from agno.memory import WorkflowMemory, WorkflowRun
from agno.run import RunEvent, RunResponse
from agno.storage import Storage, WorkflowSession
from copy import copy, deepcopy


def merge_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dictionaries(a[key], b[key])
        else:
            a[key] = b[key]


def nested_model_dump(value):
    if isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, dict):
        return {k: nested_model_dump(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [nested_model_dump(item) for item in value]
    return value


class Workflow:
    def __init__(self, name: Optional[str] = None, workflow_id: Optional[str] = None, description: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, session_name: Optional[str] = None, session_state: Optional[Dict[str, Any]] = None, memory: Optional[WorkflowMemory] = None, storage: Optional[Storage] = None, extra_data: Optional[Dict[str, Any]] = None, debug_mode: bool = False, monitoring: bool = False, telemetry: bool = True):
        self.name = name or self.__class__.__name__
        self.workflow_id = workflow_id
        self.description = description or self.__class__.description
        self.user_id = user_id
        self.session_id = session_id
        self.session_name = session_name
        self.session_state: Dict[str, Any] = session_state or {}
        self.memory = memory
        self.storage = storage
        self.extra_data = extra_data
        self.debug_mode = debug_mode
        self.monitoring = monitoring
        self.telemetry = telemetry
        self.run_id = None
        self.run_input = None
        self.run_response = None
        self.images = None
        self.videos = None
        self.audio = None
        self.workflow_session: Optional[WorkflowSession] = None
        self._subclass_run: Optional[Callable] = None
        self._run_parameters: Optional[Dict[str, Any]] = None
        self._run_return_type: Optional[str] = None
        self.update_run_method()
        self.__post_init__()

    def __post_init__(self):
        for field_name, value in self.__class__.__dict__.items():
            if isinstance(value, Agent):
                value.session_id = self.session_id

    def run(self, **kwargs: Any):
        print(f'{self.__class__.__name__}.run() method not implemented.')
        return

    def run_workflow(self, **kwargs: Any):
        self.set_storage_mode()
        self.set_debug()
        self.set_workflow_id()
        self.set_session_id()
        self.initialize_memory()
        self.memory = cast(WorkflowMemory, self.memory)
        self.run_id = str(uuid.uuid4())
        self.run_input = kwargs
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, workflow_id=self.workflow_id)
        self.read_from_storage()
        self.update_agent_session_ids()
        print(f'*********** Workflow Run Start: {self.run_id} ***********')
        try:
            self._subclass_run = cast(Callable, self._subclass_run)
            result = self._subclass_run(**kwargs)
        except Exception as e:
            print(f'Workflow.run() failed: {e}')
            raise e
        if isinstance(result, (GeneratorType, collections.abc.Iterator)):
            self.run_response.content = ''
            def result_generator():
                self.run_response = cast(RunResponse, self.run_response)
                self.memory = cast(WorkflowMemory, self.memory)
                for item in result:
                    if isinstance(item, RunResponse):
                        item.run_id = self.run_id
                        item.session_id = self.session_id
                        item.workflow_id = self.workflow_id
                        if item.content is not None and isinstance(item.content, str):
                            self.run_response.content += item.content
                    else:
                        print(f'Workflow.run() should only yield RunResponse objects, got: {type(item)}')
                    yield item
                self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
                self.write_to_storage()
                print(f'*********** Workflow Run End: {self.run_id} ***********')
            return result_generator()
        elif isinstance(result, RunResponse):
            result.run_id = self.run_id
            result.session_id = self.session_id
            result.workflow_id = self.workflow_id
            if result.content is not None and isinstance(result.content, str):
                self.run_response.content = result.content
            self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
            self.write_to_storage()
            print(f'*********** Workflow Run End: {self.run_id} ***********')
            return result
        else:
            print(f'Workflow.run() should only return RunResponse objects, got: {type(result)}')
            return None

    def set_storage_mode(self):
        if self.storage is not None:
            if self.storage.mode in ['agent', 'team']:
                print(f'You should not use storage in multiple modes. Current mode is {self.storage.mode}.')
            self.storage.mode = 'workflow'

    def set_workflow_id(self) -> str:
        if self.workflow_id is None:
            self.workflow_id = str(uuid.uuid4())
        print(f'*********** Workflow ID: {self.workflow_id} ***********')
        return self.workflow_id

    def set_session_id(self) -> str:
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        print(f'*********** Session ID: {self.session_id} ***********')
        return self.session_id

    def set_debug(self) -> None:
        if self.debug_mode or os.getenv('AGNO_DEBUG', 'false').lower() == 'true':
            self.debug_mode = True
            print('Debug logs enabled')

    def set_monitoring(self) -> None:
        if self.monitoring or os.getenv('AGNO_MONITOR', 'false').lower() == 'true':
            self.monitoring = True
        else:
            self.monitoring = False
        if self.telemetry or os.getenv('AGNO_TELEMETRY', 'true').lower() == 'true':
            self.telemetry = True
        else:
            self.telemetry = False

    def initialize_memory(self) -> None:
        if self.memory is None:
            self.memory = WorkflowMemory()

    def update_run_method(self):
        if self.__class__.run is not Workflow.run:
            self._subclass_run = self.__class__.run.__get__(self)
            sig = inspect.signature(self.__class__.run)
            self._run_parameters = {param_name: {'name': param_name, 'default': param.default.default
                    if hasattr(param.default, '__class__') and param.default.__class__.__name__ == 'FieldInfo'
                    else (param.default if param.default is not inspect.Parameter.empty else None), 'annotation': (param.annotation.__name__
                        if hasattr(param.annotation, '__name__')
                        else (str(param.annotation).replace('typing.Optional[', '').replace(']', '')
                            if 'typing.Optional' in str(param.annotation)
                            else str(param.annotation)))
                    if param.annotation is not inspect.Parameter.empty
                    else None, 'required': param.default is inspect.Parameter.empty}
                for param_name, param in sig.parameters.items()
                if param_name != 'self'}
            return_annotation = sig.return_annotation
            self._run_return_type = (return_annotation.__name__
                if return_annotation is not inspect.Signature.empty and hasattr(return_annotation, '__name__')
                else str(return_annotation)
                if return_annotation is not inspect.Signature.empty
                else None)
            object.__setattr__(self, 'run', self.run_workflow.__get__(self))
        else:
            self._subclass_run = self.run
            self._run_parameters = {}
            self._run_return_type = None

    def update_agent_session_ids(self):
        for field_name, value in self.__class__.__dict__.items():
            if isinstance(value, Agent):
                field_value = getattr(self, field_name)
                field_value.session_id = self.session_id

    def get_workflow_data(self) -> Dict[str, Any]:
        workflow_data: Dict[str, Any] = {}
        if self.name is not None:
            workflow_data['name'] = self.name
        if self.workflow_id is not None:
            workflow_data['workflow_id'] = self.workflow_id
        if self.description is not None:
            workflow_data['description'] = self.description
        return workflow_data

    def get_session_data(self) -> Dict[str, Any]:
        session_data: Dict[str, Any] = {}
        if self.session_name is not None:
            session_data['session_name'] = self.session_name
        if self.session_state and len(self.session_state) > 0:
            session_data['session_state'] = nested_model_dump(self.session_state)
        if self.images is not None:
            session_data['images'] = [img.model_dump() for img in self.images]
        if self.videos is not None:
            session_data['videos'] = [vid.model_dump() for vid in self.videos]
        if self.audio is not None:
            session_data['audio'] = [aud.model_dump() for aud in self.audio]
        return session_data

    def get_workflow_session(self) -> WorkflowSession:
        self.memory = cast(WorkflowMemory, self.memory)
        self.session_id = cast(str, self.session_id)
        self.workflow_id = cast(str, self.workflow_id)
        return WorkflowSession(session_id=self.session_id, workflow_id=self.workflow_id, user_id=self.user_id, memory=self.memory.to_dict() if self.memory is not None else None, workflow_data=self.get_workflow_data(), session_data=self.get_session_data(), extra_data=self.extra_data)

    def load_workflow_session(self, session: WorkflowSession):
        if self.workflow_id is None and session.workflow_id is not None:
            self.workflow_id = session.workflow_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if session.workflow_data is not None:
            if self.name is None and 'name' in session.workflow_data:
                self.name = session.workflow_data.get('name')
        if session.session_data is not None:
            if self.session_name is None and 'session_name' in session.session_data:
                self.session_name = session.session_data.get('session_name')
            if 'session_state' in session.session_data:
                session_state_from_db = session.session_data.get('session_state')
                if session_state_from_db is not None and isinstance(session_state_from_db, dict) and len(session_state_from_db) > 0:
                    if len(self.session_state) > 0:
                        merge_dictionaries(session_state_from_db, self.session_state)
                    self.session_state = session_state_from_db
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
        if session.memory is not None:
            if self.memory is None:
                self.memory = WorkflowMemory()
            try:
                if 'runs' in session.memory:
                    try:
                        self.memory.runs = [WorkflowRun(**m) for m in session.memory['runs']]
                    except Exception as e:
                        print(f'Failed to load runs from memory: {e}')
            except Exception as e:
                print(f'Failed to load WorkflowMemory: {e}')
        print(f'-*- WorkflowSession loaded: {session.session_id}')

    def read_from_storage(self) -> Optional[WorkflowSession]:
        if self.storage is not None and self.session_id is not None:
            self.workflow_session = cast(WorkflowSession, self.storage.read(session_id=self.session_id))
            if self.workflow_session is not None:
                self.load_workflow_session(session=self.workflow_session)
        return self.workflow_session

    def write_to_storage(self) -> Optional[WorkflowSession]:
        if self.storage is not None:
            self.workflow_session = cast(WorkflowSession, self.storage.upsert(session=self.get_workflow_session()))
        return self.workflow_session

    def load_session(self, force: bool = False) -> Optional[str]:
        if self.workflow_session is not None and not force:
            if self.session_id is not None and self.workflow_session.session_id == self.session_id:
                return self.workflow_session.session_id
        if self.storage is not None:
            print(f'Reading WorkflowSession: {self.session_id}')
            self.read_from_storage()
            if self.workflow_session is None:
                print('-*- Creating new WorkflowSession')
                self.write_to_storage()
                if self.workflow_session is None:
                    raise Exception('Failed to create new WorkflowSession in storage')
                print(f'-*- Created WorkflowSession: {self.workflow_session.session_id}')
                self.log_workflow_session()
        return self.session_id

    def new_session(self) -> None:
        self.workflow_session = None
        self.session_id = str(uuid.uuid4())
        self.load_session(force=True)

    def log_workflow_session(self):
        print(f'*********** Logging WorkflowSession: {self.session_id} ***********')

    def rename(self, name: str) -> None:
        self.read_from_storage()
        self.name = name
        self.write_to_storage()
        self.log_workflow_session()

    def rename_session(self, session_name: str):
        self.read_from_storage()
        self.session_name = session_name
        self.write_to_storage()
        self.log_workflow_session()

    def delete_session(self, session_id: str):
        if self.storage is None:
            return
        self.storage.delete_session(session_id=session_id)

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> 'Workflow':
        fields_for_new_workflow: Dict[str, Any] = {}

        for field_name, value in self.__class__.__dict__.items():
            if value is not None:
                if isinstance(value, Agent):
                    fields_for_new_workflow[field_name] = value.deep_copy()
                else:
                    fields_for_new_workflow[field_name] = self._deep_copy_field(field_name, value)
        if update:
            fields_for_new_workflow.update(update)
        new_workflow = self.__class__(**fields_for_new_workflow)
        print(f'Created new {self.__class__.__name__}')
        return new_workflow

    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        if field_name == 'memory':
            return field_value.deep_copy()
        if isinstance(field_value, (list, dict, set, Storage)):
            try:
                return deepcopy(field_value)
            except Exception as e:
                print(f'Failed to deepcopy field: {field_name} - {e}')
                try:
                    return copy(field_value)
                except Exception as e:
                    print(f'Failed to copy field: {field_name} - {e}')
                    return field_value
        if isinstance(field_value, BaseModel):
            try:
                return field_value.model_copy(deep=True)
            except Exception as e:
                print(f'Failed to deepcopy field: {field_name} - {e}')
                try:
                    return field_value.model_copy(deep=False)
                except Exception as e:
                    print(f'Failed to copy field: {field_name} - {e}')
                    return field_value
        return field_value
