import time
from pathlib import Path
from sqlalchemy.dialects import sqlite
from sqlalchemy.types import String
import json
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.engine.row import Row
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session as SqlSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Column, MetaData, Table
from sqlalchemy.sql.expression import select, text
from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional
from typing import Union

@dataclass
class AgentSession:
    """Agent Session that is stored in the database"""
    # Session UUID
    session_id: str
    # ID of the user interacting with this agent
    user_id: Optional[str] = None
    # ID of the team session this agent session is associated with
    team_session_id: Optional[str] = None
    # Agent Memory
    memory: Optional[Dict[str, Any]] = None
    # Session Data: session_name, session_state, images, videos, audio
    session_data: Optional[Dict[str, Any]] = None
    # Extra Data stored with this agent
    extra_data: Optional[Dict[str, Any]] = None
    # The unix timestamp when this session was created
    created_at: Optional[int] = None
    # The unix timestamp when this session was last updated
    updated_at: Optional[int] = None
    # ID of the agent that this session is associated with
    agent_id: Optional[str] = None
    # Agent Data: agent_id, name and model
    agent_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def telemetry_data(self) -> Dict[str, Any]:
        return {
            "model": self.agent_data.get("model") if self.agent_data else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    @classmethod

    def from_dict(cls, data: Mapping[str, Any]):
        if data is None or data.get("session_id") is None:
            print("AgentSession is missing session_id")
            return None
        return cls(session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            team_session_id=data.get("team_session_id"),
            user_id=data.get("user_id"),
            memory=data.get("memory"),
            agent_data=data.get("agent_data"),
            session_data=data.get("session_data"),
            extra_data=data.get("extra_data"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"))

@dataclass
class TeamSession:
    """Team Session that is stored in the database"""
    # Session UUID
    session_id: str
    # ID of the team session this team session is associated with (so for sub-teams)
    team_session_id: Optional[str] = None
    # ID of the team that this session is associated with
    team_id: Optional[str] = None
    # ID of the user interacting with this team
    user_id: Optional[str] = None
    # Team Memory
    memory: Optional[Dict[str, Any]] = None
    # Team Data: agent_id, name and model
    team_data: Optional[Dict[str, Any]] = None
    # Session Data: session_name, session_state, images, videos, audio
    session_data: Optional[Dict[str, Any]] = None
    # Extra Data stored with this agent
    extra_data: Optional[Dict[str, Any]] = None
    # The unix timestamp when this session was created
    created_at: Optional[int] = None
    # The unix timestamp when this session was last updated
    updated_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def telemetry_data(self) -> Dict[str, Any]:
        return {
            "model": self.team_data.get("model") if self.team_data else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    @classmethod

    def from_dict(cls, data: Mapping[str, Any]):
        if data is None or data.get("session_id") is None:
            print("AgentSession is missing session_id")
            return None
        return cls(session_id=data.get("session_id"),
            team_id=data.get("team_id"),
            team_session_id=data.get("team_session_id"),
            user_id=data.get("user_id"),
            memory=data.get("memory"),
            team_data=data.get("team_data"),
            session_data=data.get("session_data"),
            extra_data=data.get("extra_data"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"))

@dataclass
class WorkflowSession:
    """Workflow Session that is stored in the database"""
    # Session UUID
    session_id: str
    # ID of the user interacting with this agent
    user_id: Optional[str] = None
    # Agent Memory
    memory: Optional[Dict[str, Any]] = None
    # Session Data: session_name, session_state, images, videos, audio
    session_data: Optional[Dict[str, Any]] = None
    # Extra Data stored with this agent
    extra_data: Optional[Dict[str, Any]] = None
    # The unix timestamp when this session was created
    created_at: Optional[int] = None
    # The unix timestamp when this session was last updated
    updated_at: Optional[int] = None
    # ID of the workflow that this session is associated with
    workflow_id: Optional[str] = None
    # Workflow Data
    workflow_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def monitoring_data(self) -> Dict[str, Any]:
        return asdict(self)

    def telemetry_data(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    @classmethod

    def from_dict(cls, data: Mapping[str, Any]):
        if data is None or data.get("session_id") is None:
            print("WorkflowSession is missing session_id")
            return None
        return cls(session_id=data.get("session_id"),
            workflow_id=data.get("workflow_id"),
            user_id=data.get("user_id"),
            memory=data.get("memory"),
            workflow_data=data.get("workflow_data"),
            session_data=data.get("session_data"),
            extra_data=data.get("extra_data"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"))

Session = Union[AgentSession, TeamSession, WorkflowSession]

class Storage(ABC):

    def __init__(self, mode: Optional[Literal["agent", "team", "workflow"]] = "agent"):
        self._mode: Literal["agent", "team", "workflow"] = "agent" if mode is None else mode
    @property

    def mode(self) -> Literal["agent", "team", "workflow"]:
        """Get the mode of the storage."""
        return self._mode
    @mode.setter

    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
        """Set the mode of the storage."""
        self._mode = "agent" if value is None else value
    @abstractmethod

    def create(self) -> None:
        raise NotImplementedError
    @abstractmethod

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        raise NotImplementedError
    @abstractmethod

    def get_all_session_ids(self, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> List[str]:
        raise NotImplementedError
    @abstractmethod

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[Session]:
        raise NotImplementedError
    @abstractmethod

    def upsert(self, session: Session) -> Optional[Session]:
        raise NotImplementedError
    @abstractmethod

    def delete_session(self, session_id: Optional[str] = None):
        raise NotImplementedError
    @abstractmethod

    def drop(self) -> None:
        raise NotImplementedError
    @abstractmethod

    def upgrade_schema(self) -> None:
        raise NotImplementedError

class SingleStoreStorage(Storage):

    def __init__(self,
        table_name: str,
        schema: Optional[str] = "ai",
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        schema_version: int = 1,
        auto_upgrade_schema: bool = False,
        mode: Optional[Literal["agent", "team", "workflow"]] = "agent"):
        """
        This class provides Agent storage using a singlestore table.
        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url if provided
        Args:
            table_name (str): The name of the table to store the agent data.
            schema (Optional[str], optional): The schema of the table. Defaults to "ai".
            db_url (Optional[str], optional): The database URL. Defaults to None.
            db_engine (Optional[Engine], optional): The database engine. Defaults to None.
            schema_version (int, optional): The schema version. Defaults to 1.
            auto_upgrade_schema (bool, optional): Automatically upgrade the schema. Defaults to False.
            mode (Optional[Literal["agent", "team", "workflow"]], optional): The mode of the storage. Defaults to "agent".
        """
        super().__init__(mode)
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url, connect_args={"charset": "utf8mb4"})
        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")
        # Database attributes
        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)
        # Table schema version
        self.schema_version: int = schema_version
        # Automatically upgrade schema if True
        self.auto_upgrade_schema: bool = auto_upgrade_schema
        self._schema_up_to_date: bool = False
        # Database session
        self.SqlSession: sessionmaker[SqlSession] = sessionmaker(bind=self.db_engine)
        # Database table for storage
        self.table: Table = self.get_table()
    @property

    def mode(self) -> Literal["agent", "team", "workflow"]:
        """Get the mode of the storage."""
        return super().mode
    @mode.setter

    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
        """Set the mode and refresh the table if mode changes."""
        super(SingleStoreStorage, type(self)).mode.fset(self, value)
        if value is not None:
            self.table = self.get_table()

    def get_table_v1(self) -> Table:
        common_columns = [
            Column("session_id", mysql.TEXT, primary_key=True),
            Column("user_id", mysql.TEXT),
            Column("memory", mysql.JSON),
            Column("session_data", mysql.JSON),
            Column("extra_data", mysql.JSON),
            Column("created_at", mysql.BIGINT),
            Column("updated_at", mysql.BIGINT),
        ]
        specific_columns = []
        if self.mode == "agent":
            specific_columns = [
                Column("agent_id", mysql.TEXT),
                Column("team_session_id", mysql.TEXT, nullable=True),
                Column("agent_data", mysql.JSON),
            ]
        elif self.mode == "team":
            specific_columns = [
                Column("team_id", mysql.TEXT),
                Column("team_session_id", mysql.TEXT, nullable=True),
                Column("team_data", mysql.JSON),
            ]
        elif self.mode == "workflow":
            specific_columns = [
                Column("workflow_id", mysql.TEXT),
                Column("workflow_data", mysql.JSON),
            ]
        # Create table with all columns
        table = Table(self.table_name,
            self.metadata,
            *common_columns,
            *specific_columns,
            extend_existing=True,
            schema=self.schema)
        return table

    def get_table(self) -> Table:
        if self.schema_version == 1:
            return self.get_table_v1()
        else:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")

    def table_exists(self) -> bool:
        print(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            print(e)
            return False

    def create(self) -> None:
        self.table = self.get_table()
        if not self.table_exists():
            print(f"Creating table: {self.table_name}\n")
            self.table.create(self.db_engine)

    def _read(self, session: SqlSession, session_id: str, user_id: Optional[str] = None) -> Optional[Row[Any]]:
        stmt = select(self.table).where(self.table.c.session_id == session_id)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        try:
            return session.execute(stmt).first()
        except Exception as e:
            print(f"Exception reading from table: {e}")
            print(f"Table does not exist: {self.table.name}")
            print(f"Creating table: {self.table_name}")
            self.create()
        return None

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        with self.SqlSession.begin() as sess:
            existing_row: Optional[Row[Any]] = self._read(session=sess, session_id=session_id, user_id=user_id)
            if existing_row is not None:
                if self.mode == "agent":
                    return AgentSession.from_dict(existing_row._mapping)
                elif self.mode == "team":
                    return TeamSession.from_dict(existing_row._mapping)
                elif self.mode == "workflow":
                    return WorkflowSession.from_dict(existing_row._mapping)
            return None

    def get_all_session_ids(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[str]:
        session_ids: List[str] = []
        try:
            with self.SqlSession.begin() as sess:
                # get all session_ids for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if entity_id is not None:
                    if self.mode == "agent":
                        stmt = stmt.where(self.table.c.agent_id == entity_id)
                    elif self.mode == "team":
                        stmt = stmt.where(self.table.c.team_id == entity_id)
                    elif self.mode == "workflow":
                        stmt = stmt.where(self.table.c.workflow_id == entity_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None and row.session_id is not None:
                        session_ids.append(row.session_id)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        return session_ids

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[Session]:
        sessions: List[Session] = []
        try:
            with self.SqlSession.begin() as sess:
                # get all sessions for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if entity_id is not None:
                    if self.mode == "agent":
                        stmt = stmt.where(self.table.c.agent_id == entity_id)
                    elif self.mode == "team":
                        stmt = stmt.where(self.table.c.team_id == entity_id)
                    elif self.mode == "workflow":
                        stmt = stmt.where(self.table.c.workflow_id == entity_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row.session_id is not None:
                        if self.mode == "agent":
                            _agent_session = AgentSession.from_dict(row._mapping)
                            if _agent_session is not None:
                                sessions.append(_agent_session)
                        elif self.mode == "team":
                            _team_session = TeamSession.from_dict(row._mapping)
                            if _team_session is not None:
                                sessions.append(_team_session)
                        elif self.mode == "workflow":
                            _workflow_session = WorkflowSession.from_dict(row._mapping)
                            if _workflow_session is not None:
                                sessions.append(_workflow_session)
        except Exception:
            print(f"Table does not exist: {self.table.name}")
        return sessions

    def upgrade_schema(self) -> None:
        """
        Upgrade the schema to the latest version.
        Currently handles adding the team_session_id column for agent mode.
        """
        if not self.auto_upgrade_schema:
            print("Auto schema upgrade disabled. Skipping upgrade.")
            return
        try:
            if self.mode == "agent" and self.table_exists():
                with self.SqlSession() as sess:
                    # Check if team_session_id column exists
                    column_exists_query = text("""
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = :schema AND table_name = :table
                        AND column_name = 'team_session_id'
                        """)
                    column_exists = (sess.execute(column_exists_query, {"schema": self.schema, "table": self.table_name}).scalar()
                        is not None)
                    if not column_exists:
                        print(f"Adding 'team_session_id' column to {self.schema}.{self.table_name}")
                        alter_table_query = text(f"ALTER TABLE {self.schema}.{self.table_name} ADD COLUMN team_session_id TEXT")
                        sess.execute(alter_table_query)
                        sess.commit()
                        self._schema_up_to_date = True
                        print("Schema upgrade completed successfully")
        except Exception as e:
            print(f"Error during schema upgrade: {e}")
            raise

    def upsert(self, session: Session) -> Optional[Session]:
        """
        Create a new session if it does not exist, otherwise update the existing session.
        """
        # Perform schema upgrade if auto_upgrade_schema is enabled
        if self.auto_upgrade_schema and not self._schema_up_to_date:
            self.upgrade_schema()
        with self.SqlSession.begin() as sess:
            # Create an insert statement using MySQL's ON DUPLICATE KEY UPDATE syntax
            if self.mode == "agent":
                upsert_sql = text(f"""
                    INSERT INTO {self.schema}.{self.table_name}
                    (session_id, agent_id, team_session_id, user_id, memory, agent_data, session_data, extra_data, created_at, updated_at)
                    VALUES
                    (:session_id, :agent_id, :team_session_id, :user_id, :memory, :agent_data, :session_data, :extra_data, UNIX_TIMESTAMP(), NULL)
                    ON DUPLICATE KEY UPDATE
                        agent_id = VALUES(agent_id),
                        team_session_id = VALUES(team_session_id),
                        user_id = VALUES(user_id),
                        memory = VALUES(memory),
                        agent_data = VALUES(agent_data),
                        session_data = VALUES(session_data),
                        extra_data = VALUES(extra_data),
                        updated_at = UNIX_TIMESTAMP();
                    """)
            elif self.mode == "team":
                upsert_sql = text(f"""
                    INSERT INTO {self.schema}.{self.table_name}
                    (session_id, team_id, user_id, team_session_id, memory, team_data, session_data, extra_data, created_at, updated_at)
                    VALUES
                    (:session_id, :team_id, :user_id, :team_session_id, :memory, :team_data, :session_data, :extra_data, UNIX_TIMESTAMP(), NULL)
                    ON DUPLICATE KEY UPDATE
                        team_id = VALUES(team_id),
                        team_session_id = VALUES(team_session_id),
                        user_id = VALUES(user_id),
                        memory = VALUES(memory),
                        team_data = VALUES(team_data),
                        session_data = VALUES(session_data),
                        extra_data = VALUES(extra_data),
                        updated_at = UNIX_TIMESTAMP();
                    """)
            elif self.mode == "workflow":
                upsert_sql = text(f"""
                    INSERT INTO {self.schema}.{self.table_name}
                    (session_id, workflow_id, user_id, memory, workflow_data, session_data, extra_data, created_at, updated_at)
                    VALUES
                    (:session_id, :workflow_id, :user_id, :memory, :workflow_data, :session_data, :extra_data, UNIX_TIMESTAMP(), NULL)
                    ON DUPLICATE KEY UPDATE
                        workflow_id = VALUES(workflow_id),
                        user_id = VALUES(user_id),
                        memory = VALUES(memory),
                        workflow_data = VALUES(workflow_data),
                        session_data = VALUES(session_data),
                        extra_data = VALUES(extra_data),
                        updated_at = UNIX_TIMESTAMP();
                    """)
            try:
                if self.mode == "agent":
                    sess.execute(upsert_sql,
                        {
                            "session_id": session.session_id,
                            "agent_id": session.agent_id,
                            "team_session_id": session.team_session_id,
                            "user_id": session.user_id,
                            "memory": json.dumps(session.memory, ensure_ascii=False)
                            if session.memory is not None
                            else None,
                            "agent_data": json.dumps(session.agent_data, ensure_ascii=False)
                            if session.agent_data is not None
                            else None,
                            "session_data": json.dumps(session.session_data, ensure_ascii=False)
                            if session.session_data is not None
                            else None,
                            "extra_data": json.dumps(session.extra_data, ensure_ascii=False)
                            if session.extra_data is not None
                            else None,
                        })
                elif self.mode == "team":
                    sess.execute(upsert_sql,
                        {
                            "session_id": session.session_id,
                            "team_id": session.team_id,
                            "user_id": session.user_id,
                            "team_session_id": session.team_session_id,
                            "memory": json.dumps(session.memory, ensure_ascii=False)
                            if session.memory is not None
                            else None,
                            "team_data": json.dumps(session.team_data, ensure_ascii=False)
                            if session.team_data is not None
                            else None,
                            "session_data": json.dumps(session.session_data, ensure_ascii=False)
                            if session.session_data is not None
                            else None,
                            "extra_data": json.dumps(session.extra_data, ensure_ascii=False)
                            if session.extra_data is not None
                            else None,
                        })
                elif self.mode == "workflow":
                    sess.execute(upsert_sql,
                        {
                            "session_id": session.session_id,
                            "workflow_id": session.workflow_id,
                            "user_id": session.user_id,
                            "memory": json.dumps(session.memory, ensure_ascii=False)
                            if session.memory is not None
                            else None,
                            "workflow_data": json.dumps(session.workflow_data, ensure_ascii=False)
                            if session.workflow_data is not None
                            else None,
                            "session_data": json.dumps(session.session_data, ensure_ascii=False)
                            if session.session_data is not None
                            else None,
                            "extra_data": json.dumps(session.extra_data, ensure_ascii=False)
                            if session.extra_data is not None
                            else None,
                        })
            except Exception as e:
                # Create table and try again
                if not self.table_exists():
                    print(f"Table does not exist: {self.table.name}")
                    print("Creating table and retrying upsert")
                    self.create()
                    return self.upsert(session)
                else:
                    print(f"Exception upserting into table: {e}")
                    print("A table upgrade might be required, please review these docs for more information: https://agno.link/upgrade-schema")
                    return None
        return self.read(session_id=session.session_id)

    def delete_session(self, session_id: Optional[str] = None):
        if session_id is None:
            print("No session_id provided for deletion.")
            return
        with self.SqlSession() as sess, sess.begin():
            try:
                # Delete the session with the given session_id
                delete_stmt = self.table.delete().where(self.table.c.session_id == session_id)
                result = sess.execute(delete_stmt)
                if result.rowcount == 0:
                    print(f"No session found with session_id: {session_id}")
                else:
                    print(f"Successfully deleted session with session_id: {session_id}")
            except Exception as e:
                print(f"Error deleting session: {e}")
                raise

    def drop(self) -> None:
        if self.table_exists():
            print(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the SingleStoreAgentStorage instance, handling unpickleable attributes.
        Args:
            memo (dict): A dictionary of objects already copied during the current copying pass.
        Returns:
            SingleStoreStorage: A deep-copied instance of SingleStoreAgentStorage.
        """
        from copy import deepcopy
        # Create a new instance without calling __init__
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        # Deep copy attributes
        for k, v in self.__dict__.items():
            if k in {"metadata", "table"}:
                continue
            # Reuse db_engine and Session without copying
            elif k in {"db_engine", "SqlSession"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
        # Recreate metadata and table for the copied instance
        copied_obj.metadata = MetaData(schema=self.schema)
        copied_obj.table = copied_obj.get_table()
        return copied_obj

class SqliteStorage(Storage):

    def __init__(self,
        table_name: str,
        db_url: Optional[str] = None,
        db_file: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        schema_version: int = 1,
        auto_upgrade_schema: bool = False,
        mode: Optional[Literal["agent", "team", "workflow"]] = "agent"):
        """
        This class provides agent storage using a sqlite database.
        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url
            3. Use the db_file
            4. Create a new in-memory database
        Args:
            table_name: The name of the table to store Agent sessions.
            db_url: The database URL to connect to.
            db_file: The database file to connect to.
            db_engine: The SQLAlchemy database engine to use.
        """
        super().__init__(mode)
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
        # Table schema version
        self.schema_version: int = schema_version
        # Automatically upgrade schema if True
        self.auto_upgrade_schema: bool = auto_upgrade_schema
        self._schema_up_to_date: bool = False
        # Database session
        self.SqlSession: sessionmaker[SqlSession] = sessionmaker(bind=self.db_engine)
        # Database table for storage
        self.table: Table = self.get_table()
    @property

    def mode(self) -> Optional[Literal["agent", "team", "workflow"]]:
        """Get the mode of the storage."""
        return super().mode
    @mode.setter

    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
        """Set the mode and refresh the table if mode changes."""
        super(SqliteStorage, type(self)).mode.fset(self, value)
        if value is not None:
            self.table = self.get_table()

    def get_table_v1(self) -> Table:
        """
        Define the table schema for version 1.
        Returns:
            Table: SQLAlchemy Table object representing the schema.
        """
        common_columns = [
            Column("session_id", String, primary_key=True),
            Column("user_id", String, index=True),
            Column("memory", sqlite.JSON),
            Column("session_data", sqlite.JSON),
            Column("extra_data", sqlite.JSON),
            Column("created_at", sqlite.INTEGER, default=lambda: int(time.time())),
            Column("updated_at", sqlite.INTEGER, onupdate=lambda: int(time.time())),
        ]
        # Mode-specific columns
        specific_columns = []
        if self.mode == "agent":
            specific_columns = [
                Column("agent_id", String, index=True),
                Column("agent_data", sqlite.JSON),
                Column("team_session_id", String, index=True, nullable=True),
            ]
        elif self.mode == "team":
            specific_columns = [
                Column("team_id", String, index=True),
                Column("team_data", sqlite.JSON),
                Column("team_session_id", String, index=True, nullable=True),
            ]
        elif self.mode == "workflow":
            specific_columns = [
                Column("workflow_id", String, index=True),
                Column("workflow_data", sqlite.JSON),
            ]
        # Create table with all columns
        table = Table(self.table_name,
            self.metadata,
            *common_columns,
            *specific_columns,
            extend_existing=True,
            sqlite_autoincrement=True)
        return table

    def get_table(self) -> Table:
        """
        Get the table schema based on the schema version.
        Returns:
            Table: SQLAlchemy Table object for the current schema version.
        Raises:
            ValueError: If an unsupported schema version is specified.
        """
        if self.schema_version == 1:
            return self.get_table_v1()
        else:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")

    def table_exists(self) -> bool:
        """
        Check if the table exists in the database.
        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            # For SQLite, we need to check the sqlite_master table
            with self.SqlSession() as sess:
                result = sess.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                    {"table_name": self.table_name}).scalar()
                return result is not None
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False

    def create(self) -> None:
        """
        Create the table if it doesn't exist.
        """
        self.table = self.get_table()
        if not self.table_exists():
            print(f"Creating table: {self.table.name}")
            try:
                # First create the table without indexes
                table_without_indexes = Table(self.table_name,
                    MetaData(),
                    *[c.copy() for c in self.table.columns])
                table_without_indexes.create(self.db_engine, checkfirst=True)
                # Then create each index individually with error handling
                for idx in self.table.indexes:
                    try:
                        idx_name = idx.name
                        print(f"Creating index: {idx_name}")
                        # Check if index already exists using SQLite's schema table
                        with self.SqlSession() as sess:
                            exists_query = text("SELECT 1 FROM sqlite_master WHERE type='index' AND name=:index_name")
                            exists = sess.execute(exists_query, {"index_name": idx_name}).scalar() is not None
                        if not exists:
                            idx.create(self.db_engine)
                        else:
                            print(f"Index {idx_name} already exists, skipping creation")
                    except Exception as e:
                        # Log the error but continue with other indexes
                        print(f"Error creating index {idx.name}: {e}")
            except Exception as e:
                print(f"Error creating table: {e}")
                raise

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """
        Read a Session from the database.
        Args:
            session_id (str): ID of the session to read.
            user_id (Optional[str]): User ID to filter by. Defaults to None.
        Returns:
            Optional[Session]: Session object if found, None otherwise.
        """
        try:
            with self.SqlSession() as sess:
                stmt = select(self.table).where(self.table.c.session_id == session_id)
                if user_id:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                result = sess.execute(stmt).fetchone()
                if self.mode == "agent":
                    return AgentSession.from_dict(result._mapping) if result is not None else None
                elif self.mode == "team":
                    return TeamSession.from_dict(result._mapping) if result is not None else None
                elif self.mode == "workflow":
                    return WorkflowSession.from_dict(result._mapping) if result is not None else None
        except Exception as e:
            if "no such table" in str(e):
                print(f"Table does not exist: {self.table.name}")
                self.create()
            else:
                print(f"Exception reading from table: {e}")
        return None

    def get_all_session_ids(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[str]:
        """
        Get all session IDs, optionally filtered by user_id and/or entity_id.
        Args:
            user_id (Optional[str]): The ID of the user to filter by.
            entity_id (Optional[str]): The ID of the agent / workflow to filter by.
        Returns:
            List[str]: List of session IDs matching the criteria.
        """
        try:
            with self.SqlSession() as sess, sess.begin():
                # get all session_ids
                stmt = select(self.table.c.session_id)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if entity_id is not None:
                    if self.mode == "agent":
                        stmt = stmt.where(self.table.c.agent_id == entity_id)
                    elif self.mode == "team":
                        stmt = stmt.where(self.table.c.team_id == entity_id)
                    elif self.mode == "workflow":
                        stmt = stmt.where(self.table.c.workflow_id == entity_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                return [row[0] for row in rows] if rows is not None else []
        except Exception as e:
            if "no such table" in str(e):
                print(f"Table does not exist: {self.table.name}")
                self.create()
            else:
                print(f"Exception reading from table: {e}")
        return []

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[Session]:
        """
        Get all sessions, optionally filtered by user_id and/or entity_id.
        Args:
            user_id (Optional[str]): The ID of the user to filter by.
            entity_id (Optional[str]): The ID of the agent / workflow to filter by.
        Returns:
            List[Session]: List of Session objects matching the criteria.
        """
        try:
            with self.SqlSession() as sess, sess.begin():
                # get all sessions
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if entity_id is not None:
                    if self.mode == "agent":
                        stmt = stmt.where(self.table.c.agent_id == entity_id)
                    elif self.mode == "team":
                        stmt = stmt.where(self.table.c.team_id == entity_id)
                    elif self.mode == "workflow":
                        stmt = stmt.where(self.table.c.workflow_id == entity_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                if rows is not None:
                    if self.mode == "agent":
                        return [AgentSession.from_dict(row._mapping) for row in rows]
                    elif self.mode == "team":
                        return [TeamSession.from_dict(row._mapping) for row in rows]
                    elif self.mode == "workflow":
                        return [WorkflowSession.from_dict(row._mapping) for row in rows]
                else:
                    return []
        except Exception as e:
            if "no such table" in str(e):
                print(f"Table does not exist: {self.table.name}")
                self.create()
            else:
                print(f"Exception reading from table: {e}")
        return []

    def upgrade_schema(self) -> None:
        """
        Upgrade the schema of the storage table.
        Currently handles adding the team_session_id column for agent mode.
        """
        if not self.auto_upgrade_schema:
            print("Auto schema upgrade disabled. Skipping upgrade.")
            return
        try:
            if self.mode == "agent" and self.table_exists():
                with self.SqlSession() as sess:
                    # Check if team_session_id column exists using SQLite PRAGMA
                    column_exists_query = text(f"PRAGMA table_info({self.table_name})")
                    columns = sess.execute(column_exists_query).fetchall()
                    column_exists = any(col[1] == "team_session_id" for col in columns)
                    if not column_exists:
                        print(f"Adding 'team_session_id' column to {self.table_name}")
                        alter_table_query = text(f"ALTER TABLE {self.table_name} ADD COLUMN team_session_id TEXT")
                        sess.execute(alter_table_query)
                        sess.commit()
                        self._schema_up_to_date = True
                        print("Schema upgrade completed successfully")
        except Exception as e:
            print(f"Error during schema upgrade: {e}")
            raise

    def upsert(self, session: Session, create_and_retry: bool = True) -> Optional[Session]:
        """
        Insert or update a Session in the database.
        Args:
            session (Session): The session data to upsert.
            create_and_retry (bool): Retry upsert if table does not exist.
        Returns:
            Optional[Session]: The upserted Session, or None if operation failed.
        """
        # Perform schema upgrade if auto_upgrade_schema is enabled
        if self.auto_upgrade_schema and not self._schema_up_to_date:
            self.upgrade_schema()
        try:
            with self.SqlSession() as sess, sess.begin():
                if self.mode == "agent":
                    # Create an insert statement
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        agent_id=session.agent_id,
                        team_session_id=session.team_session_id,
                        user_id=session.user_id,
                        memory=session.memory,
                        agent_data=session.agent_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    # Define the upsert if the session_id already exists
                    # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(agent_id=session.agent_id,
                            team_session_id=session.team_session_id,
                            user_id=session.user_id,
                            memory=session.memory,
                            agent_data=session.agent_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())),  # The updated value for each column)
                elif self.mode == "team":
                    # Create an insert statement
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        team_id=session.team_id,
                        user_id=session.user_id,
                        team_session_id=session.team_session_id,
                        memory=session.memory,
                        team_data=session.team_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    # Define the upsert if the session_id already exists
                    # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(team_id=session.team_id,
                            user_id=session.user_id,
                            team_session_id=session.team_session_id,
                            memory=session.memory,
                            team_data=session.team_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())),  # The updated value for each column)
                elif self.mode == "workflow":
                    # Create an insert statement
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        workflow_id=session.workflow_id,
                        user_id=session.user_id,
                        memory=session.memory,
                        workflow_data=session.workflow_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    # Define the upsert if the session_id already exists
                    # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(workflow_id=session.workflow_id,
                            user_id=session.user_id,
                            memory=session.memory,
                            workflow_data=session.workflow_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())),  # The updated value for each column)
                sess.execute(stmt)
        except Exception as e:
            if create_and_retry and not self.table_exists():
                print(f"Table does not exist: {self.table.name}")
                print("Creating table and retrying upsert")
                self.create()
                return self.upsert(session, create_and_retry=False)
            else:
                print(f"Exception upserting into table: {e}")
                print("A table upgrade might be required, please review these docs for more information: https://agno.link/upgrade-schema")
                return None
        return self.read(session_id=session.session_id)

    def delete_session(self, session_id: Optional[str] = None):
        """
        Delete a workflow session from the database.
        Args:
            session_id (Optional[str]): The ID of the session to delete.
        Raises:
            ValueError: If session_id is not provided.
        """
        if session_id is None:
            print("No session_id provided for deletion.")
            return
        try:
            with self.SqlSession() as sess, sess.begin():
                # Delete the session with the given session_id
                delete_stmt = self.table.delete().where(self.table.c.session_id == session_id)
                result = sess.execute(delete_stmt)
                if result.rowcount == 0:
                    print(f"No session found with session_id: {session_id}")
                else:
                    print(f"Successfully deleted session with session_id: {session_id}")
        except Exception as e:
            print(f"Error deleting session: {e}")

    def drop(self) -> None:
        """
        Drop the table from the database if it exists.
        """
        if self.table_exists():
            print(f"Deleting table: {self.table_name}")
            # Drop with checkfirst=True to avoid errors if the table doesn't exist
            self.table.drop(self.db_engine, checkfirst=True)
            # Clear metadata to ensure indexes are recreated properly
            self.metadata = MetaData()
            self.table = self.get_table()

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the SqliteAgentStorage instance, handling unpickleable attributes.
        Args:
            memo (dict): A dictionary of objects already copied during the current copying pass.
        Returns:
            SqliteStorage: A deep-copied instance of SqliteAgentStorage.
        """
        from copy import deepcopy
        # Create a new instance without calling __init__
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        # Deep copy attributes
        for k, v in self.__dict__.items():
            if k in {"metadata", "table", "inspector"}:
                continue
            # Reuse db_engine and Session without copying
            elif k in {"db_engine", "SqlSession"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
        # Recreate metadata and table for the copied instance
        copied_obj.metadata = MetaData()
        copied_obj.inspector = inspect(copied_obj.db_engine)
        copied_obj.table = copied_obj.get_table()
        return copied_obj

class JsonStorage(Storage):

    def __init__(self, dir_path: Union[str, Path], mode: Optional[Literal["agent", "team", "workflow"]] = "agent"):
        super().__init__(mode)
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def serialize(self, data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, indent=4)

    def deserialize(self, data: str) -> dict:
        return json.loads(data)

    def create(self) -> None:
        """Create the storage if it doesn't exist."""
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """Read an AgentSession from storage."""
        try:
            with open(self.dir_path / f"{session_id}.json", "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id and data["user_id"] != user_id:
                    return None
                if self.mode == "agent":
                    return AgentSession.from_dict(data)
                elif self.mode == "team":
                    return TeamSession.from_dict(data)
                elif self.mode == "workflow":
                    return WorkflowSession.from_dict(data)
        except FileNotFoundError:
            return None

    def get_all_session_ids(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[str]:
        """Get all session IDs, optionally filtered by user_id and/or entity_id."""
        session_ids = []
        for file in self.dir_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id or entity_id:
                    if user_id and entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id and data["user_id"] == user_id:
                            session_ids.append(data["session_id"])
                        elif self.mode == "team" and data["team_id"] == entity_id and data["user_id"] == user_id:
                            session_ids.append(data["session_id"])
                        elif (self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id):
                            session_ids.append(data["session_id"])
                    elif user_id and data["user_id"] == user_id:
                        session_ids.append(data["session_id"])
                    elif entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id:
                            session_ids.append(data["session_id"])
                        elif self.mode == "team" and data["team_id"] == entity_id:
                            session_ids.append(data["session_id"])
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id:
                            session_ids.append(data["session_id"])
                else:
                    # No filters applied, add all session_ids
                    session_ids.append(data["session_id"])
        return session_ids

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[Session]:
        """Get all sessions, optionally filtered by user_id and/or entity_id."""
        sessions: List[Session] = []
        for file in self.dir_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id or entity_id:
                    _session: Optional[Session] = None
                    if user_id and entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id and data["user_id"] == user_id:
                            _session = AgentSession.from_dict(data)
                        elif self.mode == "team" and data["team_id"] == entity_id and data["user_id"] == user_id:
                            _session = TeamSession.from_dict(data)
                        elif (self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id):
                            _session = WorkflowSession.from_dict(data)
                    elif user_id and data["user_id"] == user_id:
                        if self.mode == "agent":
                            _session = AgentSession.from_dict(data)
                        elif self.mode == "team":
                            _session = TeamSession.from_dict(data)
                        elif self.mode == "workflow":
                            _session = WorkflowSession.from_dict(data)
                    elif entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id:
                            _session = AgentSession.from_dict(data)
                        elif self.mode == "team" and data["team_id"] == entity_id:
                            _session = TeamSession.from_dict(data)
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id:
                            _session = WorkflowSession.from_dict(data)
                    if _session:
                        sessions.append(_session)
                else:
                    # No filters applied, add all sessions
                    if self.mode == "agent":
                        _session = AgentSession.from_dict(data)
                    elif self.mode == "team":
                        _session = TeamSession.from_dict(data)
                    elif self.mode == "workflow":
                        _session = WorkflowSession.from_dict(data)
                    if _session:
                        sessions.append(_session)
        return sessions

    def upsert(self, session: Session) -> Optional[Session]:
        """Insert or update a Session in storage."""
        try:
            data = asdict(session)
            data["updated_at"] = int(time.time())
            if "created_at" not in data:
                data["created_at"] = data["updated_at"]
            with open(self.dir_path / f"{session.session_id}.json", "w", encoding="utf-8") as f:
                f.write(self.serialize(data))
            return session
        except Exception as e:
            print(f"Error upserting session: {e}")
            return None

    def delete_session(self, session_id: Optional[str] = None):
        """Delete a session from storage."""
        if session_id is None:
            return
        try:
            (self.dir_path / f"{session_id}.json").unlink(missing_ok=True)
        except Exception as e:
            print(f"Error deleting session: {e}")

    def drop(self) -> None:
        """Drop all sessions from storage."""
        for file in self.dir_path.glob("*.json"):
            file.unlink()

    def upgrade_schema(self) -> None:
        """Upgrade the schema of the storage."""
        pass
