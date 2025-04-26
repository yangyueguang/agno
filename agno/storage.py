import time
import json
from pathlib import Path
from sqlalchemy.dialects import sqlite, mysql
from sqlalchemy.types import String
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.engine.row import Row
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import sessionmaker, Session as SqlSession
from sqlalchemy.schema import Column, MetaData, Table
from sqlalchemy.sql.expression import select, text
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Any, Dict, Mapping, Optional, Union

@dataclass
class AgentSession:
    session_id: str
    user_id: Optional[str] = None
    team_session_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    extra_data: Optional[Dict[str, Any]] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    agent_id: Optional[str] = None
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
    session_id: str
    team_session_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None
    team_data: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    extra_data: Optional[Dict[str, Any]] = None
    created_at: Optional[int] = None
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
    session_id: str
    user_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    extra_data: Optional[Dict[str, Any]] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    workflow_id: Optional[str] = None
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
        return self._mode

    @mode.setter
    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
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
        super().__init__(mode)
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url, connect_args={"charset": "utf8mb4"})
        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")
        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)
        self.schema_version: int = schema_version
        self.auto_upgrade_schema: bool = auto_upgrade_schema
        self._schema_up_to_date: bool = False
        self.SqlSession: sessionmaker[SqlSession] = sessionmaker(bind=self.db_engine)
        self.table: Table = self.get_table()

    @property
    def mode(self) -> Literal["agent", "team", "workflow"]:
        return super().mode

    @mode.setter
    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
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
                stmt = stmt.order_by(self.table.c.created_at.desc())
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
                stmt = stmt.order_by(self.table.c.created_at.desc())
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
        if not self.auto_upgrade_schema:
            print("Auto schema upgrade disabled. Skipping upgrade.")
            return
        try:
            if self.mode == "agent" and self.table_exists():
                with self.SqlSession() as sess:
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
        if self.auto_upgrade_schema and not self._schema_up_to_date:
            self.upgrade_schema()
        with self.SqlSession.begin() as sess:
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
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        for k, v in self.__dict__.items():
            if k in {"metadata", "table"}:
                continue
            elif k in {"db_engine", "SqlSession"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
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
        super().__init__(mode)
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            db_path = Path(db_file).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{db_path}")
        else:
            _engine = create_engine("sqlite://")
        if _engine is None:
            raise ValueError("Must provide either db_url, db_file or db_engine")
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()
        self.inspector = inspect(self.db_engine)
        self.schema_version: int = schema_version
        self.auto_upgrade_schema: bool = auto_upgrade_schema
        self._schema_up_to_date: bool = False
        self.SqlSession: sessionmaker[SqlSession] = sessionmaker(bind=self.db_engine)
        self.table: Table = self.get_table()

    @property
    def mode(self) -> Optional[Literal["agent", "team", "workflow"]]:
        return super().mode

    @mode.setter
    def mode(self, value: Optional[Literal["agent", "team", "workflow"]]) -> None:
        super(SqliteStorage, type(self)).mode.fset(self, value)
        if value is not None:
            self.table = self.get_table()

    def get_table_v1(self) -> Table:
        common_columns = [
            Column("session_id", String, primary_key=True),
            Column("user_id", String, index=True),
            Column("memory", sqlite.JSON),
            Column("session_data", sqlite.JSON),
            Column("extra_data", sqlite.JSON),
            Column("created_at", sqlite.INTEGER, default=lambda: int(time.time())),
            Column("updated_at", sqlite.INTEGER, onupdate=lambda: int(time.time())),
        ]
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
        table = Table(self.table_name,
            self.metadata,
            *common_columns,
            *specific_columns,
            extend_existing=True,
            sqlite_autoincrement=True)
        return table

    def get_table(self) -> Table:
        if self.schema_version == 1:
            return self.get_table_v1()
        else:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")

    def table_exists(self) -> bool:
        try:
            with self.SqlSession() as sess:
                result = sess.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                    {"table_name": self.table_name}).scalar()
                return result is not None
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False

    def create(self) -> None:
        self.table = self.get_table()
        if not self.table_exists():
            print(f"Creating table: {self.table.name}")
            try:
                table_without_indexes = Table(self.table_name,
                    MetaData(),
                    *[c.copy() for c in self.table.columns])
                table_without_indexes.create(self.db_engine, checkfirst=True)
                for idx in self.table.indexes:
                    try:
                        idx_name = idx.name
                        print(f"Creating index: {idx_name}")
                        with self.SqlSession() as sess:
                            exists_query = text("SELECT 1 FROM sqlite_master WHERE type='index' AND name=:index_name")
                            exists = sess.execute(exists_query, {"index_name": idx_name}).scalar() is not None
                        if not exists:
                            idx.create(self.db_engine)
                        else:
                            print(f"Index {idx_name} already exists, skipping creation")
                    except Exception as e:
                        print(f"Error creating index {idx.name}: {e}")
            except Exception as e:
                print(f"Error creating table: {e}")
                raise

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
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
        try:
            with self.SqlSession() as sess, sess.begin():
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
                stmt = stmt.order_by(self.table.c.created_at.desc())
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
        try:
            with self.SqlSession() as sess, sess.begin():
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
                stmt = stmt.order_by(self.table.c.created_at.desc())
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
        if not self.auto_upgrade_schema:
            print("Auto schema upgrade disabled. Skipping upgrade.")
            return
        try:
            if self.mode == "agent" and self.table_exists():
                with self.SqlSession() as sess:
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
        if self.auto_upgrade_schema and not self._schema_up_to_date:
            self.upgrade_schema()
        try:
            with self.SqlSession() as sess, sess.begin():
                if self.mode == "agent":
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        agent_id=session.agent_id,
                        team_session_id=session.team_session_id,
                        user_id=session.user_id,
                        memory=session.memory,
                        agent_data=session.agent_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(agent_id=session.agent_id,
                            team_session_id=session.team_session_id,
                            user_id=session.user_id,
                            memory=session.memory,
                            agent_data=session.agent_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())))
                elif self.mode == "team":
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        team_id=session.team_id,
                        user_id=session.user_id,
                        team_session_id=session.team_session_id,
                        memory=session.memory,
                        team_data=session.team_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(team_id=session.team_id,
                            user_id=session.user_id,
                            team_session_id=session.team_session_id,
                            memory=session.memory,
                            team_data=session.team_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())))
                elif self.mode == "workflow":
                    stmt = sqlite.insert(self.table).values(session_id=session.session_id,
                        workflow_id=session.workflow_id,
                        user_id=session.user_id,
                        memory=session.memory,
                        workflow_data=session.workflow_data,
                        session_data=session.session_data,
                        extra_data=session.extra_data)
                    stmt = stmt.on_conflict_do_update(index_elements=["session_id"],
                        set_=dict(workflow_id=session.workflow_id,
                            user_id=session.user_id,
                            memory=session.memory,
                            workflow_data=session.workflow_data,
                            session_data=session.session_data,
                            extra_data=session.extra_data,
                            updated_at=int(time.time())))
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
        if session_id is None:
            print("No session_id provided for deletion.")
            return
        try:
            with self.SqlSession() as sess, sess.begin():
                delete_stmt = self.table.delete().where(self.table.c.session_id == session_id)
                result = sess.execute(delete_stmt)
                if result.rowcount == 0:
                    print(f"No session found with session_id: {session_id}")
                else:
                    print(f"Successfully deleted session with session_id: {session_id}")
        except Exception as e:
            print(f"Error deleting session: {e}")

    def drop(self) -> None:
        if self.table_exists():
            print(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine, checkfirst=True)
            self.metadata = MetaData()
            self.table = self.get_table()

    def __deepcopy__(self, memo):
        from copy import deepcopy
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        for k, v in self.__dict__.items():
            if k in {"metadata", "table", "inspector"}:
                continue
            elif k in {"db_engine", "SqlSession"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
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
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
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
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id:
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
                    session_ids.append(data["session_id"])
        return session_ids

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[Session]:
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
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id:
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
        if session_id is None:
            return
        try:
            (self.dir_path / f"{session_id}.json").unlink(missing_ok=True)
        except Exception as e:
            print(f"Error deleting session: {e}")

    def drop(self) -> None:
        for file in self.dir_path.glob("*.json"):
            file.unlink()

    def upgrade_schema(self) -> None:
        pass
