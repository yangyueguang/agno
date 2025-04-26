from hashlib import md5
from chromadb import Client as ChromaDbClient, PersistentClient as PersistentChromaDbClient
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import GetResult, IncludeEnum, QueryResult
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from agno.reader import Document, Embedder, OllamaEmbedder


class SearchType(str, Enum):
    vector = 'vector'
    keyword = 'keyword'
    hybrid = 'hybrid'


class Distance(str, Enum):
    cosine = 'cosine'
    l2 = 'l2'
    max_inner_product = 'max_inner_product'


class VectorDb(ABC):
    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def async_create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def doc_exists(self, document: Document) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def async_doc_exists(self, document: Document) -> bool:
        raise NotImplementedError

    @abstractmethod
    def name_exists(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def async_name_exists(self, name: str) -> bool:
        raise NotImplementedError

    def id_exists(self, id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def upsert_available(self) -> bool:
        return False

    @abstractmethod
    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    async def async_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        raise NotImplementedError

    def vector_search(self, query: str, limit: int = 5) -> List[Document]:
        raise NotImplementedError

    def keyword_search(self, query: str, limit: int = 5) -> List[Document]:
        raise NotImplementedError

    def hybrid_search(self, query: str, limit: int = 5) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def drop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def async_drop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def async_exists(self) -> bool:
        raise NotImplementedError

    def optimize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> bool:
        raise NotImplementedError


class ChromaDb(VectorDb):
    def __init__(self, collection: str, embedder: Optional[Embedder] = None, distance: Distance = Distance.cosine, path: str = 'tmp/chromadb', persistent_client: bool = False, reranker = None, **kwargs):
        self.collection_name: str = collection
        if embedder is None:
            embedder = OllamaEmbedder()
        self.embedder: Embedder = embedder
        self.distance: Distance = distance
        self._client: Optional[ClientAPI] = None
        self._collection: Optional[Collection] = None
        self.persistent_client: bool = persistent_client
        self.path: str = path
        self.reranker = reranker
        self.kwargs = kwargs

    @property
    def client(self) -> ClientAPI:
        if self._client is None:
            if not self.persistent_client:
                print('Creating Chroma Client')
                self._client = ChromaDbClient(**self.kwargs)
            elif self.persistent_client:
                print('Creating Persistent Chroma Client')
                self._client = PersistentChromaDbClient(path=self.path, **self.kwargs)
        return self._client

    def create(self) -> None:
        if self.exists():
            print(f'Collection already exists: {self.collection_name}')
            self._collection = self.client.get_collection(name=self.collection_name)
        else:
            print(f'Creating collection: {self.collection_name}')
            self._collection = self.client.create_collection(name=self.collection_name, metadata={'hnsw:space': self.distance.value})

    def doc_exists(self, document: Document) -> bool:
        if not self.client:
            print('Client not initialized')
            return False
        try:
            collection: Collection = self.client.get_collection(name=self.collection_name)
            collection_data: GetResult = collection.get(include=[IncludeEnum.documents])
            existing_documents = collection_data.get('documents', [])
            cleaned_content = document.content.replace('\x00', '\ufffd')
            if cleaned_content in existing_documents:
                return True
        except Exception as e:
            print(f'Document does not exist: {e}')
        return False

    def name_exists(self, name: str) -> bool:
        if self.client:
            try:
                collections: Collection = self.client.get_collection(name=self.collection_name)
                for collection in collections:
                    if name in collection:
                        return True
            except Exception as e:
                print(f'Document with given name does not exist: {e}')
        return False

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        print(f'Inserting {len(documents)} documents')
        ids: List = []
        docs: List = []
        docs_embeddings: List = []
        docs_metadata: List = []
        if not self._collection:
            self._collection = self.client.get_collection(name=self.collection_name)
        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace('\x00', '\ufffd')
            doc_id = md5(cleaned_content.encode()).hexdigest()
            docs_embeddings.append(document.embedding)
            docs.append(cleaned_content)
            ids.append(doc_id)
            docs_metadata.append(document.meta_data)
            print(f'Inserted document: {document.id} | {document.name} | {document.meta_data}')
        if self._collection is None:
            print('Collection does not exist')
        else:
            if len(docs) > 0:
                self._collection.add(ids=ids, embeddings=docs_embeddings, documents=docs, metadatas=docs_metadata)
                print(f'Committed {len(docs)} documents')

    def upsert_available(self) -> bool:
        return True

    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        print(f'Upserting {len(documents)} documents')
        ids: List = []
        docs: List = []
        docs_embeddings: List = []
        docs_metadata: List = []
        if not self._collection:
            self._collection = self.client.get_collection(name=self.collection_name)
        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace('\x00', '\ufffd')
            doc_id = md5(cleaned_content.encode()).hexdigest()
            docs_embeddings.append(document.embedding)
            docs.append(cleaned_content)
            ids.append(doc_id)
            docs_metadata.append(document.meta_data)
            print(f'Upserted document: {document.id} | {document.name} | {document.meta_data}')
        if self._collection is None:
            print('Collection does not exist')
        else:
            if len(docs) > 0:
                self._collection.upsert(ids=ids, embeddings=docs_embeddings, documents=docs, metadatas=docs_metadata)
                print(f'Committed {len(docs)} documents')

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            print(f'Error getting embedding for Query: {query}')
            return []
        if not self._collection:
            self._collection = self.client.get_collection(name=self.collection_name)
        result: QueryResult = self._collection.query(query_embeddings=query_embedding, n_results=limit, include=['metadatas', 'documents', 'embeddings', 'distances', 'uris'])
        search_results: List[Document] = []
        ids = result.get('ids', [[]])[0]
        metadata = result.get('metadatas', [{}])[0]
        documents = result.get('documents', [[]])[0]
        embeddings = result.get('embeddings')[0]
        embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
        distances = result.get('distances', [[]])[0]
        for idx, distance in enumerate(distances):
            metadata[idx]['distances'] = distance
        try:
            for idx, (id_, metadata, document) in enumerate(zip(ids, metadata, documents)):
                search_results.append(Document(id=id_, meta_data=metadata, content=document, embedding=embeddings[idx]))
        except Exception as e:
            print(f'Error building search results: {e}')
        if self.reranker:
            search_results = self.reranker.rerank(query=query, documents=search_results)
        return search_results

    def drop(self) -> None:
        if self.exists():
            print(f'Deleting collection: {self.collection_name}')
            self.client.delete_collection(name=self.collection_name)

    def exists(self) -> bool:
        try:
            self.client.get_collection(name=self.collection_name)
            return True
        except Exception as e:
            print(f'Collection does not exist: {e}')
        return False

    def get_count(self) -> int:
        if self.exists():
            try:
                collection: Collection = self.client.get_collection(name=self.collection_name)
                return collection.count()
            except Exception as e:
                print(f'Error getting count: {e}')
        return 0

    def optimize(self) -> None:
        raise NotImplementedError

    def delete(self) -> bool:
        try:
            self.client.delete_collection(name=self.collection_name)
            return True
        except Exception as e:
            print(f'Error clearing collection: {e}')
            return False

    async def async_create(self) -> None:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_doc_exists(self, document: Document) -> bool:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_drop(self) -> None:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_exists(self) -> bool:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')

    async def async_name_exists(self, name: str) -> bool:
        raise NotImplementedError(f'Async not supported on {self.__class__.__name__}.')
