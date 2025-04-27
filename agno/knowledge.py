from agno.reader import *
from pathlib import Path
from agno.vectordb import VectorDb
from typing import AsyncIterator, Iterator, List, Union, Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class AgentKnowledge(BaseModel):
    reader: Optional[Reader] = None
    vector_db: Optional[VectorDb] = None
    num_documents: int = 5
    optimize_on: Optional[int] = 1000
    chunking_strategy: ChunkingStrategy = ChunkingStrategy()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        raise NotImplementedError

    @property
    def async_document_lists(self) -> AsyncIterator[List[Document]]:
        raise NotImplementedError

    def search(self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        try:
            if self.vector_db is None:
                print('未提供矢量数据库')
                return []
            _num_documents = num_documents or self.num_documents
            print(f'Getting {_num_documents} relevant documents for query: {query}')
            return self.vector_db.search(query=query, limit=_num_documents, filters=filters)
        except Exception as e:
            print(f'搜索文档时出错: {e}')
            return []

    async def async_search(self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        try:
            if self.vector_db is None:
                print('未提供矢量数据库')
                return []
            _num_documents = num_documents or self.num_documents
            print(f'Getting {_num_documents} relevant documents for query: {query}')
            try:
                return await self.vector_db.async_search(query=query, limit=_num_documents, filters=filters)
            except NotImplementedError:
                print('Vector db does not support async search')
                return self.search(query=query, num_documents=_num_documents, filters=filters)
        except Exception as e:
            print(f'搜索文档时出错: {e}')
            return []

    def load(self, recreate: bool = False, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        if self.vector_db is None:
            print('No vector db provided')
            return
        if recreate:
            print('Dropping collection')
            self.vector_db.drop()
        if not self.vector_db.exists():
            print('Creating collection')
            self.vector_db.create()
        print('Loading knowledge base')
        num_documents = 0
        for document_list in self.document_lists:
            documents_to_load = document_list
            if upsert and self.vector_db.upsert_available():
                self.vector_db.upsert(documents=documents_to_load, filters=filters)
            else:
                if skip_existing:
                    seen_content = set()
                    documents_to_load = []
                    for doc in document_list:
                        if doc.content not in seen_content and not self.vector_db.doc_exists(doc):
                            seen_content.add(doc.content)
                            documents_to_load.append(doc)
                self.vector_db.insert(documents=documents_to_load, filters=filters)
            num_documents += len(documents_to_load)
            print(f'Added {len(documents_to_load)} documents to knowledge base')

    async def aload(self, recreate: bool = False, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        if self.vector_db is None:
            print('No vector db provided')
            return
        if recreate:
            print('Dropping collection')
            await self.vector_db.async_drop()
        if not await self.vector_db.async_exists():
            print('Creating collection')
            await self.vector_db.async_create()
        print('Loading knowledge base')
        num_documents = 0
        async for document_list in self.async_document_lists:
            documents_to_load = document_list
            if upsert and self.vector_db.upsert_available():
                await self.vector_db.async_upsert(documents=documents_to_load, filters=filters)
            else:
                if skip_existing:
                    seen_content = set()
                    documents_to_load = []
                    for doc in document_list:
                        if doc.content not in seen_content and not (await self.vector_db.async_doc_exists(doc)):
                            seen_content.add(doc.content)
                            documents_to_load.append(doc)
                await self.vector_db.async_insert(documents=documents_to_load, filters=filters)
            num_documents += len(documents_to_load)
            print(f'Added {len(documents_to_load)} documents to knowledge base')

    def load_documents(self, documents: List[Document], upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        print('Loading knowledge base')
        if self.vector_db is None:
            print('No vector db provided')
            return
        print('Creating collection')
        self.vector_db.create()
        if upsert and self.vector_db.upsert_available():
            self.vector_db.upsert(documents=documents, filters=filters)
            print(f'Loaded {len(documents)} documents to knowledge base')
        else:
            documents_to_load = ([document for document in documents if not self.vector_db.doc_exists(document)]
                if skip_existing
                else documents)
            if len(documents_to_load) > 0:
                self.vector_db.insert(documents=documents_to_load, filters=filters)
                print(f'Loaded {len(documents_to_load)} documents to knowledge base')
            else:
                print('No new documents to load')

    async def async_load_documents(self, documents: List[Document], upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        print('Loading knowledge base')
        if self.vector_db is None:
            print('No vector db provided')
            return
        print('Creating collection')
        try:
            await self.vector_db.async_create()
        except NotImplementedError:
            print('Vector db does not support async create')
            self.vector_db.create()
        if upsert and self.vector_db.upsert_available():
            try:
                await self.vector_db.async_upsert(documents=documents, filters=filters)
            except NotImplementedError:
                print('Vector db does not support async upsert')
                self.vector_db.upsert(documents=documents, filters=filters)
            print(f'Loaded {len(documents)} documents to knowledge base')
        else:
            if skip_existing:
                try:
                    existence_checks = await asyncio.gather(*[self.vector_db.async_doc_exists(document) for document in documents], return_exceptions=True)
                    documents_to_load = [
                        doc
                        for doc, exists in zip(documents, existence_checks)
                        if not (isinstance(exists, bool) and exists)
                    ]
                except NotImplementedError:
                    print('Vector db does not support async doc_exists')
                    documents_to_load = [document for document in documents if not self.vector_db.doc_exists(document)]
            else:
                documents_to_load = documents
            if len(documents_to_load) > 0:
                try:
                    await self.vector_db.async_insert(documents=documents_to_load, filters=filters)
                except NotImplementedError:
                    print('Vector db does not support async insert')
                    self.vector_db.insert(documents=documents_to_load, filters=filters)
                print(f'Loaded {len(documents_to_load)} documents to knowledge base')
            else:
                print('No new documents to load')

    def load_document(self, document: Document, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        self.load_documents(documents=[document], upsert=upsert, skip_existing=skip_existing, filters=filters)

    async def async_load_document(self, document: Document, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        await self.async_load_documents(documents=[document], upsert=upsert, skip_existing=skip_existing, filters=filters)

    def load_dict(self, document: Dict[str, Any], upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        self.load_documents(documents=[Document.from_dict(document)], upsert=upsert, skip_existing=skip_existing, filters=filters)

    def load_json(self, document: str, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        self.load_documents(documents=[Document.from_json(document)], upsert=upsert, skip_existing=skip_existing, filters=filters)

    def load_text(self, text: str, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        self.load_documents(documents=[Document(content=text)], upsert=upsert, skip_existing=skip_existing, filters=filters)

    def exists(self) -> bool:
        if self.vector_db is None:
            print('No vector db provided')
            return False
        return self.vector_db.exists()

    def delete(self) -> bool:
        if self.vector_db is None:
            print('No vector db available')
            return True
        return self.vector_db.delete()


class CombinedKnowledgeBase(AgentKnowledge):
    sources: List[AgentKnowledge] = []

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        for kb in self.sources:
            print(f'Loading documents from {kb.__class__.__name__}')
            yield from kb.document_lists


class CSVKnowledgeBase(AgentKnowledge):
    path: Union[str, Path]
    exclude_files: List[str] = Field(default_factory=list)
    reader: CSVReader = CSVReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        _csv_path: Path = Path(self.path) if isinstance(self.path, str) else self.path
        if _csv_path.exists() and _csv_path.is_dir():
            for _csv in _csv_path.glob('**/*.csv'):
                if _csv.name in self.exclude_files:
                    continue
                yield self.reader.read(file=_csv)
        elif _csv_path.exists() and _csv_path.is_file() and _csv_path.suffix == '.csv':
            if _csv_path.name in self.exclude_files:
                return
            yield self.reader.read(file=_csv_path)

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        _csv_path: Path = Path(self.path) if isinstance(self.path, str) else self.path
        if _csv_path.exists() and _csv_path.is_dir():
            for _csv in _csv_path.glob('**/*.csv'):
                if _csv.name in self.exclude_files:
                    continue
                yield await self.reader.async_read(file=_csv)
        elif _csv_path.exists() and _csv_path.is_file() and _csv_path.suffix == '.csv':
            if _csv_path.name in self.exclude_files:
                return
            yield await self.reader.async_read(file=_csv_path)


class CSVUrlKnowledgeBase(AgentKnowledge):
    urls: List[str]
    reader: CSVUrlReader = CSVUrlReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        for url in self.urls:
            if url.endswith('.csv'):
                yield self.reader.read(url=url)
            else:
                print(f'Unsupported URL: {url}')

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        for url in self.urls:
            if url.endswith('.csv'):
                yield await self.reader.async_read(url=url)
            else:
                print(f'Unsupported URL: {url}')


class DocxKnowledgeBase(AgentKnowledge):
    path: Union[str, Path]
    formats: List[str] = ['.doc', '.docx']
    reader: DocxReader = DocxReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        _file_path: Path = Path(self.path) if isinstance(self.path, str) else self.path
        if _file_path.exists() and _file_path.is_dir():
            for _file in _file_path.glob('**/*'):
                if _file.suffix in self.formats:
                    yield self.reader.read(file=_file)
        elif _file_path.exists() and _file_path.is_file() and _file_path.suffix in self.formats:
            yield self.reader.read(file=_file_path)


class LlamaIndexKnowledgeBase(AgentKnowledge):
    retriever: BaseRetriever
    loader: Optional[Callable] = None

    def search(self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if not isinstance(self.retriever, BaseRetriever):
            raise ValueError(f'Retriever is not of type BaseRetriever: {self.retriever}')
        lc_documents: List[NodeWithScore] = self.retriever.retrieve(query)
        if num_documents is not None:
            lc_documents = lc_documents[:num_documents]
        documents = []
        for lc_doc in lc_documents:
            documents.append(Document(content=lc_doc.text, meta_data=lc_doc.metadata))
        return documents

    def load(self, recreate: bool = False, upsert: bool = True, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        if self.loader is None:
            print('No loader provided for LlamaIndexKnowledgeBase')
            return
        self.loader()

    def exists(self) -> bool:
        print('LlamaIndexKnowledgeBase.exists() not supported - please check the vectorstore manually.')
        return True


class PDFUrlKnowledgeBase(AgentKnowledge):
    urls: List[str] = []
    reader: Union[PDFUrlReader, PDFUrlImageReader] = PDFUrlReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        for url in self.urls:
            if url.endswith('.pdf'):
                yield self.reader.read(url=url)
            else:
                print(f'Unsupported URL: {url}')

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        for url in self.urls:
            if url.endswith('.pdf'):
                yield await self.reader.async_read(url=url)
            else:
                print(f'Unsupported URL: {url}')


class TextKnowledgeBase(AgentKnowledge):
    path: Union[str, Path]
    formats: List[str] = ['.txt']
    reader: TextReader = TextReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        _file_path: Path = Path(self.path) if isinstance(self.path, str) else self.path
        if _file_path.exists() and _file_path.is_dir():
            for _file in _file_path.glob('**/*'):
                if _file.suffix in self.formats:
                    yield self.reader.read(file=_file)
        elif _file_path.exists() and _file_path.is_file() and _file_path.suffix in self.formats:
            yield self.reader.read(file=_file_path)

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        _file_path: Path = Path(self.path) if isinstance(self.path, str) else self.path
        if _file_path.exists() and _file_path.is_dir():
            for _file in _file_path.glob('**/*'):
                if _file.suffix in self.formats:
                    yield await self.reader.async_read(file=_file)
        elif _file_path.exists() and _file_path.is_file() and _file_path.suffix in self.formats:
            yield await self.reader.async_read(file=_file_path)


class UrlKnowledge(AgentKnowledge):
    urls: List[str] = []
    reader: URLReader = URLReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        for url in self.urls:
            try:
                yield self.reader.read(url=url)
            except Exception as e:
                print(f'Error reading URL {url}: {str(e)}')


class WebsiteKnowledgeBase(AgentKnowledge):
    urls: List[str] = []
    reader: Optional[WebsiteReader] = WebsiteReader(max_depth=3, max_links=10)
    max_depth: int = 3
    max_links: int = 10

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        if self.reader is not None:
            for _url in self.urls:
                yield self.reader.read(url=_url)

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        if self.reader is not None:
            for _url in self.urls:
                yield await self.reader.async_read(url=_url)

    def load(self, recreate: bool = False, upsert: bool = True, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        if self.vector_db is None:
            print('No vector db provided')
            return
        if self.reader is None:
            print('No reader provided')
            return
        if recreate:
            print('Dropping collection')
            self.vector_db.drop()
        print('Creating collection')
        self.vector_db.create()
        print('Loading knowledge base')
        num_documents = 0
        urls_to_read = self.urls.copy()
        if not recreate:
            for url in urls_to_read:
                print(f'Checking if {url} exists in the vector db')
                if self.vector_db.name_exists(name=url):
                    print(f'Skipping {url} as it exists in the vector db')
                    urls_to_read.remove(url)
        for url in urls_to_read:
            document_list = self.reader.read(url=url)
            if not recreate:
                document_list = [document for document in document_list if not self.vector_db.doc_exists(document)]
            if upsert and self.vector_db.upsert_available():
                self.vector_db.upsert(documents=document_list, filters=filters)
            else:
                self.vector_db.insert(documents=document_list, filters=filters)
            num_documents += len(document_list)
            print(f'Loaded {num_documents} documents to knowledge base')
        if self.optimize_on is not None and num_documents > self.optimize_on:
            print('Optimizing Vector DB')
            self.vector_db.optimize()

    async def async_load(self, recreate: bool = False, upsert: bool = True, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None) -> None:
        if self.vector_db is None:
            print('No vector db provided')
            return
        if self.reader is None:
            print('No reader provided')
            return
        vector_db = self.vector_db
        reader = self.reader
        if recreate:
            print('Dropping collection asynchronously')
            await vector_db.async_drop()
        print('Creating collection asynchronously')
        await vector_db.async_create()
        print('Loading knowledge base asynchronously')
        num_documents = 0
        urls_to_read = self.urls.copy()
        if not recreate:
            for url in urls_to_read[:]:
                print(f'Checking if {url} exists in the vector db')
                name_exists = vector_db.async_name_exists(name=url)
                if name_exists:
                    print(f'Skipping {url} as it exists in the vector db')
                    urls_to_read.remove(url)

        async def process_url(url: str) -> List[Document]:
            try:
                document_list = await reader.async_read(url=url)
                if not recreate:
                    filtered_documents = []
                    for document in document_list:
                        if not await vector_db.async_doc_exists(document):
                            filtered_documents.append(document)
                    document_list = filtered_documents
                return document_list
            except Exception as e:
                print(f'Error processing URL {url}: {e}')
                return []
        url_tasks = [process_url(url) for url in urls_to_read]
        all_document_lists = await asyncio.gather(*url_tasks)
        for document_list in all_document_lists:
            if document_list:
                if upsert and vector_db.upsert_available():
                    await vector_db.async_upsert(documents=document_list, filters=filters)
                else:
                    await vector_db.async_insert(documents=document_list, filters=filters)
                num_documents += len(document_list)
                print(f'Loaded {num_documents} documents to knowledge base asynchronously')
        if self.optimize_on is not None and num_documents > self.optimize_on:
            print('Optimizing Vector DB')
            vector_db.optimize()
