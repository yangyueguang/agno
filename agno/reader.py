from typing import Any, Dict, List, Optional
from agno.embedder import Embedder
import random
import time
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag  # noqa: F401
from pypdf import PdfReader as DocumentReader  # noqa: F401
from pypdf.errors import PdfStreamError
from docx import Document as DocxDocument  # type: ignore
import csv
import io
import os
from pathlib import Path
from typing import IO, Any, List, Union
from urllib.parse import urlparse
import httpx
from time import sleep
import aiofiles
import asyncio
from dataclasses import dataclass, field
from typing import Any, List
from agno.chunking import FixedSizeChunking, ChunkingStrategy


@dataclass
class Document:
    """Dataclass for managing a document"""

    content: str
    id: Optional[str] = None
    name: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
    embedder: Optional[Embedder] = None
    embedding: Optional[List[float]] = None
    usage: Optional[Dict[str, Any]] = None
    reranking_score: Optional[float] = None

    def embed(self, embedder: Optional[Embedder] = None) -> None:
        """Embed the document using the provided embedder"""

        _embedder = embedder or self.embedder
        if _embedder is None:
            raise ValueError("No embedder provided")

        self.embedding, self.usage = _embedder.get_embedding_and_usage(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the document"""
        fields = {"name", "meta_data", "content"}
        return {
            field: getattr(self, field)
            for field in fields
            if getattr(self, field) is not None or field == "content"  # content is always included
        }

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "Document":
        """Returns a Document object from a dictionary representation"""
        return cls(**document)

    @classmethod
    def from_json(cls, document: str) -> "Document":
        """Returns a Document object from a json string representation"""
        import json

        return cls(**json.loads(document))


@dataclass
class Reader:
    """Base class for reading documents"""

    chunk: bool = True
    chunk_size: int = 3000
    separators: List[str] = field(default_factory=lambda: ["\n", "\n\n", "\r", "\r\n", "\n\r", "\t", " ", "  "])
    chunking_strategy: ChunkingStrategy = field(default_factory=FixedSizeChunking)

    def read(self, obj: Any) -> List[Document]:
        raise NotImplementedError

    async def async_read(self, obj: Any) -> List[Document]:
        raise NotImplementedError

    def chunk_document(self, document: Document) -> List[Document]:
        return self.chunking_strategy.chunk(document)

    async def chunk_documents_async(self, documents: List[Document]) -> List[Document]:
        """
        Asynchronously chunk a list of documents using the instance's chunk_document method.

        Args:
            documents: List of documents to be chunked.

        Returns:
            A flattened list of chunked documents.
        """

        async def _chunk_document_async(doc: Document) -> List[Document]:
            return await asyncio.to_thread(self.chunk_document, doc)

        # Process chunking in parallel for all documents
        chunked_lists = await asyncio.gather(*[_chunk_document_async(doc) for doc in documents])
        # Flatten the result
        return [chunk for sublist in chunked_lists for chunk in sublist]


def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    backoff_factor: int = 2,
):
    """Synchronous HTTP GET with retry logic."""

    for attempt in range(max_retries):
        try:
            response = httpx.get(url)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                raise
            wait_time = backoff_factor**attempt
            print(f"Request failed (attempt {attempt + 1}), retrying in {wait_time} seconds...")
            sleep(wait_time)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error for {url}: {e.response.status_code} - {e.response.text}")
            raise

    raise httpx.RequestError(f"Failed to fetch {url} after {max_retries} attempts")


class CSVReader(Reader):
    """Reader for CSV files"""

    def read(self, file: Union[Path, IO[Any]], delimiter: str = ",", quotechar: str = '"') -> List[Document]:
        try:
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                print(f"Reading: {file}")
                file_content = file.open(newline="", mode="r", encoding="utf-8")
            else:
                print(f"Reading uploaded file: {file.name}")
                file.seek(0)
                file_content = io.StringIO(file.read().decode("utf-8"))  # type: ignore

            csv_name = Path(file.name).stem if isinstance(file, Path) else file.name.split(".")[0]
            csv_content = ""
            with file_content as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
                for row in csv_reader:
                    csv_content += ", ".join(row) + "\n"

            documents = [
                Document(
                    name=csv_name,
                    id=csv_name,
                    content=csv_content,
                )
            ]
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents
            return documents
        except Exception as e:
            print(f"Error reading: {file.name if isinstance(file, IO) else file}: {e}")
            return []

    async def async_read(
        self, file: Union[Path, IO[Any]], delimiter: str = ",", quotechar: str = '"', page_size: int = 1000
    ) -> List[Document]:
        """
        Read a CSV file asynchronously, processing batches of rows concurrently.

        Args:
            file: Path or file-like object
            delimiter: CSV delimiter
            quotechar: CSV quote character
            page_size: Number of rows per page

        Returns:
            List of Document objects
        """
        try:
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                print(f"Reading async: {file}")
                async with aiofiles.open(file, mode="r", encoding="utf-8", newline="") as file_content:
                    content = await file_content.read()
                    file_content_io = io.StringIO(content)
            else:
                print(f"Reading uploaded file async: {file.name}")
                file.seek(0)
                file_content_io = io.StringIO(file.read().decode("utf-8"))  # type: ignore

            csv_name = Path(file.name).stem if isinstance(file, Path) else file.name.split(".")[0]

            file_content_io.seek(0)
            csv_reader = csv.reader(file_content_io, delimiter=delimiter, quotechar=quotechar)
            rows = list(csv_reader)
            total_rows = len(rows)

            if total_rows <= 10:
                csv_content = " ".join(", ".join(row) for row in rows)
                documents = [
                    Document(
                        name=csv_name,
                        id=csv_name,
                        content=csv_content,
                    )
                ]
            else:
                pages = []
                for i in range(0, total_rows, page_size):
                    pages.append(rows[i : i + page_size])

                async def _process_page(page_number: int, page_rows: List[List[str]]) -> Document:
                    """Process a page of rows into a document"""
                    start_row = (page_number - 1) * page_size + 1
                    page_content = " ".join(", ".join(row) for row in page_rows)

                    return Document(
                        name=csv_name,
                        id=f"{csv_name}_page{page_number}",
                        meta_data={"page": page_number, "start_row": start_row, "rows": len(page_rows)},
                        content=page_content,
                    )

                documents = await asyncio.gather(
                    *[_process_page(page_number, page) for page_number, page in enumerate(pages, start=1)]
                )

            if self.chunk:
                documents = await self.chunk_documents_async(documents)

            return documents
        except Exception as e:
            print(f"Error reading async: {file.name if isinstance(file, IO) else file}: {e}")
            return []


class CSVUrlReader(Reader):
    """Reader for CSV files"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No URL provided")

        print(f"Reading: {url}")
        # Retry the request up to 3 times with exponential backoff
        response = fetch_with_retry(url)

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "data.csv"

        file_obj = io.BytesIO(response.content)
        file_obj.name = filename

        documents = CSVReader().read(file=file_obj)

        file_obj.close()

        return documents


class DocxReader(Reader):
    """Reader for Doc/Docx files"""

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            if isinstance(file, Path):
                print(f"Reading: {file}")
                docx_document = DocxDocument(str(file))
                doc_name = file.stem
            else:  # Handle file-like object from upload
                print(f"Reading uploaded file: {file.name}")
                docx_document = DocxDocument(file)
                doc_name = file.name.split(".")[0]

            doc_content = "\n\n".join([para.text for para in docx_document.paragraphs])

            documents = [
                Document(
                    name=doc_name,
                    id=doc_name,
                    content=doc_content,
                )
            ]
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents
            return documents
        except Exception as e:
            print(f"Error reading file: {e}")
            return []


def process_image_page(doc_name: str, page_number: int, page: Any) -> Document:
    try:
        import rapidocr_onnxruntime as rapidocr
    except ImportError:
        raise ImportError(
            "`rapidocr_onnxruntime` not installed. Please install it via `pip install rapidocr_onnxruntime`."
        )
    ocr = rapidocr.RapidOCR()
    page_text = page.extract_text() or ""
    images_text_list = []

    # Extract and process images
    for image_object in page.images:
        image_data = image_object.data

        # Perform OCR on the image
        ocr_result, elapse = ocr(image_data)

        # Extract text from OCR result
        if ocr_result:
            images_text_list += [item[1] for item in ocr_result]

    images_text = "\n".join(images_text_list)
    content = page_text + "\n" + images_text

    # Append the document
    return Document(
        name=doc_name,
        id=f"{doc_name}_{page_number}",
        meta_data={"page": page_number},
        content=content,
    )


async def async_process_image_page(doc_name: str, page_number: int, page: Any) -> Document:
    try:
        import rapidocr_onnxruntime as rapidocr
    except ImportError:
        raise ImportError(
            "`rapidocr_onnxruntime` not installed. Please install it via `pip install rapidocr_onnxruntime`."
        )
    ocr = rapidocr.RapidOCR()

    page_text = page.extract_text() or ""
    images_text_list: List = []

    # Process images in parallel
    async def process_image(image_data: bytes) -> List[str]:
        ocr_result, _ = ocr(image_data)
        return [item[1] for item in ocr_result] if ocr_result else []

    image_tasks = [process_image(image.data) for image in page.images]
    images_results = await asyncio.gather(*image_tasks)

    for result in images_results:
        images_text_list.extend(result)

    images_text = "\n".join(images_text_list)
    content = page_text + "\n" + images_text

    return Document(
        name=doc_name,
        id=f"{doc_name}_{page_number}",
        meta_data={"page": page_number},
        content=content,
    )


class BasePDFReader(Reader):
    def _build_chunked_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents: List[Document] = []
        for document in documents:
            chunked_documents.extend(self.chunk_document(document))
        return chunked_documents


class PDFReader(BasePDFReader):
    """Reader for PDF files"""

    def read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        print(f"Reading: {doc_name}")

        try:
            doc_reader = DocumentReader(pdf)
        except PdfStreamError as e:
            print(f"Error reading PDF: {e}")
            return []

        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            documents.append(
                Document(
                    name=doc_name,
                    id=f"{doc_name}_{page_number}",
                    meta_data={"page": page_number},
                    content=page.extract_text(),
                )
            )
        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents

    async def async_read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        print(f"Reading: {doc_name}")

        try:
            doc_reader = DocumentReader(pdf)
        except PdfStreamError as e:
            print(f"Error reading PDF: {e}")
            return []

        async def _process_document(doc_name: str, page_number: int, page: Any) -> Document:
            return Document(
                name=doc_name,
                id=f"{doc_name}_{page_number}",
                meta_data={"page": page_number},
                content=page.extract_text(),
            )

        # Process pages in parallel using asyncio.gather
        documents = await asyncio.gather(
            *[
                _process_document(doc_name, page_number, page)
                for page_number, page in enumerate(doc_reader.pages, start=1)
            ]
        )

        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents


class PDFUrlReader(BasePDFReader):
    """Reader for PDF files from URL"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        import httpx

        print(f"Reading: {url}")
        # Retry the request up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                response = httpx.get(url)
                break
            except httpx.RequestError as e:
                if attempt == 2:  # Last attempt
                    print(f"Failed to fetch PDF after 3 attempts: {e}")
                    raise
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Request failed, retrying in {wait_time} seconds...")
                sleep(wait_time)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise

        doc_name = url.split("/")[-1].split(".")[0].replace("/", "_").replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            documents.append(
                Document(
                    name=doc_name,
                    id=f"{doc_name}_{page_number}",
                    meta_data={"page": page_number},
                    content=page.extract_text(),
                )
            )
        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents

    async def async_read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        try:
            import httpx
        except ImportError:
            raise ImportError("`httpx` not installed. Please install it via `pip install httpx`.")

        print(f"Reading: {url}")

        async with httpx.AsyncClient() as client:
            # Retry the request up to 3 times with exponential backoff
            for attempt in range(3):
                try:
                    response = await client.get(url)
                    break
                except httpx.RequestError as e:
                    if attempt == 2:  # Last attempt
                        print(f"Failed to fetch PDF after 3 attempts: {e}")
                        raise
                    wait_time = 2**attempt
                    print(f"Request failed, retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
                raise

        doc_name = url.split("/")[-1].split(".")[0].replace("/", "_").replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        async def _process_document(doc_name: str, page_number: int, page: Any) -> Document:
            return Document(
                name=doc_name,
                id=f"{doc_name}_{page_number}",
                meta_data={"page": page_number},
                content=page.extract_text(),
            )

        # Process pages in parallel using asyncio.gather
        documents = await asyncio.gather(
            *[
                _process_document(doc_name, page_number, page)
                for page_number, page in enumerate(doc_reader.pages, start=1)
            ]
        )

        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents


class PDFImageReader(BasePDFReader):
    """Reader for PDF files with text and images extraction"""

    def read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        if not pdf:
            raise ValueError("No pdf provided")

        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        print(f"Reading: {doc_name}")
        doc_reader = DocumentReader(pdf)

        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            documents.append(process_image_page(doc_name, page_number, page))

        if self.chunk:
            return self._build_chunked_documents(documents)

        return documents

    async def async_read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        if not pdf:
            raise ValueError("No pdf provided")

        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        print(f"Reading: {doc_name}")
        doc_reader = DocumentReader(pdf)

        documents = await asyncio.gather(
            *[
                async_process_image_page(doc_name, page_number, page)
                for page_number, page in enumerate(doc_reader.pages, start=1)
            ]
        )

        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents


class PDFUrlImageReader(BasePDFReader):
    """Reader for PDF files from URL with text and images extraction"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        import httpx

        # Read the PDF from the URL
        print(f"Reading: {url}")
        response = httpx.get(url)

        doc_name = url.split("/")[-1].split(".")[0].replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            documents.append(process_image_page(doc_name, page_number, page))

        # Optionally chunk documents
        if self.chunk:
            return self._build_chunked_documents(documents)

        return documents

    async def async_read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        import httpx

        print(f"Reading: {url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

        doc_name = url.split("/")[-1].split(".")[0].replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        documents = await asyncio.gather(
            *[
                async_process_image_page(doc_name, page_number, page)
                for page_number, page in enumerate(doc_reader.pages, start=1)
            ]
        )

        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents


class TextReader(Reader):
    """Reader for Text files"""

    def read(self, file: Union[Path, IO[Any]]) -> List[Document]:
        try:
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                print(f"Reading: {file}")
                file_name = file.stem
                file_contents = file.read_text("utf-8")
            else:
                print(f"Reading uploaded file: {file.name}")
                file_name = file.name.split(".")[0]
                file.seek(0)
                file_contents = file.read().decode("utf-8")

            documents = [
                Document(
                    name=file_name,
                    id=file_name,
                    content=file_contents,
                )
            ]
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents
            return documents
        except Exception as e:
            print(f"Error reading: {file}: {e}")
            return []

    async def async_read(self, file: Union[Path, IO[Any]]) -> List[Document]:
        try:
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")

                print(f"Reading asynchronously: {file}")
                file_name = file.stem

                try:
                    import aiofiles

                    async with aiofiles.open(file, "r", encoding="utf-8") as f:
                        file_contents = await f.read()
                except ImportError:
                    print("aiofiles not installed, using synchronous file I/O")
                    file_contents = file.read_text("utf-8")
            else:
                print(f"Reading uploaded file asynchronously: {file.name}")
                file_name = file.name.split(".")[0]
                file.seek(0)
                file_contents = file.read().decode("utf-8")

            document = Document(
                name=file_name,
                id=file_name,
                content=file_contents,
            )

            if self.chunk:
                return await self._async_chunk_document(document)
            return [document]
        except Exception as e:
            print(f"Error reading asynchronously: {file}: {e}")
            return []

    async def _async_chunk_document(self, document: Document) -> List[Document]:
        if not self.chunk or not document:
            return [document]

        async def process_chunk(chunk_doc: Document) -> Document:
            return chunk_doc

        chunked_documents = self.chunk_document(document)

        if not chunked_documents:
            return [document]

        tasks = [process_chunk(chunk_doc) for chunk_doc in chunked_documents]
        return await asyncio.gather(*tasks)


class URLReader(Reader):
    """Reader for general URL content"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        try:
            import httpx
        except ImportError:
            raise ImportError("`httpx` not installed. Please install it via `pip install httpx`.")

        print(f"Reading: {url}")
        # Retry the request up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                response = httpx.get(url)
                break
            except httpx.RequestError as e:
                if attempt == 2:  # Last attempt
                    print(f"Failed to fetch PDF after 3 attempts: {e}")
                    raise
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Request failed, retrying in {wait_time} seconds...")
                sleep(wait_time)

        try:
            print(f"Status: {response.status_code}")
            print(f"Content size: {len(response.content)} bytes")
        except Exception:
            pass

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise

        # Create a clean document name from the URL
        parsed_url = urlparse(url)
        doc_name = parsed_url.path.strip("/").replace("/", "_").replace(" ", "_")
        if not doc_name:
            doc_name = parsed_url.netloc

        # Create a single document with the URL content
        document = Document(
            name=doc_name,
            id=doc_name,
            meta_data={"url": url},
            content=response.text,
        )

        if self.chunk:
            return self.chunk_document(document)
        return [document]


@dataclass
class WebsiteReader(Reader):
    """Reader for Websites"""

    max_depth: int = 3
    max_links: int = 10

    _visited: Set[str] = field(default_factory=set)
    _urls_to_crawl: List[Tuple[str, int]] = field(default_factory=list)

    def delay(self, min_seconds=1, max_seconds=3):
        """
        Introduce a random delay.

        :param min_seconds: Minimum number of seconds to delay. Default is 1.
        :param max_seconds: Maximum number of seconds to delay. Default is 3.
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    async def async_delay(self, min_seconds=1, max_seconds=3):
        """
        Introduce a random delay asynchronously.

        :param min_seconds: Minimum number of seconds to delay. Default is 1.
        :param max_seconds: Maximum number of seconds to delay. Default is 3.
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(sleep_time)

    def _get_primary_domain(self, url: str) -> str:
        """
        Extract primary domain from the given URL.

        :param url: The URL to extract the primary domain from.
        :return: The primary domain.
        """
        domain_parts = urlparse(url).netloc.split(".")
        # Return primary domain (excluding subdomains)
        return ".".join(domain_parts[-2:])

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from a BeautifulSoup object.

        :param soup: The BeautifulSoup object to extract the main content from.
        :return: The main content.
        """
        # Try to find main content by specific tags or class names
        for tag in ["article", "main"]:
            element = soup.find(tag)
            if element:
                return element.get_text(strip=True, separator=" ")

        for class_name in ["content", "main-content", "post-content"]:
            element = soup.find(class_=class_name)
            if element:
                return element.get_text(strip=True, separator=" ")

        return ""

    def crawl(self, url: str, starting_depth: int = 1) -> Dict[str, str]:
        """
        Crawls a website and returns a dictionary of URLs and their corresponding content.

        Parameters:
        - url (str): The starting URL to begin the crawl.
        - starting_depth (int, optional): The starting depth level for the crawl. Defaults to 1.

        Returns:
        - Dict[str, str]: A dictionary where each key is a URL and the corresponding value is the main
                          content extracted from that URL.

        Note:
        The function focuses on extracting the main content by prioritizing content inside common HTML tags
        like `<article>`, `<main>`, and `<div>` with class names such as "content", "main-content", etc.
        The crawler will also respect the `max_depth` attribute of the WebCrawler class, ensuring it does not
        crawl deeper than the specified depth.
        """
        num_links = 0
        crawler_result: Dict[str, str] = {}
        primary_domain = self._get_primary_domain(url)
        # Add starting URL with its depth to the global list
        self._urls_to_crawl.append((url, starting_depth))
        while self._urls_to_crawl:
            # Unpack URL and depth from the global list
            current_url, current_depth = self._urls_to_crawl.pop(0)

            # Skip if
            # - URL is already visited
            # - does not end with the primary domain,
            # - exceeds max depth
            # - exceeds max links
            if (
                current_url in self._visited
                or not urlparse(current_url).netloc.endswith(primary_domain)
                or current_depth > self.max_depth
                or num_links >= self.max_links
            ):
                continue

            self._visited.add(current_url)
            self.delay()

            try:
                print(f"Crawling: {current_url}")
                response = httpx.get(current_url, timeout=10)

                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract main content
                main_content = self._extract_main_content(soup)
                if main_content:
                    crawler_result[current_url] = main_content
                    num_links += 1

                # Add found URLs to the global list, with incremented depth
                for link in soup.find_all("a", href=True):
                    if not isinstance(link, Tag):
                        continue

                    href_str = str(link["href"])
                    full_url = urljoin(current_url, href_str)

                    if not isinstance(full_url, str):
                        continue

                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc.endswith(primary_domain) and not any(
                        parsed_url.path.endswith(ext) for ext in [".pdf", ".jpg", ".png"]
                    ):
                        full_url_str = str(full_url)
                        if (
                            full_url_str not in self._visited
                            and (full_url_str, current_depth + 1) not in self._urls_to_crawl
                        ):
                            self._urls_to_crawl.append((full_url_str, current_depth + 1))

            except Exception as e:
                print(f"Failed to crawl: {current_url}: {e}")
                pass

        return crawler_result

    async def async_crawl(self, url: str, starting_depth: int = 1) -> Dict[str, str]:
        """
        Asynchronously crawls a website and returns a dictionary of URLs and their corresponding content.

        Parameters:
        - url (str): The starting URL to begin the crawl.
        - starting_depth (int, optional): The starting depth level for the crawl. Defaults to 1.

        Returns:
        - Dict[str, str]: A dictionary where each key is a URL and the corresponding value is the main
                        content extracted from that URL.
        """
        num_links = 0
        crawler_result: Dict[str, str] = {}
        primary_domain = self._get_primary_domain(url)

        # Clear previously visited URLs and URLs to crawl
        self._visited = set()
        self._urls_to_crawl = [(url, starting_depth)]

        async with httpx.AsyncClient() as client:
            while self._urls_to_crawl and num_links < self.max_links:
                current_url, current_depth = self._urls_to_crawl.pop(0)

                if (
                    current_url in self._visited
                    or not urlparse(current_url).netloc.endswith(primary_domain)
                    or current_depth > self.max_depth
                    or num_links >= self.max_links
                ):
                    continue

                self._visited.add(current_url)
                await self.async_delay()

                try:
                    print(f"Crawling asynchronously: {current_url}")
                    response = await client.get(current_url, timeout=10, follow_redirects=True)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, "html.parser")

                    # Extract main content
                    main_content = self._extract_main_content(soup)
                    if main_content:
                        crawler_result[current_url] = main_content
                        num_links += 1

                    # Add found URLs to the list, with incremented depth
                    for link in soup.find_all("a", href=True):
                        if not isinstance(link, Tag):
                            continue

                        href_str = str(link["href"])
                        full_url = urljoin(current_url, href_str)

                        if not isinstance(full_url, str):
                            continue

                        parsed_url = urlparse(full_url)
                        if parsed_url.netloc.endswith(primary_domain) and not any(
                            parsed_url.path.endswith(ext) for ext in [".pdf", ".jpg", ".png"]
                        ):
                            full_url_str = str(full_url)
                            if (
                                full_url_str not in self._visited
                                and (full_url_str, current_depth + 1) not in self._urls_to_crawl
                            ):
                                self._urls_to_crawl.append((full_url_str, current_depth + 1))

                except Exception as e:
                    print(f"Failed to crawl asynchronously: {current_url}: {e}")

        return crawler_result

    def read(self, url: str) -> List[Document]:
        """
        Reads a website and returns a list of documents.

        This function first converts the website into a dictionary of URLs and their corresponding content.
        Then iterates through the dictionary and returns chunks of content.

        :param url: The URL of the website to read.
        :return: A list of documents.
        """

        print(f"Reading: {url}")
        crawler_result = self.crawl(url)
        documents = []
        for crawled_url, crawled_content in crawler_result.items():
            if self.chunk:
                documents.extend(
                    self.chunk_document(
                        Document(
                            name=url, id=str(crawled_url), meta_data={"url": str(crawled_url)}, content=crawled_content
                        )
                    )
                )
            else:
                documents.append(
                    Document(
                        name=url,
                        id=str(crawled_url),
                        meta_data={"url": str(crawled_url)},
                        content=crawled_content,
                    )
                )
        return documents

    async def async_read(self, url: str) -> List[Document]:
        """
        Asynchronously reads a website and returns a list of documents.

        This function first converts the website into a dictionary of URLs and their corresponding content.
        Then iterates through the dictionary and returns chunks of content.

        :param url: The URL of the website to read.
        :return: A list of documents.
        """
        print(f"Reading asynchronously: {url}")
        crawler_result = await self.async_crawl(url)
        documents = []

        # Process documents in parallel
        async def process_document(crawled_url, crawled_content):
            if self.chunk:
                doc = Document(
                    name=url, id=str(crawled_url), meta_data={"url": str(crawled_url)}, content=crawled_content
                )
                return self.chunk_document(doc)
            else:
                return [
                    Document(
                        name=url,
                        id=str(crawled_url),
                        meta_data={"url": str(crawled_url)},
                        content=crawled_content,
                    )
                ]

        # Use asyncio.gather to process all documents in parallel
        tasks = [
            process_document(crawled_url, crawled_content) for crawled_url, crawled_content in crawler_result.items()
        ]
        results = await asyncio.gather(*tasks)

        # Flatten the results
        for doc_list in results:
            documents.extend(doc_list)

        return documents
