from abc import ABC, abstractmethod
from agno.embedder import OllamaEmbedder
from chonkie import SemanticChunker
import warnings
from typing import List, Optional
from agno.reader import Document
from agno.models.base import Model
from agno.models.message import Message


class ChunkingStrategy(ABC):
    """Base class for chunking strategies"""

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        """Clean the text by replacing multiple newlines with a single newline"""
        import re

        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Replace multiple tabs with a single tab
        cleaned_text = re.sub(r"\t+", "\t", cleaned_text)
        # Replace multiple carriage returns with a single carriage return
        cleaned_text = re.sub(r"\r+", "\r", cleaned_text)
        # Replace multiple form feeds with a single form feed
        cleaned_text = re.sub(r"\f+", "\f", cleaned_text)
        # Replace multiple vertical tabs with a single vertical tab
        cleaned_text = re.sub(r"\v+", "\v", cleaned_text)

        return cleaned_text


class AgenticChunking(ChunkingStrategy):
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text"""

    def __init__(self, model: Optional[Model] = None, max_chunk_size: int = 5000):
        if model is None:
            try:
                from agno.models.ollama import Ollama
            except Exception:
                raise ValueError("`openai` isn't installed. Please install it with `pip install openai`")
            model = Ollama()
        self.max_chunk_size = max_chunk_size
        self.model = model

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LLM to determine natural breakpoints based on context"""
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_meta_data = document.meta_data
        chunk_number = 1

        while remaining_text:
            # Ask model to find a good breakpoint within max_chunk_size
            prompt = f"""Analyze this text and determine a natural breakpoint within the first {self.max_chunk_size} characters.
            Consider semantic completeness, paragraph boundaries, and topic transitions.
            Return only the character position number of where to break the text:

            {remaining_text[: self.max_chunk_size]}"""

            try:
                response = self.model.response([Message(role="user", content=prompt)])
                if response and response.content:
                    break_point = min(int(response.content.strip()), self.max_chunk_size)
                else:
                    break_point = self.max_chunk_size
            except Exception:
                # Fallback to max size if model fails
                break_point = self.max_chunk_size

            # Extract chunk and update remaining text
            chunk = remaining_text[:break_point].strip()
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            elif document.name:
                chunk_id = f"{document.name}_{chunk_number}"
            meta_data["chunk_size"] = len(chunk)
            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk,
                )
            )
            chunk_number += 1

            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks


class DocumentChunking(ChunkingStrategy):
    """A chunking strategy that splits text based on document structure like paragraphs and sections"""

    def __init__(self, chunk_size: int = 5000, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Document]:
        """Split document into chunks based on document structure"""
        if len(document.content) <= self.chunk_size:
            return [document]

        # Split on double newlines first (paragraphs)
        paragraphs = self.clean_text(document.content).split("\n\n")
        chunks: List[Document] = []
        current_chunk = []
        current_size = 0
        chunk_meta_data = document.meta_data
        chunk_number = 1

        for para in paragraphs:
            para = para.strip()
            para_size = len(para)

            if current_size + para_size <= self.chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                meta_data = chunk_meta_data.copy()
                meta_data["chunk"] = chunk_number
                chunk_id = None
                if document.id:
                    chunk_id = f"{document.id}_{chunk_number}"
                elif document.name:
                    chunk_id = f"{document.name}_{chunk_number}"
                meta_data["chunk_size"] = len("\n\n".join(current_chunk))
                if current_chunk:
                    chunks.append(
                        Document(
                            id=chunk_id, name=document.name, meta_data=meta_data, content="\n\n".join(current_chunk)
                        )
                    )
                current_chunk = [para]
                current_size = para_size

        if current_chunk:
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            elif document.name:
                chunk_id = f"{document.name}_{chunk_number}"
            meta_data["chunk_size"] = len("\n\n".join(current_chunk))
            chunks.append(
                Document(id=chunk_id, name=document.name, meta_data=meta_data, content="\n\n".join(current_chunk))
            )

        # Handle overlap if specified
        if self.overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    # Add overlap from previous chunk
                    prev_text = chunks[i - 1].content[-self.overlap :]
                    meta_data = chunk_meta_data.copy()
                    meta_data["chunk"] = chunk_number
                    chunk_id = None
                    if document.id:
                        chunk_id = f"{document.id}_{chunk_number}"
                    meta_data["chunk_size"] = len(prev_text + chunks[i].content)
                    if prev_text:
                        overlapped_chunks.append(
                            Document(
                                id=chunk_id,
                                name=document.name,
                                meta_data=meta_data,
                                content=prev_text + chunks[i].content,
                            )
                        )
                else:
                    overlapped_chunks.append(chunks[i])
            chunks = overlapped_chunks

        return chunks


class FixedSizeChunking(ChunkingStrategy):
    """Chunking strategy that splits text into fixed-size chunks with optional overlap"""

    def __init__(self, chunk_size: int = 5000, overlap: int = 0):
        # overlap must be less than chunk size
        if overlap >= chunk_size:
            raise ValueError(f"Invalid parameters: overlap ({overlap}) must be less than chunk size ({chunk_size}).")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Document]:
        """Split document into fixed-size chunks with optional overlap"""
        content = self.clean_text(document.content)
        content_length = len(content)
        chunked_documents: List[Document] = []
        chunk_number = 1
        chunk_meta_data = document.meta_data

        start = 0
        while start + self.overlap < content_length:
            end = min(start + self.chunk_size, content_length)

            # Ensure we're not splitting a word in half
            if end < content_length:
                while end > start and content[end] not in [" ", "\n", "\r", "\t"]:
                    end -= 1

            # If the entire chunk is a word, then just split it at chunk_size
            if end == start:
                end = start + self.chunk_size

            chunk = content[start:end]
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            elif document.name:
                chunk_id = f"{document.name}_{chunk_number}"
            meta_data["chunk_size"] = len(chunk)
            chunked_documents.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk,
                )
            )
            chunk_number += 1
            start = end - self.overlap

        return chunked_documents


class RecursiveChunking(ChunkingStrategy):
    """Chunking strategy that recursively splits text into chunks by finding natural break points"""

    def __init__(self, chunk_size: int = 5000, overlap: int = 0):
        # overlap must be less than chunk size
        if overlap >= chunk_size:
            raise ValueError(f"Invalid parameters: overlap ({overlap}) must be less than chunk size ({chunk_size}).")

        if overlap > chunk_size * 0.15:
            warnings.warn(
                f"High overlap: {overlap} > 15% of chunk size ({chunk_size}). May cause slow processing.",
                RuntimeWarning,
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Document]:
        """Recursively chunk text by finding natural break points"""
        if len(document.content) <= self.chunk_size:
            return [document]

        chunks: List[Document] = []
        start = 0
        chunk_meta_data = document.meta_data
        chunk_number = 1
        content = self.clean_text(document.content)

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            if end < len(content):
                for sep in ["\n", "."]:
                    last_sep = content[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + 1
                        break

            chunk = content[start:end]
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            chunk_number += 1
            meta_data["chunk_size"] = len(chunk)
            chunks.append(Document(id=chunk_id, name=document.name, meta_data=meta_data, content=chunk))

            new_start = end - self.overlap
            if new_start <= start:  # Prevent infinite loop
                new_start = min(
                    len(content), start + max(1, self.chunk_size // 10)
                )  # Move forward by at least 10% of chunk size
            start = new_start

        return chunks


class SemanticChunking(ChunkingStrategy):
    """Chunking strategy that splits text into semantic chunks using chonkie"""

    def __init__(
        self, embedder = None, chunk_size: int = 5000, similarity_threshold: Optional[float] = 0.5
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunker = SemanticChunker(
            embedding_model=self.embedder.id,  # type: ignore
            chunk_size=self.chunk_size,
            threshold=self.similarity_threshold,
        )

    def chunk(self, document: Document) -> List[Document]:
        """Split document into semantic chunks using chokie"""
        if not document.content:
            return [document]

        # Use chonkie to split into semantic chunks
        chunks = self.chunker.chunk(self.clean_text(document.content))

        # Convert chunks to Documents
        chunked_documents: List[Document] = []
        for i, chunk in enumerate(chunks, 1):
            meta_data = document.meta_data.copy()
            meta_data["chunk"] = i
            chunk_id = f"{document.id}_{i}" if document.id else None
            meta_data["chunk_size"] = len(chunk.text)

            chunked_documents.append(Document(id=chunk_id, name=document.name, meta_data=meta_data, content=chunk.text))

        return chunked_documents

