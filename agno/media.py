import base64
import zlib
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, field_validator, model_validator


class Media(BaseModel):
    id: str
    original_prompt: Optional[str] = None
    revised_prompt: Optional[str] = None


class VideoArtifact(Media):
    url: str
    eta: Optional[str] = None
    length: Optional[str] = None


class ImageArtifact(Media):
    url: Optional[str] = None
    content: Optional[bytes] = None
    mime_type: Optional[str] = None
    alt_text: Optional[str] = None


class AudioArtifact(Media):
    url: Optional[str] = None
    base64_audio: Optional[str] = None
    length: Optional[str] = None
    mime_type: Optional[str] = None


class Video(BaseModel):
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    format: Optional[str] = 'mp4'

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8') if self.content else None, 'filepath': self.filepath, 'format': self.format}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: VideoArtifact) -> 'Video':
        return cls(url=artifact.url)


class Audio(BaseModel):
    content: Optional[Any] = None
    filepath: Optional[Union[Path, str]] = None
    url: Optional[str] = None
    format: Optional[str] = None

    @property
    def audio_url_content(self) -> Optional[bytes]:
        if self.url:
            return httpx.get(self.url).content
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8')
            if self.content
            else None, 'filepath': self.filepath, 'format': self.format}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: AudioArtifact) -> 'Audio':
        return cls(url=artifact.url, content=artifact.base64_audio, format=artifact.mime_type)


class AudioResponse(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    expires_at: Optional[int] = None
    transcript: Optional[str] = None
    mime_type: Optional[str] = None
    sample_rate: Optional[int] = 24000
    channels: Optional[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'id': self.id, 'content': base64.b64encode(self.content).decode('utf-8')
            if isinstance(self.content, bytes)
            else self.content, 'expires_at': self.expires_at, 'transcript': self.transcript, 'mime_type': self.mime_type, 'sample_rate': self.sample_rate, 'channels': self.channels}
        return {k: v for k, v in response_dict.items() if v is not None}


class Image(BaseModel):
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    format: Optional[str] = 'jpeg'
    detail: Optional[str] = None
    id: Optional[str] = None

    @property
    def image_url_content(self) -> Optional[bytes]:
        if self.url:
            return httpx.get(self.url).content
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        response_dict = {'content': base64.b64encode(zlib.compress(self.content) if isinstance(self.content, bytes) else self.content.encode('utf-8')).decode('utf-8')
            if self.content
            else None, 'filepath': self.filepath, 'url': self.url, 'detail': self.detail}
        return {k: v for k, v in response_dict.items() if v is not None}

    @classmethod
    def from_artifact(cls, artifact: ImageArtifact) -> 'Image':
        return cls(url=artifact.url)


class File(BaseModel):
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    mime_type: Optional[str] = None

    @classmethod
    def valid_mime_types(cls) -> List[str]:
        return ['application/pdf', 'application/x-javascript', 'text/javascript', 'application/x-python', 'text/x-python', 'text/plain', 'text/html', 'text/css', 'text/md', 'text/csv', 'text/xml', 'text/rtf']

    @property
    def file_url_content(self) -> Optional[Tuple[bytes, str]]:
        if self.url:
            response = httpx.get(self.url)
            content = response.content
            mime_type = response.headers.get('Content-Type', '').split(';')[0]
            return content, mime_type
        else:
            return None
