from typing import Iterator, List

from agno.document import Document
from agno.document.reader.url_reader import URLReader
from agno.knowledge.agent import AgentKnowledge



class UrlKnowledge(AgentKnowledge):
    urls: List[str] = []
    reader: URLReader = URLReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        """Iterate over URLs and yield lists of documents.
        Each object yielded by the iterator is a list of documents.

        Returns:
            Iterator[List[Document]]: Iterator yielding list of documents
        """

        for url in self.urls:
            try:
                yield self.reader.read(url=url)
            except Exception as e:
                print(f"Error reading URL {url}: {str(e)}")
