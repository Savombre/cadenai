from pypdf import PdfReader
from typing import List, Any, Iterator

from ..schema import DocumentHandler, Loader

class PDFHandler(Loader):
    
    file_path: str
    _reader: Any
    pages : List

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._reader = PdfReader(self.file_path)
        self.pages = self._reader.pages

    def load_a_page(self, page_number: int) -> str:
        page_loaded = DocumentHandler(page_content=self.pages[page_number].extract_text())
        return page_loaded
    
    def lazy_load(self) -> Iterator[DocumentHandler]:
        for page in self.pages : 
            yield DocumentHandler(page_content=page.extract_text())

    def __len__(self) -> int:
        return len(self.pages)