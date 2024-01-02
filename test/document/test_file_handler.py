import pytest
#from cadenai.document.file_handler import DocumentHandler, Loader, PDFHandler
from cadenai.document.file_handler import DocumentHandler, Loader, PDFHandler

def test_document_handler_initialization():
    page_content = "Test content"
    metadata = {"author": "John Doe"}

    doc = DocumentHandler(page_content=page_content, metadata=metadata)

    assert doc.page_content == page_content
    assert doc.metadata == metadata

def test_document_handler_default_metadata():

    page_content="Me gustas tu"

    doc = DocumentHandler(page_content=page_content)

    assert doc.page_content == page_content
    assert doc.metadata == {}

class MockLoader(Loader):
    def lazy_load(self):
        pass

    def __len__(self):
        pass

def test_loader_load_method(mocker):
    mock_loader = MockLoader()
    
    sample_content = "Je pourrais passer ma vie dans cette bulle musicale"

    mocked_lazy_load = mocker.patch.object(mock_loader, "lazy_load", return_value=(DocumentHandler(page_content=sample_content) for _ in range(3)))

    documents = mock_loader.load()
    
    # Vérification que lazy_load a été appelé
    mocked_lazy_load.assert_called_once()

    assert len(documents) == 3
    for doc in documents:
        assert isinstance(doc, DocumentHandler)
        assert doc.page_content == "Je pourrais passer ma vie dans cette bulle musicale"

@pytest.fixture
def mock_pdf_pages(mocker) :

    mock_page_1 = mocker.MagicMock()
    mock_page_1.extract_text.return_value = "Page 1 content"

    mock_page_2 = mocker.MagicMock()
    mock_page_2.extract_text.return_value = "Page 2 content"

    mock_page_3 = mocker.MagicMock()
    mock_page_3.extract_text.return_value = "Page 3 content"

    return [mock_page_1, mock_page_2, mock_page_3]

@pytest.fixture
def mock_pdf_reader(mocker,  mock_pdf_pages) :
    mock_reader = mocker.MagicMock()
    mock_reader.pages = mock_pdf_pages
    return mock_reader

def test_pdf_handler_initialization(mock_pdf_reader,mocker): 
    mocker.patch('cadenai.document.file_handler.PdfReader', return_value=mock_pdf_reader)

    pdf = PDFHandler('dummy_path.pdf')
    assert pdf.file_path == 'dummy_path.pdf'

def test_pdf_handler_load_a_page(mocker, mock_pdf_reader):
    mocker.patch('cadenai.document.file_handler.PdfReader', return_value=mock_pdf_reader)
    handler = PDFHandler("fake_path.pdf")
    doc = handler.load_a_page(1)
    assert isinstance(doc, DocumentHandler)
    assert doc.page_content == "Page 2 content"

def test_pdf_handler_lazy_load(mocker, mock_pdf_reader):
    mocker.patch('cadenai.document.file_handler.PdfReader', return_value=mock_pdf_reader)
    handler = PDFHandler("fake_path.pdf")
    pages_gen = handler.lazy_load()
        
    page_1 = next(pages_gen)
    assert isinstance(page_1, DocumentHandler)
    assert page_1.page_content == "Page 1 content"

    page_2 = next(pages_gen)
    assert page_2.page_content == "Page 2 content"

    page_3 = next(pages_gen)
    assert page_3.page_content == "Page 3 content"

    # S'assurer qu'il n'y a pas d'autres pages
    with pytest.raises(StopIteration):
        next(pages_gen)

def test_pdf_handler_len(mocker, mock_pdf_reader) : 

    mocker.patch('cadenai.document.file_handler.PdfReader', return_value=mock_pdf_reader)

    pdf_handler = PDFHandler('kamasutra.pdf')

    assert len(pdf_handler) == 3
