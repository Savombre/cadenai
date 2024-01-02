import pytest 
from cadenai.document.text_splitter import SizeSplitter, SeparatorSplitter, ChunkType, LLMSplitter 
from cadenai.schema import DocumentHandler, Loader

@pytest.fixture
def mock_llm(mocker):
    return mocker.Mock()

@pytest.fixture
def splitter(mock_llm):
    return LLMSplitter(llm=mock_llm)


def test_sizesplitter_default_attributes():
    splitter = SizeSplitter()
    
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 100
    assert splitter.chunk_type == ChunkType.CHARACTER

def test_separatorsplitter_default_attributes():
    splitter = SeparatorSplitter()
    
    assert splitter.separator == "\n"
    assert splitter.is_separator_regex == False

def test_sizesplitter_split_text_by_characters():
    splitter = SizeSplitter(chunk_size=5, chunk_overlap=2,chunk_type="characters")
    input_text = "1234567890"
    result = splitter.split_text(input_text)
    assert len(result) == 3
    assert result[0].page_content == "12345"
    assert result[1].page_content == "45678"
    assert result[2].page_content == "7890"

def test_sizesplitter_split_text_by_characters_with_document_handler():
    splitter = SizeSplitter(chunk_size=5, chunk_overlap=2,chunk_type="characters")
    input_doc = DocumentHandler(page_content="1234567890",metadata={"nature": "just numbers"})
    result = splitter.split_text(input_doc)
    assert len(result) == 3
    assert result[0].page_content == "12345"
    assert result[0].metadata == {"nature": "just numbers"}
    assert result[1].page_content == "45678"
    assert result[1].metadata == {"nature": "just numbers"}
    assert result[2].page_content == "7890"
    assert result[2].metadata == {"nature": "just numbers"}

def test_sizesplitter_split_text_by_words():
    splitter = SizeSplitter(chunk_size=3, chunk_overlap=1, chunk_type="words")
    input_text = "one two three four five six"
    result = splitter.split_text(input_text)
    assert len(result) == 3
    assert result[0].page_content == "one two three"
    assert result[1].page_content == "three four five"
    assert result[2].page_content == "five six"

def test_sizesplitter_split_text_by_words_with_document_handler():
    splitter = SizeSplitter(chunk_size=3, chunk_overlap=1, chunk_type="words")
    input_doc = DocumentHandler(page_content="one two three four five six",metadata={"nature": "just words"})
    result = splitter.split_text(input_doc)
    assert len(result) == 3
    assert result[0].page_content == "one two three"
    assert result[0].metadata == {"nature": "just words"}
    assert result[1].page_content == "three four five"
    assert result[1].metadata == {"nature": "just words"}
    assert result[2].page_content == "five six"
    assert result[2].metadata == {"nature": "just words"}

def test_sizesplitter_split_text_by_tokens(mocker):

    mocked_encoding = mocker.Mock()
    mocked_encoding.decode.return_value = "mocked_decoded_text"
    mocked_encoding.encode.return_value = "mocked_encoded_text"
    mocker.patch('cadenai.document.text_splitter.tiktoken.get_encoding', return_value=mocked_encoding)

    splitter = SizeSplitter(chunk_size=5, chunk_overlap=2, chunk_type="tokens")

    text = "Moi, j'aime les moches car j'ai vu ta mère sur chatroulette"
    result = [doc.page_content for doc in splitter.split_text(text)]
    #Avec ce texte, cela produit normalement 6 chunks, mais on a mocké l'encoding, donc on a 6 fois "mocked_encoded_text"
    expected = ["mocked_decoded_text", "mocked_decoded_text", "mocked_decoded_text", "mocked_decoded_text","mocked_decoded_text", "mocked_decoded_text"]

    assert result == expected

def test_sizesplitter_split_text_by_tokens_with_document_handler(mocker):

    mocked_encoding = mocker.Mock()
    mocked_encoding.decode.return_value = "mocked_decoded_text"
    mocked_encoding.encode.return_value = "mocked_encoded_text"
    mocker.patch('cadenai.document.text_splitter.tiktoken.get_encoding', return_value=mocked_encoding)

    splitter = SizeSplitter(chunk_size=5, chunk_overlap=2, chunk_type="tokens")

    text = "Moi, j'aime les moches car j'ai vu ta mère sur chatroulette"
    metadata = {"author": "Max Boublil"}
    input_doc = DocumentHandler(page_content=text, metadata=metadata)
    result_text = [doc.page_content for doc in splitter.split_text(input_doc)]
    result_metadata = [doc.metadata for doc in splitter.split_text(input_doc)]
    #Avec ce texte, cela produit normalement 6 chunks, mais on a mocké l'encoding, donc on a 6 fois "mocked_encoded_text"
    expected_text = ["mocked_decoded_text", "mocked_decoded_text", "mocked_decoded_text", "mocked_decoded_text","mocked_decoded_text", "mocked_decoded_text"]
    expected_metadata = [{"author": "Max Boublil"}, {"author": "Max Boublil"}, {"author": "Max Boublil"}, {"author": "Max Boublil"}, {"author": "Max Boublil"}, {"author": "Max Boublil"}]

    assert result_text == expected_text
    assert result_metadata == expected_metadata

@pytest.mark.parametrize(
    "separator, is_separator_regex, input_text, input_metadata, expected_text_output, expected_metadata_output",
    [
        (":", False, "one:two:three", {"c'est quoi ?" : "des chiffres en anglais"}, ["one", "two", "three"],[{"c'est quoi ?" : "des chiffres en anglais"} ,{"c'est quoi ?" : "des chiffres en anglais"} ,{"c'est quoi ?" : "des chiffres en anglais"}]),
        (r"\(.*?\)", True, "J'ai vu (ta mère) sur chatroulette", {"author": "Max Boublil"}, ["J'ai vu ", ' sur chatroulette'], [{"author": "Max Boublil"}, {"author": "Max Boublil"}]),
    ]
)
def test_separatorsplit(separator, is_separator_regex, input_text, input_metadata, expected_text_output, expected_metadata_output):

    # Without DocumentHandler
    splitter = SeparatorSplitter(separator=separator, is_separator_regex=is_separator_regex)
    result = splitter.split_text(input_text)
    assert [doc.page_content for doc in result] == expected_text_output

    # With DocumentHandler
    input_doc = DocumentHandler(page_content=input_text, metadata=input_metadata)
    result = splitter.split_text(input_doc)
    assert [doc.page_content for doc in result] == expected_text_output
    assert [doc.metadata for doc in result] == expected_metadata_output

def test_sizesplitter_word_list_to_sentence_list():
    splitter = SizeSplitter()
    word_list = [["Moi,", "j'aime", "les", "moches"],["J'ai", "vu", "ta", "mère", "sur", "chatroulette"],["Ce", "soir", "tu", "vas", "prendre"]]
    result = splitter._word_list_to_sentence_list(word_list)
    expected = ["Moi, j'aime les moches", "J'ai vu ta mère sur chatroulette", "Ce soir tu vas prendre"]

    assert result == expected

class MockLoader(Loader):
        def lazy_load(self):
            yield DocumentHandler(page_content="test1 test2 test3", metadata={"number range": "1-3"})
            yield DocumentHandler(page_content="test4 test5 test6", metadata={"number range": "4-6"})

        def __len__(self):
            return 2

def test_split_text_with_loader(mocker):

    splitter = SizeSplitter(chunk_size=2, chunk_overlap=1,chunk_type="words")
    mocked_loader = MockLoader()
    result = splitter.split_text(mocked_loader)
    
    assert len(result) == 4
    assert result[0].page_content == "test1 test2"
    assert result[0].metadata == {"number range": "1-3"}
    assert result[1].page_content == "test2 test3"
    assert result[1].metadata == {"number range": "1-3"}
    assert result[2].page_content == "test4 test5"
    assert result[2].metadata == {"number range": "4-6"}
    assert result[3].page_content == "test5 test6"
    assert result[3].metadata == {"number range": "4-6"}

def test_split_text_invalid_type():
    splitter = SizeSplitter()
    with pytest.raises(TypeError):
        splitter.split_text(input_data=12345)  # Ici, 12345 est un int, ce qui est un type non valide.

def test_split_text_str(mocker, splitter):
    input_data = "Some input text"
    mock_response = """
    [
        {
            "page_content": "Some content",
            "metadata": {
                "page_number": 1,
                "chunk_id": 1
            }
        }
    ]
    """
    expected_output = [DocumentHandler(page_content="Some content", metadata={"page_number": 1, "chunk_id": 1})]

    mocker.patch.object(splitter, 'run', return_value=mock_response)

    result = splitter._split_text_str(input_data)
    assert result == expected_output