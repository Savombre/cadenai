import pytest
from cadenai.encoder import OpenAIEncoder

# Création d'une fixture pour simuler l'encoder
@pytest.fixture
def mock_encoder(mocker):
    mock = mocker.Mock()
    mock.encode.return_value = [1, 2, 3]
    mock.decode.return_value = "decoded text"
    return mock

def test_openai_encoder_initialization(mocker, mock_encoder):
    # Mock tiktoken.get_encoding pour retourner notre simulateur d'encoder
    mocker.patch("cadenai.encoder.tiktoken.get_encoding", return_value=mock_encoder)

    encoder = OpenAIEncoder()
    
    assert encoder.model_name == "cl100k_base"
    assert encoder.encoder == mock_encoder

def test_openai_encoder_encode_a_string(mocker, mock_encoder):
    # Mock tiktoken.get_encoding pour retourner notre simulateur d'encoder
    mocker.patch("cadenai.encoder.tiktoken.get_encoding", return_value=mock_encoder)

    encoder = OpenAIEncoder()
    result = encoder.encode_a_string("some text")
    
    assert result == [1, 2, 3]
    mock_encoder.encode.assert_called_once_with("some text")

def test_openai_encoder_decode_a_string(mocker, mock_encoder):
    # Mock tiktoken.get_encoding pour retourner notre simulateur d'encoder
    mocker.patch("cadenai.encoder.tiktoken.get_encoding", return_value=mock_encoder)

    encoder = OpenAIEncoder()
    result = encoder.decode_a_string([1, 2, 3])
    
    assert result == "decoded text"
    mock_encoder.decode.assert_called_once_with([1, 2, 3])
