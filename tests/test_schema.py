import pytest
import json

from cadenai.schema import DocumentHandler

def test_document_handler_save(mocker):
    test_data = {
        "page_content": "Test content",
        "metadata": {"key": "value"}
    }
    document_handler = DocumentHandler(**test_data)

    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.dump")

    test_path = "test_path.json"

    # Appel de la méthode save
    document_handler.save(test_path)

    open.assert_called_once_with(test_path, 'w', encoding='utf-8')

    # Vérifie si json.dump a été appelé avec les bonnes données
    json.dump.assert_called_once_with(document_handler.model_dump(), mocker.ANY, ensure_ascii=False, indent=4)