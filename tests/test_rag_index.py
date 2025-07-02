import sys
from pathlib import Path
import os
import pytest

# Добавляем корневую папку в путь, чтобы можно было импортировать src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import rag_system
from src.core.rag_system_instance import recreate_rag_system

TEST_PDF_PATH = r'C:\Users\ia_sv\Downloads\Federal_zakon.pdf'

@pytest.fixture(autouse=True)
def reset_rag_system():
    # Перед каждым тестом пересоздаём rag_system (чистое состояние)
    recreate_rag_system()
    yield
    recreate_rag_system()


def test_ingest_pdf():
    assert os.path.exists(TEST_PDF_PATH), f"Тестовый PDF не найден: {TEST_PDF_PATH}"
    result = rag_system.ingest_pdf(TEST_PDF_PATH)
    assert result, "Загрузка PDF не удалась"
    info = rag_system.get_collection_info()
    assert info.get('total_documents', 0) > 0, f"В коллекции нет документов: {info}"


def test_query_after_ingest():
    rag_system.ingest_pdf(TEST_PDF_PATH)
    question = "О чем этот документ?"
    answer = rag_system.query(question)
    assert isinstance(answer, str), "Ответ не строка"
    assert answer.strip(), "Ответ пустой"
    assert not answer.lower().startswith("ошибка"), f"Ответ содержит ошибку: {answer}"


def test_clear_collection():
    rag_system.ingest_pdf(TEST_PDF_PATH)
    info_before = rag_system.get_collection_info()
    assert info_before.get('total_documents', 0) > 0, "Документы не были загружены"
    cleared = rag_system.clear_collection()
    assert cleared, "Очистка коллекции не удалась"
    info_after = rag_system.get_collection_info()
    assert info_after.get('total_documents', 0) == 0, "Коллекция не пуста после очистки" 