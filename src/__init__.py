"""
RAG System — основной пакет для работы с документами, содержит инициализацию логирования и метаданные пакета.
"""

from .utils.logger import setup_logger

# Инициализируем логирование при импорте модуля
setup_logger()

__version__ = "2.0.0"
__author__ = "RAG System Team" 