# Модуль отвечает за настройку и инициализацию логирования через loguru.
import os
import sys
from pathlib import Path
from loguru import logger

def setup_logger():
    """Настраивает логирование для всей системы через loguru: выводит логи в консоль и файл, форматирует и архивирует их."""
    
    # Удаляем стандартный обработчик
    logger.remove()
    
    # Получаем настройки из переменных окружения
    #log_level = os.getenv("LOG_LEVEL", "INFO")
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    log_file = os.getenv("LOG_FILE", "./logs/rag_system.log")
    
    # Создаем директорию для логов если её нет
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Формат логов
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Добавляем обработчик для консоли
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Добавляем обработчик для файла
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Логирование настроено", level=log_level, file=log_file)

# Инициализируем логгер при импорте модуля
setup_logger() 