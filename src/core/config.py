# Модуль отвечает за загрузку, хранение и валидацию конфигурации всей системы RAG.
import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class ConfigError(Exception):
    """Исключение для ошибок конфигурации"""
    pass


class Config:
    """Класс для управления конфигурацией системы"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Файл конфигурации {self.config_path} не найден")
        except yaml.YAMLError as e:
            raise ConfigError(f"Ошибка парсинга YAML: {e}")
    
    def _validate_config(self):
        """Валидирует конфигурацию"""
        required_sections = [
            "chunking", "embedding", "llm", "chroma", "retrieval", "sources", "logging", "scheduling", "google_drive"
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigError(f"Отсутствует секция {section} в конфигурации")
        
        # Проверяем обязательные переменные окружения
        required_env_vars = ["OPENAI_API_KEY"]
        if self._config["sources"].get("telegram_enabled", False):
            required_env_vars.append("TELEGRAM_BOT_TOKEN")
        if self._config["sources"].get("google_drive_enabled", False):
            required_env_vars.append("GOOGLE_DRIVE_FOLDER_ID")
        
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                raise ConfigError(f"Отсутствует переменная окружения {env_var}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из конфигурации по ключу"""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def chunking(self) -> Dict[str, Any]:
        """Параметры чанкинга"""
        return self._config["chunking"]
    
    @property
    def embedding(self) -> Dict[str, Any]:
        """Параметры эмбеддингов"""
        return self._config["embedding"]
    
    @property
    def llm(self) -> Dict[str, Any]:
        """Параметры языковой модели"""
        return self._config["llm"]
    
    @property
    def chroma(self) -> Dict[str, Any]:
        """Параметры ChromaDB"""
        return self._config["chroma"]
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        """Параметры поиска"""
        return self._config["retrieval"]
    
    @property
    def sources(self) -> Dict[str, Any]:
        """Параметры источников данных"""
        return self._config["sources"]
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Параметры логирования"""
        return self._config["logging"]
    
    @property
    def scheduling(self) -> Dict[str, Any]:
        """Параметры планировщика"""
        return self._config["scheduling"]
    
    @property
    def google_drive(self) -> Dict[str, Any]:
        """Параметры Google Drive"""
        return self._config["google_drive"]


# Глобальный экземпляр конфигурации
config = Config() 