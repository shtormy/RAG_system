# Модуль реализует работу с Google Drive: синхронизация, загрузка, скачивание, хранение индекса, интеграция с RAG.
import os
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO
from loguru import logger
import json

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from ..core.config import config
from ..core.rag_system import rag_system
from ..utils.pdf_utils import extract_text_from_pdf, get_pdf_metadata

logger = logger.bind(name=__name__)

INDEXED_FILES_PATH = os.path.join(config.chroma["persist_directory"], "indexed_files.json")


class GoogleDriveError(Exception):
    """Исключение для ошибок Google Drive"""
    pass


def ensure_gdrive_token(gauth, credentials_file: str = 'token.json'):
    """
    Проверяет наличие refresh_token, если его нет — инициирует OAuth flow.
    Для консольного режима использует LocalWebserverAuth.
    """
    # Пытаемся загрузить сохраненные учетные данные
    if os.path.exists(credentials_file):
        gauth.LoadCredentialsFile(credentials_file)
    # Проверяем наличие refresh_token
    refresh_token = None
    if gauth.credentials is not None:
        refresh_token = getattr(gauth.credentials, 'refresh_token', None)
    if not refresh_token:
        logger.warning("refresh_token отсутствует, требуется авторизация Google Drive!")
        gauth.settings['oauth_flow_params'] = {'access_type': 'offline', 'prompt': 'consent'}
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile(credentials_file)
        logger.info("refresh_token успешно получен и сохранён.")
    else:
        # logger.info("refresh_token найден, авторизация не требуется.")
        pass
    return gauth


class GoogleDriveClient:
    """Клиент для работы с Google Drive используя pydrive2"""
    
    def __init__(self):
        """Инициализирует клиент Google Drive"""
        self.folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not self.folder_id:
            raise GoogleDriveError("GOOGLE_DRIVE_FOLDER_ID не установлен")
        
        self.drive = self._setup_drive()
        logger.info("Google Drive клиент инициализирован", folder_id=self.folder_id)
    
    def _setup_drive(self) -> GoogleDrive:
        """Настраивает подключение к Google Drive"""
        try:
            gauth = GoogleAuth()
            google_drive_config = config.google_drive
            credentials_file = google_drive_config.get('save_credentials_file', 'token.json')
            # Используем универсальную функцию
            gauth = ensure_gdrive_token(gauth, credentials_file)
            if gauth.access_token_expired:
                gauth.Refresh()
            else:
                gauth.Authorize()
            gauth.SaveCredentialsFile(credentials_file)
            drive = GoogleDrive(gauth)
            
            logger.info("Google Drive сервис настроен")
            return drive
            
        except Exception as e:
            logger.error(
                "Ошибка при настройке Google Drive сервиса",
                error=str(e),
                error_type=str(type(e)),
                config_google_drive=config.google_drive,
            )
            raise GoogleDriveError(f"Не удалось настроить Google Drive сервис: {e}")
    
    def list_pdf_files(self) -> List[Dict[str, Any]]:
        """
        Получает список PDF файлов из указанной папки
        
        Returns:
            Список файлов с метаданными
        """
        try:
            # Запрос файлов в папке
            file_list = self.drive.ListFile({
                'q': f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            }).GetList()
            
            files = []
            for file in file_list:
                files.append({
                    'id': file['id'],
                    'name': file['title'],
                    'size': file.get('fileSize', 0),
                    'modified': file.get('modifiedDate', ''),
                    'parents': file.get('parents', [])
                })
            
            logger.info("Получен список файлов", count=len(files))
            return files
            
        except Exception as e:
            logger.error("Ошибка при получении списка файлов", error=str(e))
            raise GoogleDriveError(f"Не удалось получить список файлов: {e}")
    
    def download_file(self, file_id: str, file_name: str) -> str:
        """
        Скачивает файл из Google Drive
        
        Args:
            file_id: ID файла в Google Drive
            file_name: Имя файла для сохранения
            
        Returns:
            Путь к скачанному файлу
        """
        try:
            # Создаем временную директорию
            temp_dir = Path(tempfile.gettempdir()) / "rag_system"
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / file_name
            
            # Скачиваем файл
            file = self.drive.CreateFile({'id': file_id})
            file.GetContentFile(str(file_path))
            
            #logger.info("Файл скачан", file_name=file_name, path=str(file_path))
            return str(file_path)
            
        except Exception as e:
            logger.error("Ошибка при скачивании файла", file_name=file_name, error=str(e))
            raise GoogleDriveError(f"Не удалось скачать файл {file_name}: {e}")
    
    def sync_to_rag(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Синхронизирует PDF файлы из Google Drive в RAG систему
        
        Args:
            dry_run: Если True, только показывает что будет сделано
            
        Returns:
            Результат синхронизации
        """
        try:
            logger.info("Начинаю синхронизацию PDF из Google Drive", dry_run=dry_run)
            
            # Получаем список файлов
            files = self.list_pdf_files()
            
            # Загружаем уже проиндексированные file_id
            indexed_file_ids = load_indexed_file_ids()
            logger.info("Загружено file_id уже проиндексированных файлов", count=len(indexed_file_ids))
            
            # Фильтруем только новые файлы
            new_files = [f for f in files if f['id'] not in indexed_file_ids]
            skipped_files = [f for f in files if f['id'] in indexed_file_ids]
            
            if dry_run:
                return {
                    "status": "dry_run",
                    "files_found": len(new_files),
                    "files": new_files,
                    "skipped": len(skipped_files)
                }
            
            processed_files = []
            errors = []
            new_indexed_ids = set()
            
            for file_info in new_files:
                try:
                    file_name = file_info['name']
                    file_id = file_info['id']
                    
                    logger.info("Обрабатываю файл", file_name=file_name)
                    
                    # Скачиваем файл
                    file_path = self.download_file(file_id, file_name)
                    
                    # Метаданные для RAG системы
                    metadata = {
                        "source": "google_drive",
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_size": file_info.get('size', 0),
                        "modified_date": file_info.get('modified', ''),
                        "folder_id": self.folder_id
                    }
                    
                    # Загружаем в RAG систему
                    success = rag_system.ingest_pdf(file_path, metadata)
                    
                    if success:
                        processed_files.append({
                            "name": file_name,
                            "id": file_id,
                            "status": "success"
                        })
                        new_indexed_ids.add(file_id)
                        logger.info("Файл успешно обработан", file_name=file_name)
                    else:
                        errors.append({
                            "name": file_name,
                            "id": file_id,
                            "status": "failed",
                            "error": "Ошибка обработки в RAG системе"
                        })
                        logger.error("Ошибка обработки файла в RAG системе", file_name=file_name)
                    
                    # Удаляем временный файл
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning("Не удалось удалить временный файл", file_path=file_path, error=str(e))
                
                except Exception as e:
                    errors.append({
                        "name": file_info.get('name', 'unknown'),
                        "id": file_info.get('id', 'unknown'),
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error("Ошибка обработки файла", file_info=file_info, error=str(e))
            
            # Обновляем список проиндексированных файлов
            if new_indexed_ids:
                indexed_file_ids.update(new_indexed_ids)
                save_indexed_file_ids(indexed_file_ids)
                logger.info("Обновлен список проиндексированных file_id", added=len(new_indexed_ids))
            
            # --- Удаление файлов, которых больше нет в Google Drive ---
            current_file_ids = set(f['id'] for f in files)
            to_remove = indexed_file_ids - current_file_ids
            removed_total = 0
            for file_id in to_remove:
                removed = rag_system.remove_documents_by_file_id(file_id)
                removed_total += removed
                logger.info(f"Удалено {removed} документов для file_id={file_id}")
            if to_remove:
                indexed_file_ids = indexed_file_ids - to_remove
                save_indexed_file_ids(indexed_file_ids)
                logger.info(f"Обновлён список file_id после удаления: {len(indexed_file_ids)}")
            # --- Конец удаления ---

            result = {
                "status": "completed",
                "total_files": len(files),
                "processed_files": len(processed_files),
                "errors": len(errors),
                "processed": processed_files,
                "errors_list": errors,
                "skipped": len(skipped_files)
            }
            
            logger.info("Синхронизация завершена", total=len(files), processed=len(processed_files), errors=len(errors), skipped=len(skipped_files))

            # === AutoTune: автоматический подбор параметров после загрузки документов ===
            autotune_cfg = config.get("autotune", {})
            if autotune_cfg.get("enabled", False):
                try:
                    docs_texts = []
                    for file_info in files:
                        file_path = self.download_file(file_info['id'], file_info['name'])
                        docs_texts.append(extract_text_from_pdf(file_path))
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                    test_queries = [file_info['name'] for file_info in files]
                    rag_system.autotune(docs_texts, queries=test_queries)
                    logger.info("[AutoTune] Автоматический подбор параметров завершён успешно.")
                except Exception as e:
                    logger.error(f"[AutoTune] Ошибка при автотюнинге: {e}")
            # === End AutoTune ===

            return result
            
        except Exception as e:
            logger.error("Ошибка синхронизации", error=str(e))
            raise GoogleDriveError(f"Ошибка синхронизации: {e}")

    def upload_file(self, local_path: str, file_name: Optional[str] = None) -> str:
        """
        Загружает локальный файл в папку Google Drive
        Args:
            local_path: Путь к локальному файлу
            file_name: Имя файла в Google Drive (если не указано, берётся из local_path)
        Returns:
            ID загруженного файла
        """
        try:
            file_name = file_name or os.path.basename(local_path)
            file_metadata = {
                'title': file_name,
                'parents': [{'id': self.folder_id}]
            }
            file_drive = self.drive.CreateFile(file_metadata)
            file_drive.SetContentFile(local_path)
            file_drive.Upload()
            logger.info("Файл загружен в Google Drive", file_name=file_name, file_id=file_drive['id'])
            return file_drive['id']
        except Exception as e:
            logger.error("Ошибка при загрузке файла в Google Drive", local_path=local_path, error=str(e))
            raise GoogleDriveError(f"Не удалось загрузить файл {local_path}: {e}")


def sync_google_drive(dry_run: bool = False) -> Dict[str, Any]:
    """
    Функция для синхронизации Google Drive
    
    Args:
        dry_run: Если True, только показывает что будет сделано
        
    Returns:
        Результат синхронизации
    """
    try:
        client = GoogleDriveClient()
        return client.sync_to_rag(dry_run)
    except Exception as e:
        logger.error("Ошибка при синхронизации Google Drive", error=str(e))
        return {
            "status": "error",
            "error": str(e)
        }


def load_indexed_file_ids() -> set:
    """Загружает множество уже проиндексированных file_id из файла."""
    if not os.path.exists(INDEXED_FILES_PATH):
        return set()
    try:
        with open(INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data)
    except Exception as e:
        logger.warning("Не удалось загрузить indexed_files.json", error=str(e))
        return set()

def save_indexed_file_ids(file_ids: set):
    """Сохраняет множество проиндексированных file_id в файл."""
    try:
        with open(INDEXED_FILES_PATH, "w", encoding="utf-8") as f:
            json.dump(list(file_ids), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Не удалось сохранить indexed_files.json", error=str(e)) 