# Модуль реализует Telegram-бота для взаимодействия с системой через чат, загрузки файлов и обработки запросов пользователей.
import os
import sys
from pathlib import Path

# Добавляем корневую папку в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio
from typing import Optional
from loguru import logger
import re
from langchain_openai import ChatOpenAI
from pydrive2.auth import GoogleAuth
from src.sources.google_drive import ensure_gdrive_token

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from tempfile import NamedTemporaryFile

from src.core.rag_system_instance import get_rag_system, recreate_rag_system
from src.core.config import config
from src.sources.google_drive import GoogleDriveClient, GoogleDriveError
from src.utils.session_manager import SessionManager

# Настраиваем логирование
logger = logger.bind(name=__name__)

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not API_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
    raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в переменных окружения")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

GDRIVE_CREDENTIALS_FILE = 'token.json'
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']

gauth_flow_cache = {}

def is_question(query: str) -> bool:
    q = query.strip().lower()
    question_words = [
        "что", "кто", "где", "когда", "почему", "зачем", "как", "сколько", "какой", "какая", "какие", "каков", "о чём", "расскажи", "поясни", "объясни"
    ]
    return any(q.startswith(word) for word in question_words) or "?" in q

def needs_llm_rewrite(query: str) -> bool:
    q = query.strip()
    # Очень короткие или неинформативные запросы (1-2 слова или только числа/коды)
    return len(q.split()) <= 2 or q.isdigit()

def llm_rewrite_func(query: str) -> str:
    """Переформулирует запрос через OpenAI LLM для универсального поиска."""
    try:
        llm = ChatOpenAI(
            model_name=config.llm["model"],
            temperature=0,
            openai_api_key=os.getenv(config.llm["api_key_env"])
        )
        prompt = f"Переформулируй этот запрос так, чтобы он был понятен для поиска в базе знаний: '{query}'"
        logger.debug(f"[DEBUG] Отправляю запрос на LLM для переформулировки: '{prompt}'")
        result = llm.invoke(prompt)
        if hasattr(result, 'content'):
            return result.content.strip()
        return str(result).strip()
    except Exception as e:
        logger.error(f"Ошибка при обращении к LLM для переформулировки: {e}")
        return f"Что вы можете рассказать про {query.strip()}?"

def preprocess_query(query: str, llm_rewrite_func=None) -> str:
    if is_question(query):
        logger.debug(f"[DEBUG] Препроцессинг: запрос уже вопрос — '{query}'")
        return query
    if needs_llm_rewrite(query) and llm_rewrite_func is not None:
        logger.debug(f"[DEBUG] Препроцессинг: запрос короткий, отправляю на LLM — '{query}'")
        return llm_rewrite_func(query)
    logger.debug(f"[DEBUG] Препроцессинг: оборачиваю в универсальный вопрос — '{query}'")
    return f"Что вы можете рассказать про {query.strip()}?"

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь PDF или задай вопрос по ранее загруженным документам.")
    logger.info("Пользователь начал работу", user_id=message.from_user.id)

@dp.message_handler(content_types=['document'])
async def handle_pdf(message: types.Message):
    logger.info(f"GOOGLE_DRIVE_FOLDER_ID={os.getenv('GOOGLE_DRIVE_FOLDER_ID')}")
    document = message.document
    if not document.file_name.lower().endswith(".pdf"):
        await message.reply("Пожалуйста, отправьте PDF-файл.")
        logger.warning("Пользователь отправил не PDF", user_id=message.from_user.id, file_name=document.file_name)
        return

    file = await bot.download_file_by_id(document.file_id)
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    gdrive_client = None
    gdrive_file_id = None
    gdrive_uploaded_path = None
    try:
        gdrive_client = GoogleDriveClient()
        # Загружаем файл на Google Drive
        gdrive_file_id = gdrive_client.upload_file(tmp_path, document.file_name)
        logger.info("Файл загружен в Google Drive", user_id=message.from_user.id, file_name=document.file_name, file_id=gdrive_file_id)
        # Скачиваем обратно для индексации (гарантируем единый путь обработки)
        gdrive_uploaded_path = gdrive_client.download_file(gdrive_file_id, document.file_name)
    except GoogleDriveError as e:
        logger.error("Ошибка загрузки файла в Google Drive", user_id=message.from_user.id, file_name=document.file_name, error=str(e))
        await message.reply("Ошибка при загрузке файла в Google Drive: " + str(e))
        os.unlink(tmp_path)
        return
    except Exception as e:
        logger.error("Неожиданная ошибка Google Drive", user_id=message.from_user.id, file_name=document.file_name, error=str(e))
        await message.reply("Неожиданная ошибка при работе с Google Drive: " + str(e))
        os.unlink(tmp_path)
        return

    metadata = {
        "source": f"telegram_{message.from_user.id}",
        "user_id": message.from_user.id,
        "file_name": document.file_name,
        "gdrive_file_id": gdrive_file_id
    }
    success = get_rag_system().ingest_pdf(gdrive_uploaded_path, metadata)

    # Добавляем file_id в индексированные (если требуется, например через save_indexed_file_ids)
    try:
        from src.sources.google_drive import load_indexed_file_ids, save_indexed_file_ids
        indexed = load_indexed_file_ids()
        indexed.add(gdrive_file_id)
        save_indexed_file_ids(indexed)
    except Exception as e:
        logger.warning("Не удалось обновить индексированные файлы Google Drive", file_id=gdrive_file_id, error=str(e))

    # Удаляем временные файлы
    os.unlink(tmp_path)
    if gdrive_uploaded_path and os.path.exists(gdrive_uploaded_path):
        try:
            os.unlink(gdrive_uploaded_path)
        except Exception as e:
            logger.warning("Не удалось удалить временный файл после индексации", file_path=gdrive_uploaded_path, error=str(e))

    if success:
        await message.reply("Файл загружен в Google Drive и проиндексирован системой.")
        logger.info("PDF успешно загружен и проиндексирован через Telegram", user_id=message.from_user.id, file_name=document.file_name, file_id=gdrive_file_id)
    else:
        await message.reply("Ошибка при обработке файла. Проверьте корректность PDF.")
        logger.error("Ошибка индексации PDF через Telegram", user_id=message.from_user.id, file_name=document.file_name, file_id=gdrive_file_id)

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_query(message: types.Message):
    query = message.text.strip()
    logger.info("Поступил вопрос от пользователя", user_id=message.from_user.id, query=query[:100])
    preprocessed_query = preprocess_query(query, llm_rewrite_func=llm_rewrite_func)
    logger.debug(f"[DEBUG] Препроцессированный запрос: '{preprocessed_query}'")
    response = get_rag_system().query(preprocessed_query)
    logger.debug(f"[DEBUG] Ответ от RAG: '{response}'")
    logger.debug(f"[DEBUG] Тип ответа от RAG: {type(response)}, содержимое: {repr(response)}")
    try:
        await message.reply(response)
    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения пользователю: {e}", response=repr(response))
    logger.info("Ответ отправлен пользователю", user_id=message.from_user.id)

@dp.message_handler(commands=['gdrive_auth'])
async def gdrive_auth(message: types.Message):
    gauth = GoogleAuth()
    gauth.DEFAULT_SETTINGS['client_config_file'] = 'client_secrets.json'
    gauth.DEFAULT_SETTINGS['save_credentials'] = True
    gauth.DEFAULT_SETTINGS['save_credentials_file'] = GDRIVE_CREDENTIALS_FILE
    gauth.DEFAULT_SETTINGS['get_refresh_token'] = True
    gauth.settings['oauth_scope'] = GDRIVE_SCOPES
    gauth.settings['oauth_flow_params'] = {'access_type': 'offline', 'prompt': 'consent'}
    # Получаем ссылку для авторизации
    auth_url = gauth.GetAuthUrl()
    gauth_flow_cache[message.from_user.id] = gauth
    await message.reply(f"Перейдите по ссылке для авторизации Google Drive и пришлите мне код командой /gdrive_code <код>\n{auth_url}")

@dp.message_handler(commands=['gdrive_code'])
async def gdrive_code(message: types.Message):
    code = message.get_args().strip()
    if not code:
        await message.reply("Пожалуйста, укажите код после команды. Пример: /gdrive_code <код>")
        return
    gauth = gauth_flow_cache.get(message.from_user.id)
    if not gauth:
        await message.reply("Сначала выполните /gdrive_auth для получения ссылки.")
        return
    try:
        gauth.Auth(code)
        gauth.SaveCredentialsFile(GDRIVE_CREDENTIALS_FILE)
        await message.reply("Авторизация Google Drive завершена, refresh_token сохранён.")
        logger.info("Google Drive OAuth завершён через Telegram", user_id=message.from_user.id)
    except Exception as e:
        await message.reply(f"Ошибка при авторизации: {e}")
        logger.error(f"Ошибка при авторизации Google Drive через Telegram: {e}", user_id=message.from_user.id)

if __name__ == '__main__':
    logger.info("Запуск Telegram-бота")
    try:
        from src.sources.google_drive import sync_google_drive
        logger.info("Выполняю синхронизацию с Google Drive при запуске...")
        sync_result = sync_google_drive()
        if sync_result.get("status") == "completed":
            logger.info(f"Синхронизация Google Drive завершена. Новых файлов: {sync_result.get('processed_files', 0)}, ошибок: {sync_result.get('errors', 0)}, пропущено: {sync_result.get('skipped', 0)}.")
        elif sync_result.get("status") == "error":
            logger.error(f"Ошибка при синхронизации Google Drive: {sync_result.get('error')}")
        info = get_rag_system().get_collection_info()
        from src.sources.google_drive import load_indexed_file_ids
        import os
        import glob
        chroma_dir = info.get('persist_directory', './chroma_db')
        indexed_files = load_indexed_file_ids()
        # Проверяем наличие других файлов кроме indexed_files.json
        files_in_chroma = [f for f in os.listdir(chroma_dir) if os.path.isfile(os.path.join(chroma_dir, f)) and f != 'indexed_files.json']
        if len(indexed_files) == 0 and len(files_in_chroma) == 0:
            logger.info("Обнаружено: indexed_files.json пустой и других индексов нет. Диагностика и полное удаление папки chroma_db.")
            import shutil
            def log_chroma_contents(stage):
                try:
                    if os.path.exists(chroma_dir):
                        files = []
                        for root, dirs, filez in os.walk(chroma_dir):
                            for name in filez:
                                files.append(os.path.relpath(os.path.join(root, name), chroma_dir))
                            for name in dirs:
                                files.append(os.path.relpath(os.path.join(root, name), chroma_dir) + '/')
                        logger.info(f"[{stage}] Содержимое chroma_db: {files}")
                    else:
                        logger.info(f"[{stage}] Папка chroma_db отсутствует")
                except Exception as e:
                    logger.warning(f"[{stage}] Не удалось получить содержимое chroma_db: {e}")
            log_chroma_contents('до удаления')
            # Удаляем всю папку chroma_db
            try:
                shutil.rmtree(chroma_dir)
            except Exception as e:
                logger.warning(f"Не удалось удалить папку {chroma_dir}: {e}")
            log_chroma_contents('после удаления')
            # Создаём папку заново
            try:
                os.makedirs(chroma_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Не удалось создать папку {chroma_dir}: {e}")
            log_chroma_contents('после создания')
            # Пересоздаём объект rag_system через модуль
            rag_system = recreate_rag_system()
            info = rag_system.get_collection_info()
            log_chroma_contents('после пересоздания rag_system')
            if info.get('total_documents', 0) > 0:
                logger.warning(f"[ДИАГНОСТИКА] После полной очистки база всё ещё содержит документы: {info.get('total_documents', 0)}. Содержимое chroma_db:")
                log_chroma_contents('после диагностики')
        logger.info(f"Всего документов в базе: {info.get('total_documents', 0)}; Коллекция: {info.get('collection_name', 'N/A')}; Директория: {info.get('persist_directory', 'N/A')}; Проиндексированных файлов: {len(indexed_files)}")
    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекции: {e}")
    executor.start_polling(dp, skip_updates=True) 