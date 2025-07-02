import sys
from pathlib import Path
import traceback

# Добавляем корневую папку в путь, чтобы можно было импортировать src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sources.google_drive import GoogleDriveClient, GoogleDriveError

if __name__ == "__main__":
    try:
        print("Пробую подключиться к Google Drive...")
        client = GoogleDriveClient()  # Авторизация происходит здесь
        files = client.list_pdf_files()  # Получаем только PDF-файлы
        print(f"Успешно подключено к Google Drive. Количество PDF-файлов: {len(files)}")
        for f in files:
            print(f"- {f.get('name', 'Без имени')} (id: {f.get('id', '-')})")
    except GoogleDriveError as e:
        print(f"Ошибка Google Drive: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        traceback.print_exc() 