# Модуль реализует планировщик задач для периодической синхронизации данных с помощью cron.
import os
import time
import threading
from typing import Optional, Callable
from datetime import datetime
from loguru import logger
from croniter import croniter

logger = logger.bind(name=__name__)


class SchedulerError(Exception):
    """Исключение для ошибок планировщика"""
    pass


class Scheduler:
    """Планировщик задач с использованием cron"""
    
    def __init__(self):
        """Инициализирует планировщик"""
        self.enabled = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
        self.sync_interval = os.getenv("SYNC_INTERVAL", "0 */6 * * *")
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        
        if self.enabled:
            logger.info("Планировщик инициализирован", 
                       enabled=self.enabled, 
                       interval=self.sync_interval)
        else:
            logger.info("Планировщик отключен")
    
    def start(self, sync_function: Callable[[], None]) -> bool:
        """
        Запускает планировщик
        
        Args:
            sync_function: Функция для выполнения синхронизации
            
        Returns:
            True если планировщик запущен успешно
        """
        logger.info("Попытка запуска планировщика", enabled=self.enabled, running=self._running)
        
        if not self.enabled:
            logger.warning("Планировщик отключен, не запускаем")
            return False
        
        if self._running:
            logger.warning("Планировщик уже запущен")
            return False
        
        try:
            logger.info("Создаю cron объект", interval=self.sync_interval)
            # Создаем cron объект
            cron = croniter(self.sync_interval, datetime.now())
            
            logger.info("Создаю поток планировщика")
            # Запускаем в отдельном потоке
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_scheduler,
                args=(cron, sync_function),
                daemon=True
            )
            
            logger.info("Запускаю поток планировщика")
            self._thread.start()
            
            self._running = True
            logger.info("Планировщик запущен", interval=self.sync_interval)
            return True
            
        except Exception as e:
            logger.error("Ошибка запуска планировщика", error=str(e), error_type=type(e).__name__)
            return False
    
    def stop(self) -> bool:
        """
        Останавливает планировщик
        
        Returns:
            True если планировщик остановлен успешно
        """
        if not self._running:
            logger.warning("Планировщик не запущен")
            return False
        
        try:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=5)
            
            self._running = False
            logger.info("Планировщик остановлен")
            return True
            
        except Exception as e:
            logger.error("Ошибка остановки планировщика", error=str(e))
            return False
    
    def _run_scheduler(self, cron: croniter, sync_function: Callable[[], None]):
        """
        Основной цикл планировщика
        
        Args:
            cron: Объект cron с расписанием
            sync_function: Функция для выполнения
        """
        logger.info("Запущен цикл планировщика")
        
        while not self._stop_event.is_set():
            try:
                # Вычисляем время до следующего запуска
                next_run_time = cron.get_next(datetime)
                now = datetime.now()
                next_run_seconds = (next_run_time - now).total_seconds()
                
                logger.info("Следующий запуск синхронизации", 
                           next_run_seconds=next_run_seconds,
                           next_run_time=next_run_time)
                
                # Ждем до следующего запуска или до остановки
                if self._stop_event.wait(timeout=max(1, next_run_seconds)):
                    break
                
                # Выполняем синхронизацию
                logger.info("Запускаю запланированную синхронизацию")
                try:
                    sync_function()
                    logger.info("Запланированная синхронизация завершена")
                except Exception as e:
                    logger.error("Ошибка при выполнении запланированной синхронизации", error=str(e))
                
            except Exception as e:
                logger.error("Ошибка в цикле планировщика", error=str(e))
                # Ждем минуту перед повторной попыткой
                if self._stop_event.wait(timeout=60):
                    break
        
        logger.info("Цикл планировщика завершен")
    
    @property
    def is_running(self) -> bool:
        """Проверяет, запущен ли планировщик"""
        return self._running
    
    def get_next_run_time(self) -> Optional[datetime]:
        """
        Получает время следующего запуска
        
        Returns:
            Время следующего запуска или None если планировщик не запущен
        """
        if not self.enabled or not self._running:
            return None
        
        try:
            cron = croniter(self.sync_interval, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            logger.error("Ошибка получения времени следующего запуска", error=str(e))
            return None


# Глобальный экземпляр планировщика
scheduler = Scheduler()


def start_scheduler(sync_function: Callable[[], None]) -> bool:
    """
    Запускает глобальный планировщик
    
    Args:
        sync_function: Функция для выполнения синхронизации
        
    Returns:
        True если планировщик запущен успешно
    """
    return scheduler.start(sync_function)


def stop_scheduler() -> bool:
    """
    Останавливает глобальный планировщик
    
    Returns:
        True если планировщик остановлен успешно
    """
    return scheduler.stop()


def is_scheduler_running() -> bool:
    """
    Проверяет, запущен ли планировщик
    
    Returns:
        True если планировщик запущен
    """
    return scheduler.is_running


def get_next_sync_time() -> Optional[datetime]:
    """
    Получает время следующей синхронизации
    
    Returns:
        Время следующей синхронизации или None
    """
    return scheduler.get_next_run_time() 