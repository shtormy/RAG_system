"""
Модуль rag_system_instance предназначен для управления единственным (синглтон) экземпляром класса RAGSystem во всём приложении.

- get_rag_system() возвращает текущий глобальный экземпляр RAGSystem.
- recreate_rag_system() пересоздаёт глобальный экземпляр RAGSystem (например, если требуется сбросить состояние).

Этот подход позволяет централизованно управлять состоянием RAGSystem и использовать его во всех частях приложения без необходимости явно передавать экземпляр.
"""
from src.core.rag_system import RAGSystem

_rag_system = RAGSystem()

def get_rag_system():
    """Возвращает глобальный экземпляр RAGSystem для использования во всём приложении."""
    return _rag_system

def recreate_rag_system():
    """Пересоздаёт глобальный экземпляр RAGSystem (например, для сброса состояния)."""
    global _rag_system
    _rag_system = RAGSystem()
    return _rag_system 