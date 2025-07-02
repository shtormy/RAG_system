"""
Модуль autotune для автоматического подбора параметров чанкинга и поиска в RAG-системе.
Содержит функции для анализа документов, подбора chunk_size, overlap, top_k и расчёта метрик.
Используется для оптимизации параметров разбиения и поиска.
"""
from typing import List, Dict
import numpy as np
from loguru import logger

# Значения по умолчанию для подбора
DEFAULT_CHUNK_SIZES = [200, 300, 400, 500, 700]
DEFAULT_OVERLAPS = [50, 100, 120, 150]
DEFAULT_TOP_K = [3, 5, 8, 10, 15]


def suggest_chunking_params(docs: List[str]) -> Dict[str, int]:
    """
    Анализирует документы и предлагает оптимальные параметры chunk_size и overlap.
    Эвристика: средняя длина абзаца, медиана длины документа, частота коротких абзацев.
    """
    # Собираем статистику по длинам абзацев и документов
    para_lengths = []
    doc_lengths = []
    for doc in docs:
        paragraphs = [p for p in doc.split('\n') if p.strip()]
        para_lengths.extend([len(p) for p in paragraphs])
        doc_lengths.append(len(doc))
    avg_para = int(np.mean(para_lengths)) if para_lengths else 300
    med_doc = int(np.median(doc_lengths)) if doc_lengths else 1000
    # Эвристика: chunk_size чуть больше средней длины абзаца, overlap — 1/3 chunk_size
    chunk_size = min(DEFAULT_CHUNK_SIZES, key=lambda x: abs(x - avg_para*1.2))
    overlap = min(DEFAULT_OVERLAPS, key=lambda x: abs(x - chunk_size//3))
    logger.info(f"[AutoTune] Средняя длина абзаца: {avg_para}, медиана документа: {med_doc}, выбран chunk_size: {chunk_size}, overlap: {overlap}")
    return {"chunk_size": chunk_size, "chunk_overlap": overlap}


def suggest_top_k(docs: List[str], queries: List[str], retriever_factory, k_values=None) -> int:
    """
    Подбирает оптимальный top_k для поиска по тестовым запросам.
    retriever_factory(chunk_size, overlap, top_k) -> retriever
    Возвращает top_k с максимальным recall@k.
    """
    if k_values is None:
        k_values = DEFAULT_TOP_K
    best_k = k_values[0]
    best_recall = 0
    for k in k_values:
        retriever = retriever_factory(top_k=k)
        recall = evaluate_recall(retriever, queries, docs, k)
        #logger.info(f"[AutoTune] Recall@{k}: {recall:.2f}")
        if recall > best_recall:
            best_recall = recall
            best_k = k
    return best_k


def evaluate_recall(retriever, queries: List[str], docs: List[str], k: int) -> float:
    """
    Оценивает recall@k: доля запросов, для которых релевантный чанк найден в топ-k.
    Для простоты: релевантен, если в ответе есть хотя бы одно ключевое слово из запроса.
    """
    found = 0
    for q in queries:
        results = retriever.get_relevant_documents(q)
        # Простейшая эвристика: ищем совпадение по ключевым словам
        result_text = ' '.join([doc.page_content for doc in results])
        if any(word in result_text.lower() for word in q.lower().split()):
            found += 1
    recall = found / len(queries) if queries else 0.0
    return recall


def generate_queries_from_doc(doc: str, n: int = 8) -> List[str]:
    """
    Генерирует список queries на основе текста документа.
    Берёт первые n предложений и превращает их в вопросы (эвристика).
    """
    import re
    # Разбиваем на предложения
    sentences = re.split(r'[.!?]\s+', doc)
    queries = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 15:
            continue
        # Эвристика: превращаем утверждение в вопрос
        if sent.lower().startswith(('что такое', 'кто такой', 'как', 'где', 'когда', 'почему', 'зачем')):
            queries.append(sent + '?')
        else:
            queries.append(f'Что такое {sent.split(",")[0]}?')
        if len(queries) >= n:
            break
    return queries 