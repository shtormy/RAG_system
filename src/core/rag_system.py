# Модуль реализует основную логику RAG-системы: загрузка, хранение, поиск, генерация ответов, работа с векторной и sparse базой, интеграция с LLM.
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from .config import config
from ..utils.pdf_utils import extract_text_from_file, get_file_metadata, is_supported_file, PDFProcessingError, DocxProcessingError

logger = logger.bind(name=__name__)


class RAGSystem:
    """Основная система RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self):
        """Инициализирует RAG систему"""
        self.documents: List[Document] = []  # для sparse индекса
        self.sparse_retriever = None
        self.hybrid_retriever = None
        self._setup_components()
        logger.info("RAG система инициализирована")
    
    def _setup_components(self):
        """Настраивает компоненты системы"""
        # Настройка чанкинга
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking["chunk_size"],
            chunk_overlap=config.chunking["chunk_overlap"]
        )
        
        # Настройка эмбеддингов
        self.embedding_model = OpenAIEmbeddings(
            model=config.embedding["model"],
            openai_api_key=os.getenv(config.embedding["api_key_env"])
        )
        
        # Настройка векторной базы данных
        persist_dir = config.chroma["persist_directory"]
        collection_name = config.chroma["collection_name"]
        
        # Создаем директорию если не существует
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )
        
        # --- Загружаем все документы из базы в self.documents ---
        try:
            all_docs = self.vectordb.get()
            from langchain.schema import Document
            if isinstance(all_docs, dict) and 'documents' in all_docs and 'metadatas' in all_docs:
                self.documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
                ]
            elif all_docs and isinstance(all_docs[0], str):
                self.documents = [Document(page_content=doc, metadata={}) for doc in all_docs]
            elif all_docs and isinstance(all_docs[0], dict):
                self.documents = [
                    Document(page_content=doc.get('page_content', ''), metadata=doc.get('metadata', {}))
                    for doc in all_docs
                ]
            else:
                self.documents = []
            logger.info(f"[INIT] Загружено документов из базы: {len(self.documents)}")
        except Exception as e:
            logger.error(f"[INIT] Не удалось загрузить документы из базы: {e}")
            self.documents = []
        
        # Настройка LLM
        self.llm = ChatOpenAI(
            model_name=config.llm["model"],
            temperature=0,
            openai_api_key=os.getenv(config.llm["api_key_env"])
        )
        
        # Настройка ретривера
        top_k = config.retrieval.get("top_k", 5)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": top_k})
        
        # Настройка QA цепи
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff"
        )
        
        logger.info("Компоненты RAG системы настроены", 
                   chunk_size=config.chunking["chunk_size"],
                   chunk_overlap=config.chunking["chunk_overlap"],
                   embedding_model=config.embedding["model"],
                   llm_model=config.llm["model"],
                   top_k=top_k)
    
    def filter_none_metadata(self, metadata: dict) -> dict:
        """Удаляет ключи с None-значениями из метаданных"""
        return {k: v for k, v in metadata.items() if v is not None}
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Загружает файл (PDF или DOCX) в систему
        
        Args:
            file_path: Путь к файлу
            metadata: Дополнительные метаданные
            
        Returns:
            True если загрузка прошла успешно
        """
        try:
            logger.info("Начинаю загрузку файла", file=file_path)
            
            # Проверяем поддержку файла
            if not is_supported_file(file_path):
                logger.error("Неподдерживаемый тип файла", file=file_path)
                return False
            
            # Извлекаем текст
            text = extract_text_from_file(file_path)
            if not text.strip():
                logger.warning("Пустой текст, пропускаем файл", file=file_path)
                return False

            # 1. autotune для подбора chunk_size и chunk_overlap
            self.autotune([text])
            # Пересоздаём text_splitter с новыми параметрами
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            from src.utils.autotune import generate_queries_from_doc
            queries = generate_queries_from_doc(text)
            logger.info(f"[AutoTune] Установлены параметры чанкинга: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

            # Получаем метаданные файла
            file_metadata = get_file_metadata(file_path)

            # Объединяем метаданные
            final_metadata = {
                "source": "cli",
                "file_path": str(file_path),
                **file_metadata
            }
            if metadata:
                final_metadata.update(metadata)

            # Фильтруем None-значения
            final_metadata = self.filter_none_metadata(final_metadata)

            # 2. Разбиваем на чанки с уже подобранными параметрами
            chunks = self.text_splitter.split_text(text)
            logger.info("Текст разбит на чанки", file=file_path, chunks=len(chunks))

            # Создаем документы с метаданными
            documents = [
                Document(page_content=chunk, metadata=final_metadata)
                for chunk in chunks
            ]
            # Добавляем в общий список документов для sparse индекса
            self.documents.extend(documents)
            # Добавляем в векторную базу
            self.vectordb.add_documents(documents)

            # 3. autotune второй раз — с queries для подбора top_k и метрики
            self.autotune([doc.page_content for doc in self.documents], queries)
            # Явно логируем итоговый top_k и recall@k, если они есть
            if hasattr(self, 'top_k'):
                logger.info(f"[AutoTune] Итоговый top_k: {self.top_k}")
            if hasattr(self, 'last_recall'):
                logger.info(f"[AutoTune] Итоговый recall@k: {self.last_recall}")
            
            logger.info("Файл успешно загружен", 
                       file=file_path, 
                       chunks=len(chunks),
                       total_chars=len(text))
            
            return True
            
        except (PDFProcessingError, DocxProcessingError) as e:
            logger.error("Ошибка обработки файла", file=file_path, error=str(e))
            import traceback
            logger.error("Traceback:", file=file_path, traceback=traceback.format_exc())
            print("TRACEBACK:\n", traceback.format_exc())
            return False
        except Exception as e:
            import traceback
            logger.error("Неожиданная ошибка при загрузке файла", 
                        file=file_path, error=str(e))
            logger.error("Traceback:", file=file_path, traceback=traceback.format_exc())
            print("TRACEBACK:\n", traceback.format_exc())
            return False
    
    def ingest_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Загружает PDF файл в систему (для обратной совместимости)
        
        Args:
            file_path: Путь к PDF файлу
            metadata: Дополнительные метаданные
            
        Returns:
            True если загрузка прошла успешно
        """
        return self.ingest_file(file_path, metadata)
    
    def query(self, question: str, metadata_filter: Optional[Dict[str, Any]] = None) -> str:
        """
        Выполняет поиск и генерирует ответ
        
        Args:
            question: Вопрос пользователя
            metadata_filter: Фильтр по метаданным
            
        Returns:
            Ответ на вопрос
        """
        try:
            logger.info("Выполняю гибридный поиск", question=question[:100] + "..." if len(question) > 100 else question)
            # --- Если self.documents пуст, подгружаем из chroma_db ---
            if not self.documents:
                try:
                    all_docs = self.vectordb.get()
                    from langchain.schema import Document
                    if all_docs and isinstance(all_docs[0], str):
                        self.documents = [Document(page_content=doc, metadata={}) for doc in all_docs]
                    elif all_docs and isinstance(all_docs[0], dict):
                        self.documents = [
                            Document(page_content=doc.get('page_content', ''), metadata=doc.get('metadata', {}))
                            for doc in all_docs
                        ]
                    else:
                        self.documents = []
                    logger.info(f"[QUERY] Подгружено документов из базы: {len(self.documents)}")
                except Exception as e:
                    logger.error(f"[QUERY] Не удалось подгрузить документы из базы: {e}")
                    self.documents = []
            # --- Конец подгрузки ---
            # --- Попытка пересоздать retriever, если он не инициализирован, но документы есть ---
            if self.hybrid_retriever is None and self.documents:
                self.sparse_retriever = SparseBM25Retriever(self.documents)
                self.hybrid_retriever = HybridRetriever(self.retriever, self.sparse_retriever, alpha=0.5)
                logger.info("[AUTO-RECOVER] Hybrid retriever был не инициализирован, пересоздаю на лету", docs=len(self.documents))
            # --- Конец автоинициализации ---
            if not self.hybrid_retriever:
                logger.error("Hybrid retriever не инициализирован (нет документов для поиска)!")
                return "В базе нет ни одного документа для поиска. Пожалуйста, загрузите документы в Google Drive."
            top_k = getattr(self, 'top_k', 5)
            retrieved_docs = self.hybrid_retriever.retrieve(question, k=top_k)
            # Генерируем ответ через LLM на основе retrieved_docs
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            response = self.llm.invoke(f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:")
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            logger.info("Ответ сгенерирован (гибрид)", question_length=len(question), response_length=len(response_text))
            return response_text
        except Exception as e:
            logger.error("Ошибка при выполнении гибридного поиска", question=question, error=str(e))
            return f"Произошла ошибка при поиске: {str(e)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получает информацию о коллекции документов
        
        Returns:
            Словарь с информацией о коллекции
        """
        try:
            collection = self.vectordb._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": config.chroma["collection_name"],
                "persist_directory": config.chroma["persist_directory"]
            }
        except Exception as e:
            logger.error("Ошибка при получении информации о коллекции", error=str(e))
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Очищает коллекцию документов
        
        Returns:
            True если очистка прошла успешно
        """
        try:
            logger.info("Пробую удалить коллекцию через delete_collection", 
                        vectordb_type=type(self.vectordb), 
                        collection_type=type(getattr(self.vectordb, '_collection', None)),
                        collection_obj=repr(getattr(self.vectordb, '_collection', None)))
            client = self.vectordb._client
            collection_name = self.vectordb._collection.name
            client.delete_collection(name=collection_name)
            logger.info(f"Коллекция '{collection_name}' удалена. Пересоздаю компоненты...")
            self._setup_components_dynamic()
            logger.info(f"Компоненты пересозданы, коллекция '{collection_name}' пуста.")
            return True
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            vectordb_type = type(self.vectordb)
            collection = getattr(self.vectordb, '_collection', None)
            collection_type = type(collection)
            collection_obj = repr(collection)
            logger.error("Ошибка при очистке коллекции", error=str(e), traceback=tb,
                         vectordb_type=vectordb_type,
                         collection_type=collection_type,
                         collection_obj=collection_obj)
            print("[DEBUG] Ошибка при очистке коллекции:")
            print("error:", str(e))
            print("traceback:\n", tb)
            print("vectordb_type:", vectordb_type)
            print("collection_type:", collection_type)
            print("collection_obj:", collection_obj)
            return False

    def autotune(self, docs: List[str], queries: Optional[List[str]] = None):
        """
        Автоматически подбирает параметры chunk_size, overlap и top_k на основе анализа документов.
        docs: список текстов документов (или чанков) для анализа.
        queries: тестовые запросы для подбора top_k (опционально).
        После подбора пересоздаёт компоненты системы с новыми параметрами.
        """
        from src.utils import autotune
        # 1. Подбор chunk_size и overlap
        params = autotune.suggest_chunking_params(docs)
        self.chunk_size = params["chunk_size"]
        self.chunk_overlap = params["chunk_overlap"]
        logger.info(f"[AutoTune] Установлены параметры чанкинга: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        # 2. Подбор top_k (если есть тестовые запросы)
        if queries:
            def retriever_factory(top_k):
                # Временный retriever для оценки
                return self.vectordb.as_retriever(search_kwargs={"k": top_k})
            top_k = autotune.suggest_top_k(docs, queries, retriever_factory)
            self.top_k = top_k
            # Сохраняем recall@k для явного логирования
            recall = autotune.evaluate_recall(self.vectordb.as_retriever(search_kwargs={"k": top_k}), queries, docs, top_k)
            self.last_recall = recall
            logger.info(f"[AutoTune] Установлен параметр top_k={self.top_k}, recall@k={recall:.2f}")
        else:
            self.top_k = getattr(self, 'top_k', 5)
        # 3. Пересоздаём компоненты системы с новыми параметрами
        self._setup_components_dynamic()
        # 4. Пересоздаём sparse и hybrid retriever, только если есть документы
        if not self.documents:
            logger.warning("[AutoTune] Пропускаю создание SparseBM25Retriever: нет документов")
            return
        self.sparse_retriever = SparseBM25Retriever(self.documents)
        self.hybrid_retriever = HybridRetriever(self.retriever, self.sparse_retriever, alpha=0.5)
        logger.info(f"[AutoTune] Sparse и Hybrid retriever пересозданы.")
        logger.info(f"[AutoTune] Компоненты системы пересозданы с новыми параметрами.")

    def _setup_components_dynamic(self):
        """
        Пересоздаёт компоненты системы с текущими self.chunk_size, self.chunk_overlap, self.top_k.
        Используется после autotune.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embedding_model = OpenAIEmbeddings(
            model=config.embedding["model"],
            openai_api_key=os.getenv(config.embedding["api_key_env"])
        )
        persist_dir = config.chroma["persist_directory"]
        collection_name = config.chroma["collection_name"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )
        self.llm = ChatOpenAI(
            model_name=config.llm["model"],
            temperature=0,
            openai_api_key=os.getenv(config.llm["api_key_env"])
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.top_k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff"
        )
        logger.info("[AutoTune] Динамические компоненты системы настроены", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, top_k=self.top_k)

    def remove_documents_by_file_id(self, file_id: str) -> int:
        """
        Удаляет все документы с заданным file_id из базы и из self.documents.
        Возвращает количество удалённых документов.
        """
        # Удаляем из self.documents
        before = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.metadata.get('file_id') != file_id]
        removed_count = before - len(self.documents)
        # Удаляем из векторной базы (Chroma)
        try:
            self.vectordb._collection.delete(where={"file_id": file_id})
            logger.info(f"Документы с file_id={file_id} удалены из ChromaDB", file_id=file_id, count=removed_count)
        except Exception as e:
            logger.error(f"Ошибка при удалении документов из ChromaDB", file_id=file_id, error=str(e))
        # Пересоздаём sparse и hybrid retriever
        self.sparse_retriever = SparseBM25Retriever(self.documents) if self.documents else None
        self.hybrid_retriever = HybridRetriever(self.retriever, self.sparse_retriever, alpha=0.5) if self.sparse_retriever else None
        return removed_count


class SparseBM25Retriever:
    """Sparse-ретривер на основе BM25 для гибридного RAG."""
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.tokenized_texts = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)

    def _tokenize(self, text: str) -> List[str]:
        # Простейший токенизатор, можно заменить на более сложный при необходимости
        return text.lower().split()

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]


class HybridRetriever:
    """Гибридный ретривер: объединяет dense и sparse retrieval."""
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        # Получаем dense результаты (документы и их score)
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        # Получаем sparse результаты (документы)
        sparse_docs = self.sparse_retriever.retrieve(query, k * 2)  # берем больше, чтобы не потерять релевантные
        # Собираем все уникальные документы
        doc_id_map = {}
        all_docs = []
        for doc in dense_docs + sparse_docs:
            doc_id = id(doc)
            if doc_id not in doc_id_map:
                doc_id_map[doc_id] = doc
                all_docs.append(doc)
        # Считаем dense_score (по позиции в dense_docs)
        dense_score_map = {id(doc): 1.0 - i / max(1, len(dense_docs)) for i, doc in enumerate(dense_docs)}
        # Считаем sparse_score (по позиции в sparse_docs)
        sparse_score_map = {id(doc): 1.0 - i / max(1, len(sparse_docs)) for i, doc in enumerate(sparse_docs)}
        # Комбинируем score
        scored = []
        for doc in all_docs:
            did = id(doc)
            dense_score = dense_score_map.get(did, 0.0)
            sparse_score = sparse_score_map.get(did, 0.0)
            score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            scored.append((score, doc))
        # Сортируем и возвращаем top-k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored[:k]]


# Глобальный экземпляр RAG системы
rag_system = RAGSystem()

# === ВРЕМЕННАЯ ДИАГНОСТИКА CHROMA DB ===
def test_chroma_db():
    try:
        from langchain.schema import Document
        print("[TEST] vectordb.get():", rag_system.vectordb.get())
        print("[TEST] vectordb.get(limit=1):", rag_system.vectordb.get(limit=1))
        print("[TEST] vectordb._collection.count():", rag_system.vectordb._collection.count())
    except Exception as e:
        print(f"[TEST] Ошибка при диагностике chroma_db: {e}")

if __name__ == '__main__':
    test_chroma_db() 