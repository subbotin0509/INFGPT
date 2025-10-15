# vectorstore.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_PATH = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def extract_keywords(question):
    """Извлечение ключевых слов из вопроса"""
    stop_words = {"что", "как", "почему", "зачем", "кто", "где", "когда", "это", "на", "в", "с", "по", "о", "у", "и", "а", "но", "или"}
    words = re.findall(r'\b\w+\b', question.lower())
    return [word for word in words if word not in stop_words and len(word) > 2]

def filter_by_relevance(docs, question, keywords):
    """Фильтрация документов по релевантности"""
    scored_docs = []
    question_lower = question.lower()
    
    for doc in docs:
        score = 0
        content = doc.page_content.lower()
        
        # Бонус за полное совпадение фраз
        question_words = set(question_lower.split())
        content_words = set(content.split())
        common_words = question_words.intersection(content_words)
        score += len(common_words) * 2
        
        # Бонус за ключевые слова
        keyword_matches = sum(1 for keyword in keywords if keyword in content)
        score += keyword_matches
        
        # Бонус за близость к началу документа
        if len(content) > 200:
            first_part = content[:200]
            early_matches = sum(1 for keyword in keywords if keyword in first_part)
            score += early_matches
            
        scored_docs.append((score, doc))
    
    # Сортировка по убыванию релевантности
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

def hybrid_search(question, vectorstore, top_k=5):
    """Гибридный поиск с улучшенной релевантностью"""
    # 1. Семантический поиск
    semantic_results = vectorstore.similarity_search(
        question, 
        k=top_k * 2  # Берем в 2 раза больше для фильтрации
    )
    
    # 2. Ключевые слова для улучшения поиска
    keywords = extract_keywords(question)
    
    # 3. Фильтрация по релевантности
    filtered_results = filter_by_relevance(semantic_results, question, keywords)
    
    return filtered_results[:top_k]

def smart_text_splitter(text, filename):
    """Умное разделение текста с учетом структуры"""
    # Улучшенный сплиттер с учетом типа документа
    if any(ext in filename.lower() for ext in ['.docx', '.doc']):
        # Для Word документов используем более агрессивное разделение
        chunk_size = 600
        chunk_overlap = 80
    else:
        # Для PDF более консервативное
        chunk_size = 800
        chunk_overlap = 100
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    return text_splitter.split_text(text)

def create_vectorstore(texts, metadatas=None):
    """Создаёт или обновляет векторную базу знаний."""
    vectorstore = None

    if os.path.exists(VECTORSTORE_PATH):
        try:
            vectorstore = Chroma(
                persist_directory=VECTORSTORE_PATH,
                embedding_function=embeddings
            )
        except Exception as e:
            print(f"Ошибка загрузки существующей базы: {e}")
            vectorstore = None

    if vectorstore is None:
        # Используем умное разделение для первого документа
        if texts and metadatas:
            filename = metadatas[0].get("source", "") if metadatas else ""
            chunks = smart_text_splitter(texts[0], filename)
            # Создаем документы с метаданными
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = metadatas[0].copy() if metadatas else {}
                metadata['chunk_id'] = i
                documents.append(chunk)
        else:
            documents = texts
            
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory=VECTORSTORE_PATH,
            metadatas=metadatas
        )
        vectorstore.persist()
        print("✅ Создана новая векторная база знаний")
    else:
        # Добавляем новые документы с умным разделением
        if texts and metadatas:
            filename = metadatas[0].get("source", "") if metadatas else ""
            chunks = smart_text_splitter(texts[0], filename)
            for i, chunk in enumerate(chunks):
                metadata = metadatas[0].copy() if metadatas else {}
                metadata['chunk_id'] = i
                vectorstore.add_texts([chunk], [metadata])
        else:
            vectorstore.add_texts(texts, metadatas=metadatas)
        
        print("✅ Добавлены новые документы в существующую базу")

    return vectorstore

def load_vectorstore():
    """Загружает существующую векторную базу знаний"""
    if not os.path.exists(VECTORSTORE_PATH):
        raise ValueError("База знаний не найдена. Сначала загрузите лекцию.")

    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )