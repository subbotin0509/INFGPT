# chatbot.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
import re
import logging

# Инициализация LLM и поиска
llm = OllamaLLM(model="qwen2:0.5b", temperature=0.2)
search = DuckDuckGoSearchRun()

PROMPT_TEMPLATE = """
Ты — помощник студентов. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации — скажи: "В материалах лекций этого нет."

Контекст:
{context}

Вопрос: {question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])

def create_qa_chain(vectorstore):
    """Создаёт стандартную RAG-цепочку"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def clean_text(text):
    """Очищает текст от китайских символов и лишних пробелов"""
    # Удаляем китайские, японские, корейские символы и другие не-кириллические/латинские
    cleaned = re.sub(r'[^\x00-\x7F\u0400-\u04FF\u0500-\u052F\s\.\,\!\?\-\:\;\(\)\d]', '', text)
    # Убираем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def is_relevant_document(doc_content, question):
    """Проверяет, релевантен ли документ вопросу"""
    doc_lower = doc_content.lower()
    question_lower = question.lower()
    
    # Разбиваем вопрос на ключевые слова
    keywords = re.findall(r'\b\w+\b', question_lower)
    
    # Убираем стоп-слова
    stop_words = {'что', 'такое', 'как', 'для', 'на', 'в', 'с', 'по', 'о', 'об', 'из', 'от', 'до', 'за', 'и', 'или', 'но', 'не', 'нет', 'да', 'есть', 'быть', 'какой', 'какая', 'какое', 'какие'}
    keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]
    
    # Если нет ключевых слов, считаем документ НЕрелевантным
    if not keywords:
        return False
    
    # Проверяем наличие ключевых слов в документе
    keyword_matches = sum(1 for keyword in keywords if keyword in doc_lower)
    
    # Документ релевантен если есть хотя бы 50% совпадений ключевых слов
    # И документ не слишком короткий
    min_required_matches = max(1, len(keywords) // 2)
    return keyword_matches >= min_required_matches and len(doc_content.strip()) > 20

def hybrid_answer(question, vectorstore):
    """Гибридный ответ: сначала лекции, потом интернет"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)

    print(f"🔍 Найдено документов: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"Документ {i+1}: {doc.page_content[:100]}...")
        print(f"Релевантен: {is_relevant_document(doc.page_content, question)}")
        print("---")

    # Фильтруем только релевантные документы
    relevant_docs = []
    if docs:
        for doc in docs:
            if is_relevant_document(doc.page_content, question):
                relevant_docs.append(doc)

    print(f"✅ Релевантных документов: {len(relevant_docs)}")

    # Если есть релевантные документы, используем лекции
    if relevant_docs:
        context = format_docs(relevant_docs)
        formatted_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt)
        
        # Проверяем, действительно ли ответ основан на контексте
        if "в материалах лекций этого нет" in answer.lower() or "нет информации" in answer.lower():
            print("🔄 LLM сказал, что в лекциях нет информации, переключаемся на интернет")
            return get_internet_answer(question), [], True
        else:
            return answer, relevant_docs, False
    else:
        # Ищем в интернете
        print("🔄 В лекциях нет релевантной информации, ищем в интернете")
        return get_internet_answer(question), [], True

def get_internet_answer(question):
    """Получает ответ из интернета"""
    try:
        web_results = search.run(question)
        # Очищаем результат от китайских символов
        cleaned_web_results = clean_text(web_results)
        
        # Если после очистки слишком мало текста, пробуем еще раз
        if len(cleaned_web_results) < 30:
            web_results = search.run(question + " на русском")
            cleaned_web_results = clean_text(web_results)
        
        # Специальный промпт для интернет-ответов
        internet_prompt = f"""
        Ты — помощник студентов. Ответь на вопрос на основе информации из интернета.
        Отвечай кратко и понятно на русском языке.

        Вопрос: {question}

        Информация из интернета:
        {cleaned_web_results}

        Твой ответ:
        """
        
        answer = llm.invoke(internet_prompt)
        # Очищаем ответ от возможных китайских символов
        answer = clean_text(answer)
        return answer
        
    except Exception as e:
        print(f"❌ Ошибка при поиске в интернете: {e}")
        return "Извините, не удалось найти информацию ни в лекциях, ни в интернете."
    
def evaluate_answer_quality(question, answer, docs, used_web):
    """Оценка качества ответа"""
    
    quality_metrics = {
        "relevance_score": 0.0,
        "completeness_score": 0.0,
        "confidence_score": 0.0,
        "overall_score": 0.0,
        "overall_quality": "unknown"
    }
    
    try:
        # 1. Оценка релевантности
        relevance_score = calculate_relevance_score(question, answer)
        
        # 2. Оценка полноты
        completeness_score = calculate_completeness_score(question, answer)
        
        # 3. Оценка уверенности (на основе источников)
        confidence_score = calculate_confidence_score(docs, used_web)
        
        # Общий балл
        overall_score = (relevance_score * 0.4 + 
                        completeness_score * 0.3 + 
                        confidence_score * 0.3)
        
        # Определение качества
        if overall_score >= 0.8:
            quality = "excellent"
        elif overall_score >= 0.6:
            quality = "good"
        elif overall_score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"
            
        quality_metrics.update({
            "relevance_score": round(relevance_score, 2),
            "completeness_score": round(completeness_score, 2),
            "confidence_score": round(confidence_score, 2),
            "overall_score": round(overall_score, 2),
            "overall_quality": quality
        })
        
    except Exception as e:
        logging.error(f"Error in quality evaluation: {e}")
    
    return quality_metrics

def calculate_relevance_score(question, answer):
    """Вычисление релевантности ответа вопросу"""
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Разбиваем на ключевые слова
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    answer_words = set(re.findall(r'\b\w+\b', answer_lower))
    
    # Убираем стоп-слова
    stop_words = {'что', 'как', 'почему', 'зачем', 'кто', 'где', 'когда', 'какой'}
    question_words = question_words - stop_words
    answer_words = answer_words - stop_words
    
    if not question_words:
        return 0.0
    
    # Совпадение ключевых слов
    common_words = question_words.intersection(answer_words)
    word_similarity = len(common_words) / len(question_words)
    
    # Штраф за стандартные ответы об отсутствии информации
    if "в материалах лекций этого нет" in answer_lower:
        word_similarity *= 0.1
    
    return min(word_similarity, 1.0)

def calculate_completeness_score(question, answer):
    """Оценка полноты ответа"""
    answer_lower = answer.lower()
    
    # Признаки неполного ответа
    negative_indicators = [
        "не знаю", "не уверен", "не могу найти", 
        "нет информации", "мало информации"
    ]
    
    # Признаки полного ответа
    positive_indicators = [
        "во-первых", "во-вторых", "с одной стороны", 
        "с другой стороны", "например", "таким образом"
    ]
    
    completeness = 0.5  # Базовый балл
    
    # Штрафы за негативные индикаторы
    for indicator in negative_indicators:
        if indicator in answer_lower:
            completeness -= 0.2
    
    # Бонусы за позитивные индикаторы
    for indicator in positive_indicators:
        if indicator in answer_lower:
            completeness += 0.1
    
    # Учитываем длину ответа (но не слишком длинные)
    answer_length = len(answer.split())
    if 20 <= answer_length <= 300:
        completeness += 0.1
    elif answer_length < 10:
        completeness -= 0.2
    
    return max(0.0, min(completeness, 1.0))

def calculate_confidence_score(docs, used_web):
    """Оценка уверенности в ответе на основе источников"""
    if used_web:
        return 0.6  # Средняя уверенность для интернет-источников
    
    if not docs:
        return 0.3  # Низкая уверенность без источников
    
    # Уверенность растет с количеством релевантных документов
    doc_count = len(docs)
    if doc_count >= 3:
        return 0.9
    elif doc_count == 2:
        return 0.7
    else:
        return 0.5
    
def update_answer_quality_with_feedback(question, user_feedback, previous_metrics):
    """Обновляет метрики качества на основе фидбэка пользователя"""
    
    try:
        # Базовая корректировка на основе лайка/дизлайка
        feedback_multiplier = 1.2 if user_feedback == "like" else 0.8
        
        # Обновляем общий балл
        updated_overall_score = min(1.0, max(0.0, previous_metrics["overall_score"] * feedback_multiplier))
        
        # Обновляем категорию качества
        if updated_overall_score >= 0.8:
            updated_quality = "excellent"
        elif updated_overall_score >= 0.6:
            updated_quality = "good"
        elif updated_overall_score >= 0.4:
            updated_quality = "fair"
        else:
            updated_quality = "poor"
        
        updated_metrics = previous_metrics.copy()
        updated_metrics.update({
            "overall_score": round(updated_overall_score, 2),
            "overall_quality": updated_quality,
            "user_feedback": user_feedback
        })
        
        return updated_metrics
        
    except Exception as e:
        logging.error(f"Error updating quality with feedback: {e}")
        return previous_metrics