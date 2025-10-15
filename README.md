# 🎓 INFGPT - Интеллектуальный помощник по лекциям

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.347-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Умная система анализа учебных материалов с AI-ассистентом и системой оценки качества ответов**

---

## ✨ Возможности

### 🎯 Для студентов
- **💬 Интеллектуальный чат** - задавайте вопросы по лекциям на естественном языке
- **🔍 Гибридный поиск** - ответы из материалов лекций + интернет-источники
- **📚 Подсветка источников** - точные цитаты из учебных материалов
- **⭐ Оценка качества** - система рейтинга полезности ответов
- **🎨 Поддержка формул** - красивый рендеринг математических выражений

### 👨‍🏫 Для преподавателей
- **📁 Управление материалами** - загрузка PDF/DOCX лекций
- **📊 Панель аналитики** - детальная статистика использования
- **⚙️ Администрирование** - управление кэшем и векторной базой
- **📈 Мониторинг качества** - отслеживание эффективности системы

---

## 📸 Скриншоты

<div align="center">

| Главный интерфейс | Панель администратора | Аналитика |
|:---:|:---:|:---:|
| [![Главный интерфейс](https://via.placeholder.com/400x250/007bff/white?text=Главный+интерфейс)](screenshots/main.png) | [![Панель администратора](https://via.placeholder.com/400x250/28a745/white?text=Админ+панель)](screenshots/admin.png) | [![Аналитика](https://via.placeholder.com/400x250/6f42c1/white?text=Аналитика)](screenshots/analytics.png) |

</div>

---

## 🔄 Алгоритм работы

```mermaid
graph LR
    A[📥 Вопрос студента] --> B[🔍 Поиск в лекциях]
    B --> C{Найдена информация?}
    C -->|✅ Да| D[🤖 Генерация ответа]
    C -->|❌ Нет| E[🌐 Поиск в интернете]
    E --> D
    D --> F[📊 Оценка качества]
    F --> G[💾 Кэширование]
    G --> H[📤 Отправка ответа]


## Архитектура

graph TB
    subgraph "Frontend Layer"
        A[🎨 Веб-интерфейс<br/>Bootstrap + MathJax]
        B[📱 Адаптивный дизайн]
    end
    
    subgraph "Backend Layer"
        C[🚀 Flask Application]
        D[🔐 Аутентификация]
        E[📊 API Routes]
    end
    
    subgraph "AI Processing Layer"
        F[🧠 LangChain Orchestrator]
        G[🤖 Ollama LLM<br/>qwen2:0.5b]
        H[🔍 DuckDuckGo Search]
    end
    
    subgraph "Data Layer"
        I[🗄️ ChromaDB<br/>Векторная база]
        J[💾 SQLite<br/>Мониторинг]
        K[⚡ Кэш ответов]
    end
    
    subgraph "Document Processing"
        L[📄 PDF Parser<br/>PyPDF2]
        M[📝 DOCX Parser<br/>python-docx]
        N[✂️ Text Splitter]
    end
    
    A --> C
    B --> C
    C --> F
    F --> G
    F --> H
    F --> I
    I --> L
    I --> M
    I --> N
    C --> J
    C --> K
    F --> K
