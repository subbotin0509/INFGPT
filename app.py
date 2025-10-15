# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import json
import time
import glob
import shutil
import secrets
import hashlib
import re
import logging
import traceback
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.utils import secure_filename
from utils import extract_text_from_pdf, extract_text_from_docx, highlight_text
from vectorstore import create_vectorstore, load_vectorstore
from chatbot import hybrid_answer, evaluate_answer_quality, update_answer_quality_with_feedback

# Настройка логирования
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

setup_logging()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['SECRET_KEY'] = secrets.token_hex(32)  # Безопасный случайный ключ
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Создаём папки, если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('vectorstore', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# Глобальные переменные
vectorstore = None
_vectorstore_loaded = False
_db_initialized = False  # Флаг инициализации БД

# Простая ролевая модель (только преподаватель требует логина)
TEACHER_PASSWORD = 'teacher123'

# Трекер использования
class UsageTracker:
    def __init__(self):
        self.questions_asked = 0
        self.cache_hits = 0
        self.web_searches = 0
        self.user_likes = 0
        self.user_dislikes = 0
        
    def track_question(self, used_cache=False, used_web=False):
        self.questions_asked += 1
        if used_cache:
            self.cache_hits += 1
        if used_web:
            self.web_searches += 1
    
    def track_feedback(self, feedback):
        if feedback == "like":
            self.user_likes += 1
        elif feedback == "dislike":
            self.user_dislikes += 1
    
    def get_stats(self):
        cache_hit_rate = (self.cache_hits / self.questions_asked * 100) if self.questions_asked > 0 else 0
        total_feedback = self.user_likes + self.user_dislikes
        satisfaction_rate = (self.user_likes / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            'total_questions': self.questions_asked,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'web_searches': self.web_searches,
            'user_likes': self.user_likes,
            'user_dislikes': self.user_dislikes,
            'satisfaction_rate': round(satisfaction_rate, 2)
        }

tracker = UsageTracker()

# Расширенная система мониторинга
class AdvancedMonitoring:
    def __init__(self):
        self.setup_monitoring_db()
    
    def setup_monitoring_db(self):
        """Создает базу данных для мониторинга"""
        conn = sqlite3.connect('monitoring.db')
        cursor = conn.cursor()
        
        # Таблица ошибок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                user_ip TEXT,
                endpoint TEXT,
                request_data TEXT
            )
        ''')
        
        # Таблица метрик качества - ОБНОВЛЕННАЯ ВЕРСИЯ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question TEXT,
                answer_quality TEXT,
                relevance_score REAL,
                completeness_score REAL,
                confidence_score REAL,
                overall_score REAL,
                used_web BOOLEAN,
                response_time REAL,
                user_feedback TEXT,
                cache_key TEXT
            )
        ''')
        
        # Проверяем существование колонки cache_key и добавляем если её нет
        try:
            cursor.execute("PRAGMA table_info(quality_metrics)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'cache_key' not in columns:
                cursor.execute('ALTER TABLE quality_metrics ADD COLUMN cache_key TEXT')
                logging.info("✅ Добавлена колонка cache_key в таблицу quality_metrics")
        except Exception as e:
            logging.error(f"Ошибка при проверке структуры таблицы: {e}")
        
        conn.commit()
        conn.close()
        logging.info("Monitoring database tables created/verified")

    def log_error(self, error_type, error_message, user_ip, endpoint, request_data=None):
        """Логирование ошибок с деталями"""
        try:
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO errors (timestamp, error_type, error_message, stack_trace, user_ip, endpoint, request_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                error_type,
                str(error_message),
                traceback.format_exc(),
                user_ip,
                endpoint,
                json.dumps(request_data) if request_data else None
            ))
            
            conn.commit()
            conn.close()
            logging.error(f"Error logged: {error_type} - {error_message}")
        except Exception as e:
            logging.error(f"Failed to log error: {e}")

    def log_quality_metric(self, question, answer_quality, relevance_score, completeness_score, 
                          confidence_score, overall_score, used_web, response_time, 
                          user_feedback=None, cache_key=None):
        """Логирование метрик качества ответов"""
        try:
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quality_metrics 
                (timestamp, question, answer_quality, relevance_score, completeness_score, 
                 confidence_score, overall_score, used_web, response_time, user_feedback, cache_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                question[:500],
                answer_quality,
                relevance_score,
                completeness_score,
                confidence_score,
                overall_score,
                used_web,
                response_time,
                user_feedback,
                cache_key
            ))
            
            conn.commit()
            conn.close()
            logging.info(f"Quality metric logged for question: {question[:50]}...")
        except Exception as e:
            logging.error(f"Failed to log quality metric: {e}")

    def update_quality_feedback(self, cache_key, user_feedback):
        """Обновляет оценку качества на основе фидбэка пользователя"""
        try:
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()
            
            # Находим запись по cache_key
            cursor.execute('''
                SELECT id, overall_score 
                FROM quality_metrics 
                WHERE cache_key = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (cache_key,))
            
            result = cursor.fetchone()
            if result:
                record_id, old_score = result
                
                # Обновляем оценку на основе фидбэка
                feedback_multiplier = 1.2 if user_feedback == "like" else 0.8
                new_overall_score = min(1.0, max(0.1, old_score * feedback_multiplier))
                
                # Обновляем категорию качества
                if new_overall_score >= 0.8:
                    new_quality = "excellent"
                elif new_overall_score >= 0.6:
                    new_quality = "good"
                elif new_overall_score >= 0.4:
                    new_quality = "fair"
                else:
                    new_quality = "poor"
                
                # Обновляем запись
                cursor.execute('''
                    UPDATE quality_metrics 
                    SET answer_quality = ?, overall_score = ?, user_feedback = ?
                    WHERE id = ?
                ''', (new_quality, new_overall_score, user_feedback, record_id))
                
                conn.commit()
                logging.info(f"Updated quality for cache_key {cache_key}: {old_score:.2f} -> {new_overall_score:.2f} ({new_quality})")
                conn.close()
                return True
            else:
                logging.warning(f"No record found for cache_key: {cache_key}")
                conn.close()
                return False
                
        except Exception as e:
            logging.error(f"Failed to update quality feedback: {e}")
            return False

# Инициализация мониторинга
monitoring = AdvancedMonitoring()

# Декораторы безопасности
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'teacher':
            return jsonify({'error': 'Требуется аутентификация'}), 401
        return f(*args, **kwargs)
    return decorated_function

def error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            client_ip = get_client_ip()
            monitoring.log_error(
                type(e).__name__,
                str(e),
                client_ip,
                request.endpoint,
                request.get_json(silent=True)
            )
            logging.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": "Внутренняя ошибка сервера"}), 500
    return decorated_function

def log_action(action, user_ip, details=""):
    """Логирование действий пользователей"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {user_ip} - {action} - {details}\n"
    
    with open('logs/actions.log', 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    logging.info(f"Action: {action} - IP: {user_ip} - Details: {details}")

# Rate Limiting (простая реализация)
class RateLimiter:
    def __init__(self):
        self.requests = {}
        
    def is_allowed(self, ip, limit=10, window=60):
        now = time.time()
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Удаляем старые запросы
        self.requests[ip] = [req_time for req_time in self.requests[ip] if now - req_time < window]
        
        # Проверяем лимит
        if len(self.requests[ip]) < limit:
            self.requests[ip].append(now)
            return True
        return False

rate_limiter = RateLimiter()

def get_client_ip():
    """Получение IP клиента"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_uploaded_file(file):
    """Проверка безопасности загружаемого файла"""
    # Проверка размера
    file.seek(0, 2)  # Перемещаемся в конец файла
    file_size = file.tell()
    file.seek(0)  # Возвращаемся в начало
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        return False, "Файл слишком большой (макс. 50MB)"
    
    # Базовая проверка типа файла
    if file.filename.endswith('.pdf'):
        # Проверка сигнатуры PDF
        signature = file.read(4)
        file.seek(0)
        if signature != b'%PDF':
            return False, "Некорректный PDF файл"
    
    return True, "OK"

def sanitize_input(text):
    """Очистка пользовательского ввода"""
    if not text or len(text) > 1000:
        return None
    
    # Удаление потенциально опасных символов
    cleaned = re.sub(r'[<>{}]', '', text)
    return cleaned.strip() if cleaned.strip() else None

def get_cache_key(question):
    """Генерирует ключ для кэша на основе вопроса и состояния файлов"""
    files_hash = get_current_files_hash()
    base_key = f"{question}_{files_hash}"
    return hashlib.md5(base_key.encode()).hexdigest()

def get_current_files_hash():
    """Хэш текущего состояния файлов для инвалидации кэша"""
    try:
        files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
        content = "".join(files)
        return hashlib.md5(content.encode()).hexdigest()
    except:
        return "error"

def save_to_cache(question, answer_data):
    """Сохраняет ответ в кэш"""
    cache_key = get_cache_key(question)
    cache_file = f"cache/{cache_key}.json"
    
    cache_data = {
        'question': question,
        'answer_data': answer_data,
        'timestamp': time.time(),
        'cache_key': cache_key
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Response cached with key: {cache_key}")
    except Exception as e:
        logging.error(f"Ошибка сохранения в кэш: {e}")

def load_from_cache(question):
    """Загружает ответ из кэша"""
    cache_key = get_cache_key(question)
    cache_file = f"cache/{cache_key}.json"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                # Проверяем, не устарел ли кэш (1 час)
                if time.time() - cache_data['timestamp'] < 3600:
                    logging.info(f"Cache hit for key: {cache_key}")
                    return cache_data
                else:
                    logging.info(f"Cache expired for key: {cache_key}")
                    os.remove(cache_file)  # Удаляем устаревший кэш
        except Exception as e:
            logging.error(f"Ошибка загрузки из кэша: {e}")
    return None

def format_formulas(text):
    """Форматирует математические формулы для красивого отображения"""
    replacements = [
        (r'\$\$(.*?)\$\$', r'<div class="math-formula">\1</div>'),
        (r'\$(.*?)\$', r'<span class="math-inline">\1</span>'),
        (r'\\\[(.*?)\\\]', r'<div class="math-formula">\1</div>'),
        (r'\\\((.*?)\\\)', r'<span class="math-inline">\1</span>'),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return text

def get_system_stats():
    """Собирает статистику системы"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        files = [f for f in files if f.lower().endswith(('.pdf', '.docx'))]
        
        cache_files = glob.glob('cache/*.json')
        total_cache_size = sum(os.path.getsize(f) for f in cache_files)
        
        vectorstore_size = 0
        if os.path.exists('vectorstore'):
            for root, dirs, files in os.walk('vectorstore'):
                for file in files:
                    vectorstore_size += os.path.getsize(os.path.join(root, file))
        
        # Получаем использование диска
        total, used, free = shutil.disk_usage("/")
        
        # Статистика использования
        usage_stats = tracker.get_stats()
        
        stats = {
            'total_files': len(files),
            'total_cache_entries': len(cache_files),
            'cache_size_mb': round(total_cache_size / (1024 * 1024), 2),
            'vectorstore_size_mb': round(vectorstore_size / (1024 * 1024), 2),
            'disk_usage_gb': round(used / (1024**3), 2),
            'disk_free_gb': round(free / (1024**3), 2),
            'disk_total_gb': round(total / (1024**3), 2),
            'system_uptime': get_system_uptime(),
            'usage_stats': usage_stats
        }
        
        return stats
    except Exception as e:
        logging.error(f"Error getting system stats: {e}")
        return {
            'total_files': 0,
            'total_cache_entries': 0,
            'cache_size_mb': 0,
            'vectorstore_size_mb': 0,
            'disk_usage_gb': 0,
            'disk_free_gb': 0,
            'disk_total_gb': 0,
            'system_uptime': 'Недоступно',
            'usage_stats': tracker.get_stats()
        }

def get_system_uptime():
    """Возвращает время работы системы"""
    try:
        if os.path.exists('/proc/uptime'):
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                return f"{hours}ч {minutes}м"
        # Для Windows и других систем
        return "Недоступно"
    except:
        return "Недоступно"

def rebuild_vectorstore():
    """Перестраивает векторную базу из всех файлов"""
    global vectorstore
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        texts = []
        metadatas = []
        
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx')):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                else:
                    text = extract_text_from_docx(filepath)
                
                if text.strip():
                    texts.append(text)
                    metadatas.append({"source": filename})
        
        if texts:
            vectorstore = create_vectorstore(texts, metadatas=metadatas)
            logging.info(f"Векторная база перестроена с {len(texts)} документами")
            return True
        return False
    except Exception as e:
        logging.error(f"Error rebuilding vectorstore: {e}")
        return False

def get_quality_text(quality):
    """Преобразует код качества в текст"""
    quality_map = {
        'excellent': 'Отлично',
        'good': 'Хорошо', 
        'fair': 'Удовлетворительно',
        'poor': 'Плохо'
    }
    return quality_map.get(quality, 'Неизвестно')

@app.before_request
def load_existing_vectorstore():
    """Загружает существующую базу знаний при первом запросе"""
    global vectorstore, _vectorstore_loaded
    if _vectorstore_loaded:
        return

    try:
        if os.path.exists("./vectorstore/chroma.sqlite3"):
            vectorstore = load_vectorstore()
            logging.info("✅ Загружена существующая векторная база знаний")
            _vectorstore_loaded = True
    except Exception as e:
        logging.warning(f"⚠️ Не удалось загрузить существующую базу: {e}")

# Инициализация базы данных при первом запросе
@app.before_request
def initialize_database():
    """Инициализация базы данных при первом запросе"""
    global _db_initialized
    if not _db_initialized:
        try:
            monitoring.setup_monitoring_db()
            _db_initialized = True
            logging.info("✅ База данных мониторинга инициализирована")
        except Exception as e:
            logging.error(f"❌ Ошибка инициализации базы данных: {e}")

# Health check endpoint
@app.route('/api/health')
def health_check():
    """Проверка здоровья системы"""
    client_ip = get_client_ip()
    log_action("health_check", client_ip)
    
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'vectorstore_loaded': _vectorstore_loaded,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'cache_files_count': len(glob.glob('cache/*.json')),
        'system_uptime': get_system_uptime(),
        'db_initialized': _db_initialized
    }
    
    # Проверка доступности векторной базы
    try:
        if vectorstore:
            test_results = vectorstore.similarity_search("test", k=1)
            status['vectorstore_working'] = True
        else:
            status['vectorstore_working'] = False
    except Exception as e:
        status['vectorstore_working'] = False
        status['status'] = 'degraded'
        status['error'] = str(e)
    
    return jsonify(status)

@app.route('/')
def index():
    client_ip = get_client_ip()
    log_action("page_view", client_ip, "index")
    return render_template('index.html')

@app.route('/teacher_login', methods=['POST'])
def teacher_login():
    client_ip = get_client_ip()
    data = request.get_json()
    password = data.get('password', '').strip()
    
    if password == TEACHER_PASSWORD:
        session['role'] = 'teacher'
        log_action("teacher_login", client_ip, "success")
        return jsonify({'success': True})
    else:
        log_action("teacher_login", client_ip, "failed")
        return jsonify({'success': False, 'error': 'Неверный пароль преподавателя'})

@app.route('/logout')
def logout():
    client_ip = get_client_ip()
    session.clear()
    log_action("logout", client_ip)
    return jsonify({'success': True})

@app.route('/user_info')
def user_info():
    """Возвращает информацию о текущем пользователе"""
    role = session.get('role', 'student')
    return jsonify({'role': role})

# Админ-панель
@app.route('/admin')
@login_required
def admin_dashboard():
    """Панель администратора для преподавателей"""
    client_ip = get_client_ip()
    log_action("admin_access", client_ip)
    
    # Получаем реальные файлы
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [f for f in files if f.lower().endswith(('.pdf', '.docx'))]
    
    # Получаем статистику с ПРАВИЛЬНЫМ количеством файлов
    stats = get_system_stats()
    stats['total_files'] = len(files)
    
    # Получаем информацию о файлах
    file_info = []
    for filename in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        size = os.path.getsize(filepath)
        file_info.append({
            'name': filename,
            'size_mb': round(size / (1024 * 1024), 2),
            'upload_date': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getctime(filepath)))
        })
    
    return render_template('admin.html', stats=stats, files=file_info)

@app.route('/admin/analytics')
@login_required
def analytics_dashboard():
    """Страница аналитики"""
    return render_template('analytics.html')

@app.route('/admin/analytics_data')
@login_required
def analytics_data():
    """API для получения данных аналитики"""
    try:
        conn = sqlite3.connect('monitoring.db')
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute('SELECT COUNT(*) FROM quality_metrics')
        total_questions = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(response_time) FROM quality_metrics WHERE response_time > 0')
        avg_response_time = round(cursor.fetchone()[0] or 0, 2)
        
        cursor.execute('SELECT AVG(overall_score) FROM quality_metrics')
        avg_quality_score = round((cursor.fetchone()[0] or 0) * 100, 1)
        
        # Распределение качества
        cursor.execute('''
            SELECT answer_quality, COUNT(*) 
            FROM quality_metrics 
            GROUP BY answer_quality
        ''')
        quality_data = cursor.fetchall()
        quality_distribution = [0, 0, 0, 0]  # excellent, good, fair, poor
        for quality, count in quality_data:
            if quality == 'excellent':
                quality_distribution[0] = count
            elif quality == 'good':
                quality_distribution[1] = count
            elif quality == 'fair':
                quality_distribution[2] = count
            elif quality == 'poor':
                quality_distribution[3] = count
        
        # Источники ответов
        cursor.execute('SELECT COUNT(*) FROM quality_metrics WHERE used_web = 0')
        lecture_answers = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM quality_metrics WHERE used_web = 1')
        web_answers = cursor.fetchone()[0]
        
        # Оценки пользователей
        cursor.execute('SELECT COUNT(*) FROM quality_metrics WHERE user_feedback = "like"')
        user_likes = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM quality_metrics WHERE user_feedback = "dislike"')
        user_dislikes = cursor.fetchone()[0]
        total_feedback = user_likes + user_dislikes
        satisfaction_rate = round((user_likes / total_feedback * 100), 1) if total_feedback > 0 else 0
        
        # Последние вопросы
        cursor.execute('''
            SELECT question, answer_quality, response_time, used_web, timestamp, user_feedback
            FROM quality_metrics 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_questions = []
        for row in cursor.fetchall():
            recent_questions.append({
                'question': row[0][:100] + '...' if len(row[0]) > 100 else row[0],
                'quality': row[1],
                'quality_text': get_quality_text(row[1]),
                'response_time': round(row[2], 2),
                'source': 'Интернет' if row[3] else 'Лекции',
                'timestamp': row[4][11:16],  # только время
                'user_feedback': row[5] or 'нет оценки'
            })
        
        # Топ вопросов
        cursor.execute('''
            SELECT question, COUNT(*) as count
            FROM quality_metrics 
            GROUP BY question 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        top_questions = []
        for row in cursor.fetchall():
            top_questions.append({
                'question': row[0][:50] + '...' if len(row[0]) > 50 else row[0],
                'count': row[1]
            })
        
        # Вопросы с низким качеством
        cursor.execute('''
            SELECT question, answer_quality, overall_score
            FROM quality_metrics 
            WHERE overall_score < 0.5 
            ORDER BY overall_score ASC 
            LIMIT 5
        ''')
        low_quality_questions = []
        for row in cursor.fetchall():
            low_quality_questions.append({
                'question': row[0][:80] + '...' if len(row[0]) > 80 else row[0],
                'quality': row[1],
                'score': round(row[2], 2)
            })
        
        conn.close()
        
        return jsonify({
            'total_questions': total_questions,
            'avg_response_time': avg_response_time,
            'avg_quality_score': avg_quality_score,
            'success_rate': round((quality_distribution[0] + quality_distribution[1]) / total_questions * 100, 1) if total_questions > 0 else 0,
            'quality_distribution': quality_distribution,
            'lecture_answers': lecture_answers,
            'web_answers': web_answers,
            'user_likes': user_likes,
            'user_dislikes': user_dislikes,
            'satisfaction_rate': satisfaction_rate,
            'recent_questions': recent_questions,
            'top_questions': top_questions,
            'low_quality_questions': low_quality_questions
        })
        
    except Exception as e:
        logging.error(f"Error in analytics data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/delete_file', methods=['POST'])
@login_required
@error_handler
def admin_delete_file():
    """Удаляет файл из системы"""
    client_ip = get_client_ip()
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'error': 'Не указано имя файла'})
    
    try:
        # Защита от path traversal
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            log_action("file_deleted", client_ip, filename)
            
            # Перестраиваем векторную базу
            success = rebuild_vectorstore()
            
            return jsonify({
                'success': True, 
                'message': f'Файл {filename} удален',
                'vectorstore_rebuilt': success
            })
        else:
            return jsonify({'success': False, 'error': 'Файл не найден'})
            
    except Exception as e:
        log_action("file_delete_error", client_ip, f"{filename}: {str(e)}")
        monitoring.log_error("delete_file", str(e), client_ip, "/admin/delete_file", {"filename": filename})
        return jsonify({'success': False, 'error': f'Ошибка при удалении: {str(e)}'})

@app.route('/admin/clear_cache', methods=['POST'])
@login_required
@error_handler
def admin_clear_cache():
    """Очищает кэш ответов"""
    client_ip = get_client_ip()
    
    try:
        cache_files = glob.glob('cache/*.json')
        deleted_count = 0
        
        for cache_file in cache_files:
            os.remove(cache_file)
            deleted_count += 1
        
        log_action("cache_cleared", client_ip, f"deleted {deleted_count} files")
        
        return jsonify({
            'success': True, 
            'message': f'Кэш очищен. Удалено записей: {deleted_count}',
            'deleted_count': deleted_count
        })
    except Exception as e:
        log_action("cache_clear_error", client_ip, str(e))
        monitoring.log_error("clear_cache", str(e), client_ip, "/admin/clear_cache")
        return jsonify({'success': False, 'error': f'Ошибка при очистке кэша: {str(e)}'})

@app.route('/admin/rebuild_vectorstore', methods=['POST'])
@login_required
@error_handler
def admin_rebuild_vectorstore():
    """Принудительно перестраивает векторную базу"""
    client_ip = get_client_ip()
    
    try:
        success = rebuild_vectorstore()
        if success:
            log_action("vectorstore_rebuilt", client_ip, "success")
            return jsonify({'success': True, 'message': 'Векторная база перестроена'})
        else:
            log_action("vectorstore_rebuild_failed", client_ip, "no documents")
            return jsonify({'success': False, 'error': 'Не удалось перестроить векторную базу'})
    except Exception as e:
        log_action("vectorstore_rebuild_error", client_ip, str(e))
        monitoring.log_error("rebuild_vectorstore", str(e), client_ip, "/admin/rebuild_vectorstore")
        return jsonify({'success': False, 'error': f'Ошибка при перестроении: {str(e)}'})

@app.route('/upload', methods=['POST'])
@login_required
@error_handler
def upload_file():
    client_ip = get_client_ip()
    global vectorstore, _vectorstore_loaded

    if 'file' not in request.files:
        return "Нет файла", 400
        
    file = request.files['file']
    if file.filename == '':
        return "Файл не выбран", 400
        
    if file and allowed_file(file.filename):
        # Валидация файла
        is_valid, message = validate_uploaded_file(file)
        if not is_valid:
            log_action("file_upload_rejected", client_ip, f"{file.filename}: {message}")
            return message, 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Извлекаем текст
        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                text = extract_text_from_docx(filepath)

            if not text.strip():
                os.remove(filepath)
                log_action("file_upload_empty", client_ip, filename)
                return "Файл пустой или не удалось извлечь текст", 400

            # Создаём или обновляем векторную базу
            vectorstore = create_vectorstore([text], metadatas=[{"source": filename}])
            _vectorstore_loaded = True
            
            log_action("file_upload_success", client_ip, filename)

            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            os.remove(filepath)
            log_action("file_upload_error", client_ip, f"{filename}: {str(e)}")
            monitoring.log_error("upload_file", str(e), client_ip, "/upload", {"filename": filename})
            return f"Ошибка обработки файла: {str(e)}", 400

    log_action("file_upload_invalid", client_ip, file.filename)
    return "Неверный формат", 400

@app.route('/ask', methods=['POST'])
@error_handler
def ask_question():
    client_ip = get_client_ip()
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip, limit=10, window=60):
        log_action("rate_limit_exceeded", client_ip)
        return jsonify({"error": "Слишком много запросов. Попробуйте позже."}), 429
    
    global vectorstore
    if not vectorstore:
        return jsonify({"error": "Сначала загрузите хотя бы одну лекцию!"})

    data = request.get_json()
    question = data.get('question', '').strip()
    
    # Санитизация ввода
    question = sanitize_input(question)
    if not question:
        return jsonify({"error": "Пустой или некорректный вопрос"})

    log_action("question_asked", client_ip, f"length: {len(question)}")

    # Проверяем кэш
    cached_data = load_from_cache(question)
    if cached_data:
        tracker.track_question(used_cache=True)
        log_action("cache_hit", client_ip)
        # Возвращаем только данные ответа, без метаданных кэша
        return jsonify(cached_data['answer_data'])

    try:
        start_time = time.time()
        
        # Получаем гибридный ответ (лекции + интернет)
        answer, docs, used_web = hybrid_answer(question, vectorstore)
        tracker.track_question(used_web=used_web)

        response_time = time.time() - start_time
        
        # Оцениваем качество ответа
        quality_metrics = evaluate_answer_quality(question, answer, docs, used_web)
        
        # Генерируем cache_key для связи с фидбэком
        cache_key = get_cache_key(question)
        
        # Логируем метрики качества
        monitoring.log_quality_metric(
            question=question,
            answer_quality=quality_metrics["overall_quality"],
            relevance_score=quality_metrics["relevance_score"],
            completeness_score=quality_metrics["completeness_score"],
            confidence_score=quality_metrics["confidence_score"],
            overall_score=quality_metrics["overall_score"],
            used_web=used_web,
            response_time=response_time,
            cache_key=cache_key
        )

        # Форматируем ответ и источники с поддержкой формул
        formatted_answer = format_formulas(answer)
        
        # Формируем ответ для фронтенда
        response_data = {
            "answer": formatted_answer,
            "used_web": used_web,
            "quality_metrics": quality_metrics,
            "cache_key": cache_key  # Добавляем для фидбэка
        }

        # Добавляем источники ТОЛЬКО если ответ из лекций
        if not used_web and docs:
            sources = []
            for doc in docs:
                formatted_content = format_formulas(doc.page_content)
                highlighted = highlight_text(formatted_content, question)
                sources.append({
                    "content": highlighted,
                    "source": doc.metadata.get("source", "неизвестно")
                })
            response_data["sources"] = sources
        else:
            response_data["sources"] = []

        # Сохраняем в кэш
        save_to_cache(question, response_data)
        log_action("question_answered", client_ip, f"web_search: {used_web}, quality: {quality_metrics['overall_quality']}")

        return jsonify(response_data)

    except Exception as e:
        log_action("question_error", client_ip, str(e))
        monitoring.log_error("ask_question", str(e), client_ip, "/ask", {"question": question})
        return jsonify({"error": f"Ошибка при генерации ответа: {str(e)}"})

@app.route('/feedback', methods=['POST'])
@error_handler
def handle_feedback():
    """Обработка лайков/дизлайков от пользователей"""
    client_ip = get_client_ip()
    data = request.get_json()
    
    cache_key = data.get('cache_key')
    feedback = data.get('feedback')  # 'like' или 'dislike'
    
    logging.info(f"Feedback received - cache_key: {cache_key}, feedback: {feedback}")
    
    if not cache_key or feedback not in ['like', 'dislike']:
        return jsonify({'success': False, 'error': 'Неверные данные'})
    
    try:
        # Обновляем оценку качества в базе данных
        success = monitoring.update_quality_feedback(cache_key, feedback)
        
        if success:
            # Трекаем фидбэк в статистике
            tracker.track_feedback(feedback)
            log_action("user_feedback", client_ip, f"{feedback} for cache_key {cache_key}")
            
            return jsonify({
                'success': True, 
                'message': 'Спасибо за оценку!'
            })
        else:
            # Если не нашли запись по cache_key, создаем новую
            logging.warning(f"Record not found for cache_key: {cache_key}, creating new feedback entry")
            
            # Создаем базовую запись для этого фидбэка
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quality_metrics 
                (timestamp, question, answer_quality, relevance_score, completeness_score, 
                 confidence_score, overall_score, used_web, response_time, user_feedback, cache_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                "Feedback only",
                "good" if feedback == "like" else "poor",
                0.7 if feedback == "like" else 0.3,
                0.7 if feedback == "like" else 0.3,
                0.7 if feedback == "like" else 0.3,
                0.7 if feedback == "like" else 0.3,
                False,
                0.0,
                feedback,
                cache_key
            ))
            
            conn.commit()
            conn.close()
            
            # Трекаем фидбэк в статистике
            tracker.track_feedback(feedback)
            log_action("user_feedback_new", client_ip, f"{feedback} for cache_key {cache_key}")
            
            return jsonify({
                'success': True, 
                'message': 'Спасибо за оценку!'
            })
            
    except Exception as e:
        log_action("feedback_error", client_ip, str(e))
        monitoring.log_error("handle_feedback", str(e), client_ip, "/feedback", data)
        return jsonify({'success': False, 'error': f'Ошибка обработки фидбэка: {str(e)}'})

@app.route('/get_files')
@error_handler
def get_files():
    """Возвращает список загруженных файлов"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        files = [f for f in files if f.lower().endswith(('.pdf', '.docx'))]
        return jsonify(files)
    except Exception as e:
        logging.error(f"Error getting files: {e}")
        return jsonify([])

@app.route('/api/usage_stats')
@login_required
@error_handler
def get_usage_stats():
    """API для получения статистики использования"""
    return jsonify(tracker.get_stats())

if __name__ == '__main__':
    logging.info("🚀 Запуск приложения INFGPT с системой пользовательских оценок")
    # Принудительная инициализация базы данных
    try:
        monitoring.setup_monitoring_db()
        logging.info("✅ База данных мониторинга инициализирована")
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации базы данных: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)