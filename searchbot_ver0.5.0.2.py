#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os
import logging
import re
from datetime import datetime, timezone
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import asyncio
import nest_asyncio
import pytz
import json

# Jupyter에서 중첩 이벤트 루프 허용
nest_asyncio.apply()

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 디버깅을 위해 DEBUG 레벨로 설정

# 외부 라이브러리 로깅 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# --- 설정값 ---
BOT_TOKEN = "8082917857:AAGJbOW8_nGHaUqQOjJeFxtX9cyLcVsjivM"
DB_FILE = "telegram_bot_messages.db"
FAISS_INDEX_FILE = "faiss_index.bin"
MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# 기본 필터 패턴 (정규표현식) - Raw string 사용
DEFAULT_FILTER_PATTERNS = [
    r'^보고\s*[)）\]]',  # 보고), 보고 ), 보고] 등
]

def format_kst_time(dt_obj):
    """datetime 객체를 한국 시간 문자열로 변환"""
    try:
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        dt_kst = dt_obj.astimezone(KST)
        return dt_kst.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"시간 변환 오류: {e}")
        return str(dt_obj)


def normalize_dates_in_text(text):
    """텍스트 내의 날짜 형식을 표준화 - 상대적 시간 표현 포함"""
    if not text:
        return text

    # 디버깅용 로그
    original_text = text

    # 현재 연도 가져오기
    current_year = datetime.now().year

    # 1단계: 상대적 시간 표현을 절대적 연도로 변환
    relative_patterns = [
        (r'\b올해\b', str(current_year) + '년'),
        (r'\b금년\b', str(current_year) + '년'),
        (r'\b작년\b', str(current_year - 1) + '년'),
        (r'\b내년\b', str(current_year + 1) + '년'),
        (r'\b재작년\b', str(current_year - 2) + '년'),
        (r'\b내후년\b', str(current_year + 2) + '년'),
    ]

    # 상대적 시간 표현 변환
    normalized_text = text
    for pattern, replacement in relative_patterns:
        normalized_text = re.sub(pattern, replacement, normalized_text)

    # 2단계: 날짜 패턴들을 처리하기 위한 함수
    def replace_date_patterns(text):
        # YYYY.MM.DD 형식 (2023.07.21)
        text = re.sub(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', 
                     lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YY.MM.DD 형식 (23.07.21, 25.03.21)
        text = re.sub(r'\b(\d{2})\.(\d{1,2})\.(\d{1,2})\b', 
                     lambda m: f"20{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YYYY-MM-DD 형식
        text = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', 
                     lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YY-MM-DD 형식
        text = re.sub(r'\b(\d{2})-(\d{1,2})-(\d{1,2})\b', 
                     lambda m: f"20{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YYYY/MM/DD 형식
        text = re.sub(r'(\d{4})/(\d{1,2})/(\d{1,2})', 
                     lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YY/MM/DD 형식
        text = re.sub(r'\b(\d{2})/(\d{1,2})/(\d{1,2})\b', 
                     lambda m: f"20{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YYYY년 MM월 DD일 형식 (이미 표준형이지만 숫자 정규화)
        text = re.sub(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', 
                     lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # YY년 MM월 DD일 형식
        text = re.sub(r'\b(\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일', 
                     lambda m: f"20{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)

        # MM.DD 형식 (연도 추가)
        text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\b(?!\d)', 
                     lambda m: f"{current_year}년 {int(m.group(1))}월 {int(m.group(2))}일", text)

        # MM-DD 형식 (연도 추가)
        text = re.sub(r'\b(\d{1,2})-(\d{1,2})\b(?!\d)', 
                     lambda m: f"{current_year}년 {int(m.group(1))}월 {int(m.group(2))}일", text)

        return text

    # 3단계: 월/일 패턴 처리를 위한 별도 함수
    def replace_month_day_patterns(text):
        # MM월 DD일 형식 - 앞에 연도가 없는 경우 현재 연도 추가
        def add_year_to_month_day(match):
            month = int(match.group(1))
            day = int(match.group(2))
            # 앞뒤 문맥 확인
            start = match.start()
            if start >= 6:
                # 앞에 연도가 있는지 확인
                before_text = text[max(0, start-6):start]
                if re.search(r'\d{4}년\s*$', before_text):
                    return match.group(0)  # 이미 연도가 있으면 그대로 반환
            return f"{current_year}년 {month}월 {day}일"

        text = re.sub(r'\b(\d{1,2})월\s*(\d{1,2})일', add_year_to_month_day, text)

        # M월 형식 (날짜 없이 월만 있는 경우)
        def add_year_to_month(match):
            month = int(match.group(1))
            # 앞뒤 문맥 확인
            start = match.start()
            end = match.end()
            if start >= 6:
                before_text = text[max(0, start-6):start]
                if re.search(r'\d{4}년\s*$', before_text):
                    return match.group(0)
            if end < len(text) - 4:
                after_text = text[end:end+4]
                if re.search(r'^\s*\d{1,2}일', after_text):
                    return match.group(0)
            return f"{current_year}년 {month}월"

        text = re.sub(r'\b(\d{1,2})월\b', add_year_to_month, text)

        return text

    # 날짜 패턴 처리
    normalized_text = replace_date_patterns(normalized_text)
    normalized_text = replace_month_day_patterns(normalized_text)

    # 디버깅: 변환 결과 로그
    if original_text != normalized_text:
        logger.debug(f"날짜 정규화: '{original_text}' → '{normalized_text}'")

    return normalized_text

class SemanticSearchEngine:
    """시맨틱 검색을 담당하는 클래스 (IndexIDMap 사용)"""
    def __init__(self, model_name=MODEL_NAME, index_file=FAISS_INDEX_FILE):
        self.model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = self._load_index()
        logger.info(f"시맨틱 검색 엔진 초기화 완료. 모델: {model_name}, 벡터 차원: {self.dimension}")

    def _load_index(self):
        """FAISS 인덱스 파일을 로드하거나 새로 생성"""
        if os.path.exists(self.index_file):
            logger.info(f"'{self.index_file}'에서 FAISS 인덱스를 로드합니다.")
            return faiss.read_index(self.index_file)
        else:
            logger.info("FAISS 인덱스를 새로 생성합니다. (IndexIDMap, IndexFlatL2 사용)")
            return faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def save_index(self):
        """FAISS 인덱스를 파일에 저장"""
        faiss.write_index(self.index, self.index_file)
        logger.info(f"FAISS 인덱스를 '{self.index_file}'에 저장했습니다.")

    def add_vector(self, db_id, text):
        """텍스트를 벡터로 변환하여 'DB ID'와 함께 인덱스에 추가"""
        if not text: return
        # 날짜 정규화 적용
        normalized_text = normalize_dates_in_text(text)
        embedding = self.model.encode([normalized_text], normalize_embeddings=True)
        self.index.add_with_ids(np.array(embedding, dtype='f4'), np.array([db_id]))
        self.save_index()

    def search(self, query, k=5):
        """쿼리와 유사한 벡터를 검색하여 (거리, DB ID) 리스트 반환"""
        if not query or self.index.ntotal == 0: 
            return [], []

        # 정규화가 이미 되어있는지 확인 (연도가 포함되어 있으면 이미 정규화된 것으로 간주)
        if not re.search(r'\d{4}년', query):
            # 정규화가 필요한 경우에만 수행
            normalized_query = normalize_dates_in_text(query)
            if query != normalized_query:
                logger.debug(f"검색 엔진 내부 정규화: '{query}' → '{normalized_query}'")
        else:
            normalized_query = query

        query_vector = self.model.encode([normalized_query], normalize_embeddings=True)
        distances, db_ids = self.index.search(np.array(query_vector, dtype='f4'), k)

        if db_ids.size == 0 or db_ids[0][0] == -1:
            return [], []

        valid_indices = db_ids[0] != -1
        return distances[0][valid_indices], db_ids[0][valid_indices]

    def rebuild_index_from_db(self, all_messages):
        """DB의 모든 메시지로 FAISS 인덱스를 처음부터 다시 빌드"""
        logger.info(f"DB로부터 FAISS 인덱스를 재구축합니다. 총 {len(all_messages)}개 메시지.")
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

        if not all_messages:
            self.save_index()
            return

        db_ids = np.array([msg[0] for msg in all_messages])
        # 모든 텍스트에 날짜 정규화 적용
        texts = [normalize_dates_in_text(msg[1]) for msg in all_messages]
        embeddings = self.model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True, normalize_embeddings=True
        )

        self.index.add_with_ids(np.array(embeddings, dtype='f4'), db_ids)
        self.save_index()
        logger.info("인덱스 재구축 완료.")


class TelegramBotDB:
    def __init__(self, db_file=DB_FILE):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, message_id INTEGER,
                user_id TEXT, username TEXT, first_name TEXT, message_text TEXT,
                timestamp TEXT, chat_type TEXT, chat_title TEXT,
                UNIQUE(chat_id, message_id)
            )
        ''')

        # 채팅방별 설정을 저장할 테이블 추가
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id TEXT PRIMARY KEY,
                filter_patterns TEXT,
                filter_enabled INTEGER DEFAULT 1,
                updated_at TEXT
            )
        ''')

        self.conn.commit()
        cursor.close()
        logger.info("데이터베이스 초기화 및 영구 연결 수립 완료")

    def save_message(self, chat_id, message_id, user_id, username, first_name, text, timestamp, chat_type, chat_title):
        cursor = self.conn.cursor()
        last_row_id = -1
        try:
            timestamp_kst = format_kst_time(timestamp)
            cursor.execute('''
                INSERT OR IGNORE INTO messages
                (chat_id, message_id, user_id, username, first_name, message_text, timestamp, chat_type, chat_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(chat_id), message_id, str(user_id), username, first_name, text, timestamp_kst, chat_type, chat_title))
            self.conn.commit()

            if cursor.rowcount > 0:
                last_row_id = cursor.lastrowid
                logger.info(f"메시지 저장됨 (ID: {last_row_id}): {text[:30]}...")
            else:
                cursor.execute("SELECT id FROM messages WHERE chat_id = ? AND message_id = ?", (str(chat_id), message_id))
                result = cursor.fetchone()
                if result: last_row_id = result[0]
        except Exception as e:
            logger.error(f"메시지 저장 오류: {e}")
        finally:
            cursor.close()
        return last_row_id

    def get_message_by_id(self, db_id):
        """데이터베이스의 PRIMARY KEY(id)로 메시지 조회"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT message_id, message_text, timestamp, username, first_name, chat_id, chat_type FROM messages WHERE id = ?', (int(db_id),))
        result = cursor.fetchone()
        cursor.close()
        return result

    def get_all_messages_for_indexing(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, message_text FROM messages ORDER BY id')
        results = cursor.fetchall()
        cursor.close()
        return results

    def get_chat_filter_patterns(self, chat_id):
        """특정 채팅방의 필터 패턴 가져오기"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT filter_patterns, filter_enabled FROM chat_settings WHERE chat_id = ?', (str(chat_id),))
        result = cursor.fetchone()
        cursor.close()

        if result:  # filter_enabled 체크 제거
            try:
                return json.loads(result[0]) if result[0] else DEFAULT_FILTER_PATTERNS
            except:
                return DEFAULT_FILTER_PATTERNS
        return DEFAULT_FILTER_PATTERNS

    def set_chat_filter_patterns(self, chat_id, patterns):
        """특정 채팅방의 필터 패턴 설정"""
        cursor = self.conn.cursor()
        timestamp_kst = format_kst_time(datetime.now(timezone.utc))

        # 패턴을 정규표현식으로 변환
        regex_patterns = []
        for pattern in patterns:
            # 특수문자 이스케이프 후 ^ 추가
            escaped_pattern = '^' + re.escape(pattern)
            regex_patterns.append(escaped_pattern)

        patterns_json = json.dumps(regex_patterns)

        cursor.execute('''
            INSERT OR REPLACE INTO chat_settings (chat_id, filter_patterns, filter_enabled, updated_at)
            VALUES (?, ?, 1, ?)
        ''', (str(chat_id), patterns_json, timestamp_kst))

        self.conn.commit()
        cursor.close()

    def toggle_chat_filter(self, chat_id, enabled):
        """필터 기능 활성화/비활성화 - 더 이상 사용하지 않음"""
        pass

    def close_connection(self):
        if self.conn:
            self.conn.close()
            logger.info("데이터베이스 연결이 안전하게 종료되었습니다.")

class TelegramSearchBot:
    def __init__(self, token):
        self.token = token
        self.db = TelegramBotDB()
        self.search_engine = SemanticSearchEngine()
        self.application = Application.builder().token(self.token).build()
        self.sync_db_and_index()
        self.setup_handlers()

    def sync_db_and_index(self):
        logger.info("DB와 FAISS 인덱스 동기화를 시작합니다.")
        all_db_messages = self.db.get_all_messages_for_indexing()
        db_ids = {msg[0] for msg in all_db_messages}

        index_ids = set()
        if self.search_engine.index.ntotal > 0:
            # IndexIDMap에서 모든 ID 가져오기 - 수정된 부분
            try:
                # FAISS IndexIDMap은 직접 ID를 가져올 수 없으므로 재구축
                logger.info("IndexIDMap 동기화를 위해 인덱스를 재구축합니다.")
                self.search_engine.rebuild_index_from_db(all_db_messages)
                return
            except Exception as e:
                logger.error(f"인덱스 동기화 중 오류: {e}")
                self.search_engine.rebuild_index_from_db(all_db_messages)
                return

    def check_message_filter(self, message_text, patterns):
        """메시지가 필터 패턴과 일치하는지 확인"""
        for pattern in patterns:
            try:
                if re.search(pattern, message_text, re.IGNORECASE):
                    return True
            except re.error:
                logger.error(f"잘못된 정규표현식 패턴: {pattern}")
        return False

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("search", self.search))
        self.application.add_handler(CommandHandler("setfilter", self.set_filter))
        self.application.add_handler(CommandHandler("showfilter", self.show_filter))
        self.application.add_handler(CommandHandler("debug_date", self.debug_date))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))

        logger.info("모든 핸들러가 등록되었습니다.")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 환영합니다! (v3.1)\n"
            "이제 메시지를 의미 기반으로 검색할 수 있습니다.\n"
            "/search [검색어]로 사용해보세요.\n\n"
            "📌 필터 기능:\n"
            "- /setfilter : 필터 패턴 설정\n"
            "- /showfilter : 현재 필터 확인\n\n"
            "🆕 날짜 형식이 자동으로 통일됩니다!"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """📖 사용법

🔍 검색: /search [검색할 문장]
예시) /search 지난 분기 매출 알려줘

📌 필터 설정 (관리자만)
- /setfilter [패턴1] [패턴2] ...
- /showfilter : 현재 필터 확인

필터 예시
/setfilter "보고)" "보고 )" "보고]" "[보고]"

🆕 날짜 검색 개선
- 25.07.21, 2025년 7월 21일, 25-07-21 등
- 모든 날짜 형식이 자동으로 통일되어 검색됩니다!
"""
        await update.message.reply_text(help_text)

    async def set_filter(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """필터 패턴 설정 (관리자만)"""
        user = update.effective_user
        chat = update.effective_chat

        # 관리자 권한 확인
        member = await context.bot.get_chat_member(chat.id, user.id)
        if member.status not in ['creator', 'administrator']:
            await update.message.reply_text("⚠️ 이 명령어는 관리자만 사용할 수 있습니다.")
            return

        if not context.args:
            await update.message.reply_text(
                "📌 필터 패턴을 설정해주세요.\n"
                "사용법: /setfilter [패턴1] [패턴2] ...\n"
                '예시: /setfilter "보고)" "보고 )" "보고]"'
            )
            return

        patterns = context.args
        self.db.set_chat_filter_patterns(chat.id, patterns)

        await update.message.reply_text(
            f"✅ 필터 패턴이 설정되었습니다.\n"
            f"설정된 패턴 ({len(patterns)}개):\n" +
            "\n".join([f"- {p}" for p in patterns])
        )

    async def debug_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """날짜 정규화 테스트 명령어"""
        if not context.args:
            await update.message.reply_text(
                "🔍 날짜 정규화 테스트\n"
                "사용법: /debug_date [텍스트]"
            )
            return

        text = " ".join(context.args)
        normalized = normalize_dates_in_text(text)

        response = f"🔍 날짜 정규화 테스트 결과:\n\n"
        response += f"원본: {text}\n"
        response += f"정규화: {normalized}\n\n"

        if text == normalized:
            response += "⚠️ 변환되지 않았습니다."
        else:
            response += "✅ 정규화되었습니다."

        await update.message.reply_text(response)

    async def show_filter(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """현재 필터 패턴 표시"""
        chat = update.effective_chat
        patterns = self.db.get_chat_filter_patterns(chat.id)

        # 사용자에게 보여줄 때는 정규표현식이 아닌 원본 형태로 변환
        display_patterns = []
        for pattern in patterns:
            # ^와 이스케이프 문자 제거
            clean_pattern = pattern.replace('^', '').replace('\\', '')
            display_patterns.append(clean_pattern)

        if patterns == DEFAULT_FILTER_PATTERNS:
            pattern_text = "기본 필터 패턴:\n- 보고)"
        else:
            pattern_text = "설정된 필터 패턴:\n" + "\n".join([f"- {p}" for p in display_patterns])

        await update.message.reply_text(f"📌 현재 필터 설정\n\n{pattern_text}")

async def search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("🔍 검색할 내용을 입력해주세요.\n사용법: /search [내용]")
        return

    query = " ".join(context.args)
    current_chat_id = update.message.chat_id

    # 한 번만 정규화하고 재사용
    normalized_query = normalize_dates_in_text(query)
    if query != normalized_query:
        logger.info(f"검색 쿼리 정규화: '{query}' → '{normalized_query}'")

    # 디버깅 메시지 (필요시에만 표시)
    debug_info = ""
    if query != normalized_query:
        debug_info = f"\n🔍 날짜 정규화: {query} → {normalized_query}"

    status_message = await update.message.reply_text(f"⏳ '{query}'와(과) 유사한 메시지를 찾고 있습니다...{debug_info}")

    # 가장 유사한 메시지 5개 검색
    distances, db_ids = self.search_engine.search(normalized_query, k=5)  # 이미 정규화된 쿼리 전달

    if len(db_ids) == 0:
        await context.bot.edit_message_text(
            text=f"❌ '{query}'에 대한 검색 결과가 없습니다.",
            chat_id=status_message.chat_id, message_id=status_message.message_id
        )
        return

    # 필터 패턴 가져오기
    filter_patterns = self.db.get_chat_filter_patterns(current_chat_id)

    # 디버깅: 상위 5개 결과 모두 출력 (옵션)
    debug_results = ""
    if logger.isEnabledFor(logging.DEBUG):  # DEBUG 레벨일 때만 생성
        debug_results = "\n📊 상위 5개 검색 결과:\n"
        for i, db_id in enumerate(db_ids[:5]):
            msg_data = self.db.get_message_by_id(db_id)
            if msg_data:
                msg_id, text, timestamp, username, first_name, chat_id, chat_type = msg_data
                similarity_score = max(0, 1 - distances[i] / 2) * 100
                debug_results += f"\n{i+1}. 유사도: {similarity_score:.2f}%"
                debug_results += f"\n   원본: {text[:50]}...\n"

    # 필터 조건에 맞는 첫 번째 메시지 찾기
    found_message = None
    found_distance = None

    for i, db_id in enumerate(db_ids):
        msg_data = self.db.get_message_by_id(db_id)
        if msg_data:
            msg_id, text, timestamp, username, first_name, chat_id, chat_type = msg_data

            # 필터 확인
            if self.check_message_filter(text, filter_patterns):
                found_message = msg_data
                found_distance = distances[i]
                break

    if not found_message:
        await context.bot.edit_message_text(
            text=f"❌ '{query}'에 대한 검색 결과 중 필터 조건에 맞는 메시지가 없습니다.{debug_results}",
            chat_id=status_message.chat_id, message_id=status_message.message_id
        )
        return

    msg_id, text, timestamp, username, first_name, chat_id, chat_type = found_message
    display_name = username or first_name or "Unknown"
    preview = text[:200] + "..." if len(text) > 200 else text
    similarity_score = max(0, 1 - found_distance / 2) * 100

    # 메시지 정보 텍스트 구성
    message_text = f"🎯 가장 유사한 메시지를 찾았습니다!\n\n"
    message_text += f"👤 작성자: {display_name}\n"
    message_text += f"📊 유사도: {similarity_score:.2f}%\n"
    message_text += f"📅 시간: {timestamp}\n"
    message_text += f"💬 내용:\n{preview}"

    if debug_results:  # DEBUG 레벨일 때만 추가
        message_text += f"\n{debug_results}"

    # 인라인 키보드 버튼 생성
    keyboard = []

    if chat_type != 'private':
        # 현재 채팅방과 같은 채팅방인지 확인
        if str(chat_id) == str(current_chat_id):
            # 같은 채팅방의 메시지
            channel_id = str(chat_id).replace('-100', '')
            deep_link = f"https://t.me/c/{channel_id}/{msg_id}"

            keyboard.append([InlineKeyboardButton(
                "🚀 메시지로 바로 이동",
                url=deep_link
            )])
        else:
            # 다른 채팅방의 메시지
            channel_id = str(chat_id).replace('-100', '')
            deep_link = f"https://t.me/c/{channel_id}/{msg_id}"

            keyboard.append([InlineKeyboardButton(
                "🚀 메시지로 이동",
                url=deep_link
            )])
    else:
        message_text += "\n\n🔒 개인 채팅 메시지는 이동할 수 없습니다."

    # 메시지 전송
    if keyboard:
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.edit_message_text(
            text=message_text,
            chat_id=status_message.chat_id,
            message_id=status_message.message_id,
            reply_markup=reply_markup
        )
    else:
        await context.bot.edit_message_text(
            text=message_text,
            chat_id=status_message.chat_id,
            message_id=status_message.message_id
        )

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if query.data.startswith("copy_"):
            parts = query.data.split("_")
            chat_id = parts[1]
            message_id = parts[2]

            channel_id = str(chat_id).replace('-100', '')

            copy_text = f"🔗 메시지 링크 정보:\n\n"
            copy_text += f"1️⃣ 다음 링크를 복사해서 메시지창에 붙여넣으세요:\n"
            copy_text += f"`https://t.me/c/{channel_id}/{message_id}`\n\n"
            copy_text += f"2️⃣ 메시지 ID: `{message_id}`\n\n"
            copy_text += f"💡 팁: 봇을 그룹 관리자로 만들면 버튼 클릭으로 바로 이동할 수 있습니다."

            await query.message.reply_text(copy_text, parse_mode='Markdown')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message or not message.text: return

        db_id = self.db.save_message(
            chat_id=message.chat.id, message_id=message.message_id,
            user_id=message.from_user.id, username=message.from_user.username,
            first_name=message.from_user.first_name, text=message.text,
            timestamp=message.date, chat_type=message.chat.type,
            chat_title=message.chat.title
        )

        if db_id != -1:
            self.search_engine.add_vector(db_id, message.text)
            logger.info(f"메시지(DB ID: {db_id})가 벡터 인덱스에 추가되었습니다.")

    async def set_bot_commands(self):
        await self.application.bot.set_my_commands([
            BotCommand("start", "봇 시작"),
            BotCommand("help", "도움말"),
            BotCommand("search", "메시지 검색"),
            BotCommand("setfilter", "필터 설정 (관리자)"),
            BotCommand("showfilter", "필터 확인"),
            BotCommand("debug_date", "날짜 정규화 테스트"),
        ])

    async def run(self):
        logger.info("텔레그램 시맨틱 검색 봇을 시작합니다.")
        try:
            await self.application.initialize()
            await self.set_bot_commands()
            await self.application.start()
            await self.application.updater.start_polling(drop_pending_updates=True)
            logger.info("봇이 성공적으로 시작되었습니다!")
            await asyncio.Event().wait()
        except Exception as e:
            logger.error(f"봇 실행 중 오류 발생: {e}", exc_info=True)
        finally:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            self.db.close_connection()
            logger.info("봇이 종료되었습니다.")

async def run_bot_jupyter():
    """Jupyter 환경에서 봇 실행"""
    if "YOUR_BOT_TOKEN" in BOT_TOKEN:
        print("🚨 봇 토큰(BOT_TOKEN)을 설정해주세요!")
        return
    bot = TelegramSearchBot(BOT_TOKEN)
    await bot.run()

if __name__ == '__main__':
    if "YOUR_BOT_TOKEN" in BOT_TOKEN:
        print("🚨 봇 토큰(BOT_TOKEN)을 설정해주세요!")
    else:
        bot = TelegramSearchBot(BOT_TOKEN)
        asyncio.run(bot.run())

print("✅ 날짜 정규화 기능이 추가된 시맨틱 검색 봇!")
print("Jupyter에서 실행하려면: await run_bot_jupyter()")
print(".py 파일로 실행하려면: python <파일명>.py")


# In[ ]:


get_ipython().system('jupyter nbconvert --to python searchbot_ver0.5.0.2.ipynb')


# In[ ]:




