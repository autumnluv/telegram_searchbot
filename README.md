# telegram_searchbot

텔레그램 메시지를 의미 기반으로 검색할 수 있는 봇입니다.

기능

시맨틱 검색: 문장의 의미를 이해하여 유사한 메시지 검색
필터링: 특정 패턴으로 시작하는 메시지만 검색
메시지 저장: 모든 메시지를 데이터베이스에 저장
벡터 인덱싱: FAISS를 사용한 빠른 검색

명령어

/start - 봇 시작
/help - 도움말 보기
/search [검색어] - 메시지 검색
/setfilter [패턴1] [패턴2]... - 필터 설정 (관리자만)
/showfilter - 현재 필터 확인

Railway 배포 방법

GitHub에 리포지토리 생성 및 파일 업로드
Railway에서 새 프로젝트 생성
GitHub 리포지토리 연결
환경 변수 설정:

BOT_TOKEN: 텔레그램 봇 토큰



환경 변수

BOT_TOKEN: 텔레그램 봇 토큰 (필수)

기술 스택

Python 3.13
python-telegram-bot
sentence-transformers
FAISS
SQLite3

주의사항

봇을 그룹 채팅방에 추가할 때는 메시지 읽기 권한이 필요합니다
검색 성능을 위해 정기적으로 FAISS 인덱스를 최적화하는 것을 권장합니다
