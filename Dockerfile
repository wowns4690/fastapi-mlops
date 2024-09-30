# 베이스 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# 컨테이너가 FastAPI 앱을 실행하도록 설정
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
