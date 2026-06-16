FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV FASTAPI_BACKEND_URL=http://127.0.0.1:8000

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download NLTK data into the directory expected by the application
RUN python -m nltk.downloader -d /home/user/app/data/nltk wordnet omw-1.4 stopwords

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "7860"]
