
FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV PYTHONPATH="/app"

CMD ["python", "-m", "src.main"]
