FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Instalar dependencias (Vital para evitar fallos de librerías de terceros)
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app"

CMD ["python", "-m", "src.main"]