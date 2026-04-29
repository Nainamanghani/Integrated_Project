FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY required files (IMPORTANT)
COPY backend ./backend
COPY frontend ./frontend
COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 7860

CMD ["./start.sh"]