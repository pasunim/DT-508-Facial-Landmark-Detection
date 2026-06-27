FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.138.1 \
    "uvicorn[standard]==0.49.0" \
    opencv-python==4.13.0.92 \
    mediapipe==0.10.35 \
    numpy==2.5.0

COPY face_landmarker.task .
COPY facial_landmark_detection.py .
COPY server.py .
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
