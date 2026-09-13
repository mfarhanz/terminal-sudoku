FROM python:3.11-slim

# Install system utilities and curl for downloading ttyd
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download and install ttyd binary
RUN curl -sLO https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 && \
    chmod +x ttyd.x86_64 && \
    mv ttyd.x86_64 /usr/local/bin/ttyd

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY terminalsudoku.py .

# Environment variable to force Python stdout/stdin to be unbuffered
ENV PYTHONUNBUFFERED=1

# -W: allow clients to write (keyboard input)
# -p: port configuration
CMD ["sh", "-c", "ttyd -W -p ${PORT:-10000} python3 -u terminalsudoku.py"]