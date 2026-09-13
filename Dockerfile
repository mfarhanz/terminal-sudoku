FROM python:3.11-slim

# Install system utilities, sudo, build tools, curl, and kbd (provides dumpkeys)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    sudo \
    kbd \
    evtest \
    && rm -rf /var/lib/apt/lists/*

# Download and install the pre-compiled ttyd binary
RUN curl -sLO https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 && \
    chmod +x ttyd.x86_64 && \
    mv ttyd.x86_64 /usr/local/bin/ttyd

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY terminalsudoku.py .

CMD sh -c "ttyd -p ${PORT:-10000} -W sudo python terminalsudoku.py"