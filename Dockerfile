FROM python:3.11-slim

# Install system utilities, sudo, build tools, and curl
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Download and install the pre-compiled ttyd binary directly
RUN curl -sLO https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 && \
    chmod +x ttyd.x86_64 && \
    mv ttyd.x86_64 /usr/local/bin/ttyd

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script
COPY terminalsudoku.py .

# Launch ttyd on Render's assigned port ($PORT) running python via sudo
CMD sh -c "ttyd -p ${PORT:-10000} -W sudo python terminalsudoku.py"