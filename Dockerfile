FROM python:3.11-slim

# Install system dependencies, ttyd, and sudo
RUN apt-get update && apt-get install -y \
    ttyd \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script
COPY terminalsudoku.py .

# Launch ttyd on the port assigned by Render ($PORT), running python via sudo
CMD sh -c "ttyd -p ${PORT:-10000} -W sudo python terminalsudoku.py"