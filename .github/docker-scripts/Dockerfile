# Use a base image with Python 3.11
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
