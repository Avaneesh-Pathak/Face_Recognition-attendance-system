# Use Python base image
FROM python:3.10

# Prevent Python from writing .pyc files and using output buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory in container
WORKDIR /app

# Install system dependencies (important for face recognition, dlib & OpenCV)
RUN apt-get update && \
    apt-get install -y build-essential cmake \
    libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy entire project
COPY . /app/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run entrypoint
ENTRYPOINT ["/entrypoint.sh"]
