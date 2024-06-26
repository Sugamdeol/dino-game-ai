# Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    chromium-browser \
    chromium-chromedriver \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    xvfb

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install selenium opencv-python-headless stable-baselines3 gym

# Download ChromeDriver
RUN wget -O /usr/local/bin/chromedriver https://chromedriver.storage.googleapis.com/`curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip && \
    chmod +x /usr/local/bin/chromedriver

# Make ChromeDriver executable
RUN chmod +x /usr/local/bin/chromedriver

# Set environment variables
ENV DISPLAY=:99
