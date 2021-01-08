FROM python:3.8.6-slim

RUN apt update \
    && apt install -y --no-install-recommends \
    apt-utils \
    git \
    gcc \
    curl \
    build-essential \
    imagemagick \
    ffmpeg \
    && pip install --no-cache-dir \
    autopep8 \
    flake8 \
    pytest \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt install -y --no-install-recommends nodejs \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*
