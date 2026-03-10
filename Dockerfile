# Base Image (lightweight Python)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies required for NeMo + audio
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Run setup script
RUN chmod +x setup.sh && bash setup.sh

# Expose port for future UI (if added)
EXPOSE 7860

# Default command
CMD ["/bin/bash"]