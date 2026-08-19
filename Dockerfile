# Sandbox for training on untrusted, AI-written reward code
FROM python:3.9-slim

# Build tools needed to compile the Box2D physics engine
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-sandbox.txt .
RUN pip install --no-cache-dir -r requirements-sandbox.txt

# Only the code training needs. No .env, no API keys, no agent code.
COPY train_with_reward.py reward_wrapper.py validator.py ./

# Run as an unprivileged user, not root
RUN useradd --create-home sandbox
USER sandbox
ENV HOME=/tmp
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
