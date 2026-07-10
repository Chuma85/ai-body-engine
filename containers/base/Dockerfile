FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/body-engine

RUN groupadd --system bodyengine && useradd --system --gid bodyengine --create-home bodyengine
WORKDIR /opt/body-engine
COPY requirements-container.lock /tmp/requirements-container.lock
RUN python -m pip install --requirement /tmp/requirements-container.lock
