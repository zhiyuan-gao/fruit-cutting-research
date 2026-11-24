FROM nvcr.io/nvidia/isaac-sim:5.1.0
USER root
COPY requirements.txt /tmp/fruit-requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/* \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/fruit-requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
