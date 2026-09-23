FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/qdrant

EXPOSE 6333 6334
ENV QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
VOLUME ["/qdrant/storage"]

ENTRYPOINT ["/usr/local/bin/qdrant"]