FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/rostanda/SimplePH"
LABEL org.opencontainers.image.description="Containerized C++/Python SPH simulation framework"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-dev.txt .
RUN pip install --upgrade pip && pip install -r requirements-dev.txt

COPY . .

RUN cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python3)
RUN cmake --build build --target SimplePH -- -j$(nproc)

ENV PYTHONPATH=/app/build:/app/python

CMD ["python"]