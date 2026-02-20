## ------------------------------- Builder Stage ------------------------------ ##
FROM ubuntu:22.04 as builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest installer, install it and then remove it
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

# Set up the UV environment path correctly
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR=/app/python

WORKDIR /app

COPY ./.python-version .
COPY ./pyproject.toml .

RUN uv sync

## ------------------------------- Production Stage ------------------------------ ##
FROM ubuntu:22.04 AS production

RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/python /app/python

RUN mkdir -p /app/SAMPLES_LOW_POWER

COPY 2G_MODEL/best_int8_openvino_model /app/2G_MODEL/best_int8_openvino_model
COPY 3G_4G_MODEL/best_openvino_model /app/3G_4G_MODEL/best_openvino_model
COPY ./*.py /app

# Set up environment variables for production
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 4444

STOPSIGNAL SIGINT

CMD ["python3", "scanner.py"]
