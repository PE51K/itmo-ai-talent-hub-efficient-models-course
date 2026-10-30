FROM ghcr.io/astral-sh/uv:0.10.7-python3.13-trixie-slim

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
# versions come from uv.lock; the PyPI torch wheel bundles CUDA, the host needs only the NVIDIA driver
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY kd.py plots.py ./
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python"]
CMD ["kd.py"]
