FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install uv

# Copy source code and project files
COPY . .

# Install dependencies using uv
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "src/worker.py"]
