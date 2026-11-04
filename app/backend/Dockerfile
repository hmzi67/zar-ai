# Build from the repo root: docker build -f app/backend/Dockerfile -t zarai-api .
FROM python:3.11-slim

WORKDIR /srv/zarai
COPY pyproject.toml README.md ./
COPY requirements ./requirements
COPY src ./src
COPY configs ./configs
COPY app/backend ./app/backend

# TODO: add the vision and nlp extras (and a GPU base image) once models are served.
RUN pip install --no-cache-dir -e ".[app]"

USER nobody
EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
