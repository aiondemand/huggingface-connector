FROM python:3.13.7-trixie

RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser
RUN python -m venv .venv
RUN .venv/bin/python -m pip install uv

COPY pyproject.toml /app/pyproject.toml
RUN .venv/bin/uv pip install -r /app/pyproject.toml

COPY . /app
RUN .venv/bin/uv pip install file:///app

ENTRYPOINT [".venv/bin/python", "/app/src/connector.py"]
CMD ["--mode", "all"]
