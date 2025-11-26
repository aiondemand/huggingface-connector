FROM python:3.13.7-trixie

# Install CA infrastructure
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Add AIoD cert and build custom bundle
RUN mkdir -p /certs
COPY certs/aiod-insight-centre.crt /certs/aiod-insight-centre.crt
RUN cat /etc/ssl/certs/ca-certificates.crt /certs/aiod-insight-centre.crt > /certs/custom-ca-bundle.crt

# Make Python requests use the custom CA bundle
ENV REQUESTS_CA_BUNDLE=/certs/custom-ca-bundle.crt

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
