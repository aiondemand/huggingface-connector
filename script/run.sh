#!/bin/bash

WHICH="${1:-prod}"
shift

ABSOLUTE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABSOLUTE_SCRIPT_DIR}/.${WHICH}.env"
CONFIG_FILE="${ABSOLUTE_SCRIPT_DIR}/config.${WHICH}.toml"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: expected env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: expected config file not found: $CONFIG_FILE" >&2
  exit 1
fi

docker run \
  -v "${ENV_FILE}":/home/appuser/.aiod/huggingface/.env \
  -v "${CONFIG_FILE}":/home/appuser/.aiod/config.toml \
  --network=host \
  --name aiod-huggingface-connector \
  -it aiondemand/huggingface-connector "$@"
