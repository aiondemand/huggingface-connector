import datetime as dt
import time
from datetime import datetime
from json import JSONDecodeError

import huggingface_hub
import math
import os
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
import logging
from enum import StrEnum, auto

import aiod
from aiod.authentication import set_token, Token
from huggingface_hub import list_datasets, DatasetInfo, dataset_info
from dotenv import load_dotenv
import requests
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError


class Modes(StrEnum):
    ALL = auto()
    SINCE = auto()
    ID = auto()
    # DELETE = auto()


logger = logging.getLogger(__name__)
HUGGING_FACE_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
REQUEST_TIMEOUT = 10
MAX_TEXT = 65535
PLATFORM_NAME = "huggingface"
HUGGING_FACE_API_KEY = None
STOP_ON_UNEXPECTED_ERROR = False
PER_DATASET_DELAY = None
BATCH_SIZE = 25


def _convert_dataset_to_aiod(dataset: DatasetInfo) -> dict:
    distributions = _get_distribution_data(dataset)

    ds_license = None
    if (card := dataset.card_data) and (license_ := card.get("license")):
        if isinstance(license_, str):
            ds_license = license_
        elif isinstance(license_, list):
            ds_license = license_[0]
            if len(license_) > 1:
                logger.warning(f"Multiple licenses for dataset {dataset.id}: {license_}")
        else:
            logger.warning(f"Cannot parse license data for {dataset.id}: {license_!r}")

    description = getattr(dataset, "description", None)
    if description and len(description) > MAX_TEXT:
        text_break = " [...]"
        description = description[: MAX_TEXT - len(text_break)] + text_break

    return dict(
        platform=PLATFORM_NAME,
        platform_resource_identifier=dataset._id,  # See REST API #385, #392
        name=dataset.id,
        same_as=f"https://huggingface.co/datasets/{dataset.id}",
        description=dict(plain=description),
        date_published=dataset.created_at.isoformat() if hasattr(dataset, "created_at") else None,
        license=ds_license,
        distribution=distributions,
        is_accessible_for_free=not dataset.private,
        keyword=dataset.tags,
    )


def _get_distribution_data(dataset: DatasetInfo) -> list[dict]:
    response = requests.get(
        HUGGING_FACE_PARQUET_URL,
        params={"dataset": dataset.id},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        try:
            msg = response.json()["error"]
        except JSONDecodeError:
            msg = response.content
        except KeyError:
            msg = response.json()
        logger.warning(f"Unable to retrieve parquet info for dataset '{dataset.id}': '{msg}'")
        return []
    return [
        dict(
            name=pq_file["filename"],
            description=(
                f"{pq_file['dataset']}. Config: {pq_file['config']}. "
                f"Split: {pq_file['split']}"
            ),
            content_url=pq_file["url"],
            content_size_kb=math.ceil(pq_file["size"] / 1000),
        )
        for pq_file in response.json()["parquet_files"]
    ]


def delete_removed_assets() -> None:
    """Removes assets from the AI-on-Demand catalogue if they are removed from Hugging Face."""

    def aiod_datasets(batch_size: int | None = None):
        if batch_size is None:
            batch_size = BATCH_SIZE

        offset = 0
        datasets = aiod.datasets.get_list(offset=offset, limit=batch_size, platform=PLATFORM_NAME)
        while datasets:
            yield from datasets
            offset += batch_size
            datasets = aiod.datasets.get_list(offset=offset, limit=batch_size, platform=PLATFORM_NAME)

    for dataset in aiod_datasets():
        if not dataset_info(dataset["name"]):
            logger.info(
                f"Dataset {dataset['name']} ({dataset['platform_resource_identifier']}) "
                f"no longer on Hugging Face. Removing asset from AI-on-Demand"
            )
            aiod.datasets.delete(dataset["identifier"])


def upsert_dataset(dataset: DatasetInfo) -> int:
    try:
        local_dataset = _convert_dataset_to_aiod(dataset)

        # Try to fetch existing asset from AIoD
        try:
            aiod_dataset = aiod.datasets.get_asset_from_platform(
                platform=PLATFORM_NAME,
                platform_identifier=dataset._id,
                data_format="json",
            )
        except KeyError:
            # Normal "not found" path -> register a new asset
            response = aiod.datasets.register(metadata=local_dataset)
            if isinstance(response, str):
                logger.debug(f"Indexed dataset {dataset.id}({dataset._id}): {response}")
                return HTTPStatus.OK
            elif isinstance(response, requests.Response):
                logger.warning(
                    f"Error uploading dataset {dataset.id} ({dataset._id}) "
                    f"({response.status_code}: {response.content})"
                )
                return response.status_code
            raise RuntimeError(
                f"Unexpected response when registering Hugging Face dataset "
                f"{dataset.id} ({dataset._id}): {response!r}"
            )
        except RequestsJSONDecodeError as e:
            # AIOD responded with non-JSON when checking for an existing asset.
            # Treat this as a transient / "not found" case and TRY a register once.
            logger.warning(
                "Non-JSON response from AI-on-Demand when checking existing asset for "
                "%s (%s); treating as missing and attempting to register. Error: %s",
                dataset.id,
                dataset._id,
                e,
            )
            response = aiod.datasets.register(metadata=local_dataset)
            if isinstance(response, str):
                logger.debug(f"Indexed dataset {dataset.id}({dataset._id}): {response}")
                return HTTPStatus.OK
            elif isinstance(response, requests.Response):
                logger.warning(
                    f"Error uploading dataset {dataset.id} ({dataset._id}) "
                    f"after JSON decode failure "
                    f"({response.status_code}: {response.content})"
                )
                return response.status_code
            raise RuntimeError(
                f"Unexpected response when registering after JSON decode failure for "
                f"{dataset.id} ({dataset._id}): {response!r}"
            )

        # If we got here, we have some JSON-decoded object from AIOD
        if "identifier" not in aiod_dataset:
            raise RuntimeError(
                f"Unexpected server response retrieving Hugging Face dataset "
                f"{dataset.id} ({dataset._id}) from AI-on-Demand: {aiod_dataset!r}"
            )

        # Update existing asset
        response = aiod.datasets.replace(
            identifier=aiod_dataset["identifier"],
            metadata=local_dataset,
        )
        if response.status_code == HTTPStatus.OK:
            logger.debug(
                "Updated dataset %s(%s): %s",
                dataset.id,
                dataset._id,
                aiod_dataset["identifier"],
            )
        else:
            logger.warning(
                "Could not update %s for repository %s (%s): %s",
                aiod_dataset["identifier"],
                dataset.id,
                response.status_code,
                response.content,
            )
        return response.status_code

    except Exception as e:
        logger.exception(
            msg=(
                f"Exception encountered when upserting dataset "
                f"{dataset.id} ({dataset._id})."
            ),
            exc_info=e,
        )
        if STOP_ON_UNEXPECTED_ERROR:
            raise
        return -1


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=list(Modes),
    )
    parser.add_argument(
        "--value",
        default=None,
        required=False,
        type=str,
        help=(
            "For mode 'ID' this must be a Hugging Face identifier. "
            "For mode 'SINCE' this must be a valid timestamp in ISO-8601 format,"
            "assuming an UTC timezone. Cannot be set with mode 'ALL'."
        ),
    )
    log_levels = [level.lower() for level in logging.getLevelNamesMapping()]
    parser.add_argument(
        "--app-log-level",
        choices=log_levels,
        default="info",
        help="Emit all log messages generated of at least this level by the app.",
    )
    parser.add_argument(
        "--root-log-level",
        choices=log_levels,
        default="error",
        help="Emit all log messages generated of at least this level by the app's dependencies.",
    )
    args = parser.parse_args()
    if args.mode == Modes.ALL and args.value:
        logger.error("Cannot run mode 'all' when a value is supplied.")
        quit(code=1)

    return args


def configure_connector():
    global BATCH_SIZE, PLATFORM_NAME, HUGGING_FACE_API_KEY, STOP_ON_UNEXPECTED_ERROR, PER_DATASET_DELAY

    dot_file = Path("~/.aiod/huggingface/.env").expanduser()
    if dot_file.exists() and load_dotenv(dot_file):
        logger.info(f"Loaded variables from {dot_file}")
    else:
        reason = "unknown reason" if dot_file else "file does not exist"
        logger.info(f"No environment variables loaded from {dot_file}: {reason}.")

    BATCH_SIZE = int(os.getenv("AIOD_BATCH_SIZE", "25"))
    PLATFORM_NAME = os.getenv("PLATFORM_NAME", PLATFORM_NAME)
    HUGGING_FACE_API_KEY = os.getenv("AIOD_HF_API_KEY") or None

    delay = os.getenv("PER_DATASET_DELAY")
    PER_DATASET_DELAY = float(delay) if delay else None

    STOP_ON_UNEXPECTED_ERROR = (
        str(os.getenv("STOP_ON_UNEXPECTED_ERROR", str(STOP_ON_UNEXPECTED_ERROR))).lower()
        == "true"
    )

    token = os.getenv("CLIENT_SECRET")
    assert token, "CLIENT_SECRET environment variable not set"

    masked_token = "*" * max(4, len(token) - 4) + token[-4:]

    logger.info(f"{'aiondemand version:':25} {aiod.version}")
    logger.info(f"{'hugging face hub version:':25} {huggingface_hub.__version__}")
    logger.info(f"{'STOP_ON_UNEXPECTED_ERROR:':25} {STOP_ON_UNEXPECTED_ERROR}")
    logger.info(f"{'PER_DATASET_DELAY:':25} {PER_DATASET_DELAY}")
    logger.info(f"{'AI-on-Demand API server:':25} {aiod.config.api_server}")
    logger.info(f"{'Platform Name:':25} {PLATFORM_NAME}")
    if HUGGING_FACE_API_KEY:
        masked_hf_key = "*" * max(4, len(HUGGING_FACE_API_KEY) - 4) + HUGGING_FACE_API_KEY[-4:]
    else:
        masked_hf_key = "not set"
    logger.info(f"{'Hugging Face API key:':25} {masked_hf_key}")
    logger.info(f"{'Authentication server:':25} {aiod.config.auth_server}")
    logger.info(f"{'Client ID:':25} {aiod.config.client_id}")
    logger.info(f"{'Using secret:':25} {masked_token}")

    set_token(Token(client_secret=token))
    logger.info(
        "Configured AI-on-Demand client token for Hugging Face connector "
        "(skipping authorization_test)."
    )


def main():
    args = parse_args()
    logging.basicConfig(level=args.root_log_level.upper())
    logger.setLevel(args.app_log_level.upper())
    configure_connector()

    match (args.mode, args.value):
        case Modes.ID, id_:
            dataset = dataset_info(id_)
            upsert_dataset(dataset)
        case Modes.SINCE, timestamp:
            parsed_time = datetime.fromisoformat(timestamp).replace(tzinfo=dt.UTC)
            for dataset in list_datasets(
                full=True,
                sort="last_modified",
                direction=-1,
                token=HUGGING_FACE_API_KEY,
            ):
                if dataset.last_modified < parsed_time:
                    break
                upsert_dataset(dataset)
                if PER_DATASET_DELAY:
                    time.sleep(PER_DATASET_DELAY)
        case Modes.ALL, None:
            for dataset in list_datasets(full=True, token=HUGGING_FACE_API_KEY):
                upsert_dataset(dataset)
                if PER_DATASET_DELAY:
                    time.sleep(PER_DATASET_DELAY)
        case _:
            raise NotImplemented(f"Unexpected arguments: {args}")


if __name__ == "__main__":
    main()