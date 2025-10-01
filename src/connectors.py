from datetime import datetime
import math
import os
from argparse import ArgumentParser
from pathlib import Path
import logging
from enum import StrEnum, auto

import aiod
from aiod.authentication import set_token, Token
from huggingface_hub import list_datasets, DatasetInfo, dataset_info
import requests


class Modes(StrEnum):
    ALL = auto()
    SINCE = auto()
    ID = auto()
    # DELETE = auto()


logger = logging.getLogger(__name__)
HUGGING_FACE_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
REQUEST_TIMEOUT = 10
MAX_TEXT = 65535


def _convert_dataset_to_aiod(dataset: DatasetInfo) -> dict:
    distributions = _get_distribution_data(dataset)

    ds_license = None
    if (card := dataset.card_data) and (license_ := card.get("license")):
        if isinstance(license_, str):
            ds_license = license_
        elif isinstance(license_, list):
            ds_license = license_[0]
            logger.warning(f"Multiple licenses for dataset {dataset.id}: {license_}")
        else:
            logger.warning(f"Cannot parse license data for {dataset.id}: {license_!r}")

    # add citations...

    description = getattr(dataset, "description", None)
    if description and len(description) > MAX_TEXT:
        text_break = " [...]"
        description = description[: MAX_TEXT - len(text_break)] + text_break

    # Related Resources?

    return dict(
        platform="huggingface",
        platform_resource_identifier=dataset._id,  # See REST API #385, #392
        name=dataset.id,
        same_as=f"https://huggingface.co/datasets/{dataset.id}",
        description=dict(plain=description),
        date_published=dataset.created_at if hasattr(dataset, "created_at") else None,
        license=ds_license,
        distribution=distributions,
        is_accessible_for_free=not dataset.private,
        keyword=dataset.tags,
    )


def _get_distribution_data(dataset: DatasetInfo) -> list[dict]:
    response = requests.get(
        HUGGING_FACE_PARQUET_URL,
        params={"dataset": dataset.id},
        timeout=REQUEST_TIMEOUT
    )
    if not response.ok:
        msg = response.json()["error"]
        logging.warning(f"Unable to retrieve parquet info for dataset '{dataset.id}': '{msg}'")
        return []
    return [
        dict(
            name=pq_file["filename"],
            description=f"{pq_file['dataset']}. Config: {pq_file['config']}. Split: "
                        f"{pq_file['split']}",
            content_url=pq_file["url"],
            content_size_kb=math.ceil(pq_file["size"] / 1000),
        )
        for pq_file in response.json()["parquet_files"]
    ]


def delete_removed_assets() -> None:
    """Removes assets from the AI-on-Demand catalogue if they are removed from Hugging Face."""

    def aiod_datasets(batch_size: int = BATCH_SIZE):
        offset = 0
        datasets = aiod.datasets.get_list(offset=offset, limit=batch_size, platform="huggingface")
        while datasets:
            yield from datasets
            offset += BATCH_SIZE
            datasets = aiod.datasets.get_list(offset=offset, limit=batch_size, platform="huggingface")

    for dataset in aiod_datasets():
        if not dataset_info(dataset.name):
            logger.info(
                f"Dataset {dataset.name} ({dataset.platform_resource_identifier}) "
                f"no longer on Hugging Face. Removing asset from AI-on-Demand"
            )
            aiod.datasets.delete(dataset['identifier'])


def upsert_dataset(dataset: DatasetInfo):
    local_dataset = _convert_dataset_to_aiod(dataset)
    try:
        aiod_dataset = aiod.datasets.get_asset_from_platform(platform="huggingface", platform_identifier=dataset._id)
        aiod.datasets.replace(aiod_dataset['identifier'], metadata=local_dataset)
    except KeyError:
        aiod.datasets.register(metadata=local_dataset)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "mode",
        choices=list(Modes),
    )
    parser.add_argument(
        "value",
        default=None,
        required=False,
        type=str,
        help=(
            "For mode 'ID' this must be a Hugging Face identifier. "
            "For mode 'SINCE' this must be a valid timestamp in ISO-8601 format."
            "Cannot be set with mode 'ALL'."
        )
    )
    args = parser.parse_args()
    if args.mode == Modes.ALL and args.value:
        logger.error("Cannot run mode 'all' when a value is supplied.")
        quit(code=1)

    return args


def configure_connector():
    global BATCH_SIZE

    load_dotenv()

    BATCH_SIZE = os.getenv("AIOD_HF_BATCH_SIZE", 25)
    HF_API_KEY = os.getenv("AIOD_HF_API_KEY", None)
    WAIT_TIME = os.getenv("AIOD_HF_WAIT_TIME", 0.1)

    token = Token.from_file(Path('secret.toml'))
    set_token(token)


def main():
    configure_connector()
    args = parse_args()
    match (args.mode, args.value):
        case Modes.ID, id_:
            dataset = dataset_info(id_)
            upsert_dataset(dataset)
        case Modes.SINCE, timestamp:
            parsed_time = datetime.fromisoformat(timestamp)
            for dataset in list_datasets(full=True, filter=...):
                upsert_dataset(dataset)
        case Modes.ALL, None:
            for dataset in list_datasets(full=True):
                upsert_dataset(dataset)
        case _:
            raise NotImplemented(f"Unexpected arguments: {args}")


if __name__ == '__main__':
    main()
