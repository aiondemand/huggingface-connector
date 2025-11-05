import datetime as dt
import time
from datetime import datetime
import os

import dateutil.parser
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
import logging
from enum import StrEnum, auto

import aiod
from aiod.authentication import set_token, Token
from dotenv import load_dotenv
import requests


class Modes(StrEnum):
    ALL = auto()
    SINCE = auto()
    ID = auto()


logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 10
MAX_TEXT = 65535
MAX_DESCRIPTION_LENGTH = 1800
PLATFORM_NAME = "openml"
STOP_ON_UNEXPECTED_ERROR = False
PER_DATASET_DELAY = None


class ParsingError(Exception):
    pass


class ServerError(Exception):
    pass


def list_datasets(from_: int | None = None):

    def paginate_all_datasets(items_per_page: int = 50):
        url_data = f"https://www.openml.org/api/v1/json/data/list/limit/{items_per_page}/offset/{{offset}}"
        for offset in range(0, 1_000_000, items_per_page):
            response = requests.get(url_data.format(offset=offset), timeout=REQUEST_TIMEOUT)
            if not response.ok:
                status_code = response.status_code
                msg = response.json()["error"]["message"]
                err_msg = f"Error while fetching {url_data} from OpenML: ({status_code}) {msg}"
                raise ServerError(err_msg)
            logger.debug(f"Paging through datasets (offset {offset})")

            try:
                dataset_summaries = response.json()["data"]["dataset"]
                if dataset_summaries:
                    yield from dataset_summaries
                    logger.debug(f"Paged through datasets (total {len(dataset_summaries)})")
                else:
                    break
            except Exception:
                raise ParsingError(f"Could not parse response ({response.status_code}): {response.content}", exc_info=True)

    from_ = from_ or 0
    for dataset in paginate_all_datasets():
        try:
            identifier = dataset["did"]
            if identifier < from_:
                continue
        except KeyError:
            logger.error(f"Received invalid summary: {dataset}", exc_info=True)
            continue

        try:
            yield fetch_openml_dataset(identifier, dataset["quality"])
        except Exception as e:
            logger.error(f"Exception when processing dataset {identifier}")
            logger.exception(e)


def fetch_openml_dataset(identifier_: int, qualities: dict | None = None):
    if not qualities:
        qualities_url = f"https://www.openml.org/api/v1/json/data/qualities/{identifier_}"
        qualities_response = requests.get(
            qualities_url,
            timeout=REQUEST_TIMEOUT,
        )
        if not qualities_response.ok:
            status_code = qualities_response.status_code
            msg = qualities_response.json()["error"]["message"]
            err_msg = f"Error while fetching {qualities_url} from OpenML: ({status_code}) {msg}"
            raise ServerError(err_msg)
        try:
            qualities = qualities_response.json()["data_qualities"]["quality"]
        except:
            raise ParsingError(f"Error parsing JSON of qualities of dataset {qualities_response.content}")

    url_data = f"https://www.openml.org/api/v1/json/data/{identifier_}"
    response = requests.get(url_data, timeout=REQUEST_TIMEOUT)

    if not response.ok:
        status_code = response.status_code
        msg = response.json()["error"]["message"]
        err_msg = f"Error while fetching {url_data} from OpenML: ({status_code}) {msg}"
        raise ServerError(err_msg)

    try:
        dataset_json = response.json()["data_set_description"]
        qualities_json = {quality["name"]: quality["value"] for quality in qualities}
        return dataset_json | {"qualities": qualities_json}
    except Exception:
        raise ParsingError(f"Error parsing JSON of dataset {response.content}")


def _convert_dataset_to_aiod(dataset: dict) -> dict:
    identifier = dataset["id"]

    description = dataset["description"]
    if isinstance(description, list) and len(description) == 0:
        description = ""
    if not isinstance(description, str):
        logger.warning(f"Ignoring description {description} of dataset {identifier}.")
        description = ""
    if len(description) > MAX_DESCRIPTION_LENGTH:
        text_break = " [...]"
        description = description[: MAX_DESCRIPTION_LENGTH - len(text_break)] + text_break

    size = None
    if n_rows := dataset["qualities"].get("NumberOfInstances"):
        size = {
            "unit": "instances",
            "value": int(float(n_rows)),  # OpenML adds the decimal: xxx.0
        }

    return dict(
        platform=PLATFORM_NAME,
        platform_resource_identifier=identifier,
        name=dataset["name"],
        version=dataset["version"],
        same_as=f"https://www.openml.org/api/v1/json/data/{identifier}",
        description=dict(plain=description),
        date_published=dateutil.parser.parse(dataset["upload_date"]).isoformat(),
        license=dataset.get("licence"),
        distribution=[dict(content_url=dataset["url"], encoding_format=dataset["format"])],
        is_accessible_for_free=True,
        keyword=dataset.get("tag", []),
        size=size,
    )


def upsert_dataset(dataset: dict) -> int:
    identifier = dataset["id"]
    try:
        local_dataset = _convert_dataset_to_aiod(dataset)

        try:
            aiod_dataset = aiod.datasets.get_asset_from_platform(
                platform=PLATFORM_NAME,
                platform_identifier=identifier,
                data_format="json",
            )
        except KeyError:
            response = aiod.datasets.register(metadata=local_dataset)
            if isinstance(response, str):
                logger.debug(f"Indexed dataset {identifier}: {response}")
                return HTTPStatus.OK
            elif isinstance(response, requests.Response):
                logger.warning(f"Error uploading dataset ({response.status_code}: {response.content}")
                breakpoint()
                return response.status_code
            raise

        if "identifier" not in aiod_dataset:
            raise RuntimeError(
                f"Unexpected server response retrieving OpenML dataset {identifier} "
                f"from AI-on-Demand: {aiod_dataset}"
            )

        response = aiod.datasets.replace(identifier=aiod_dataset['identifier'], metadata=local_dataset)
        if response.status_code == HTTPStatus.OK:
            logger.debug(f"Updated dataset {identifier}: {aiod_dataset['identifier']}")
        else:
            logger.warning(f"Could not update {aiod_dataset['identifier']} for openml dataset {identifier} "
                           f"({response.status_code}): {response.content}")
    except Exception as e:
        logger.exception(
            msg=f"Exception encountered when upserting dataset {identifier}.",
            exc_info=e,
        )
        if STOP_ON_UNEXPECTED_ERROR:
            raise
        return -1
    return response.status_code


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
            "For mode 'ID' this must be an openml identifier. "
            "For mode 'SINCE' this must be an openml identifier, this dataset and "
            "all datasets with higher identifier will be indexed."
            "Cannot be set with mode 'ALL'."
        )
    )
    log_levels = [level.lower() for level in logging.getLevelNamesMapping()]
    parser.add_argument(
        "--app-log-level",
        choices=log_levels,
        default='info',
        help="Emit all log messages generated of at least this level by the app."
    )
    parser.add_argument(
        '--root-log-level',
        choices=log_levels,
        default='error',
        help="Emit all log messages generated of at least this level by the app's dependencies."
    )
    args = parser.parse_args()
    if args.mode == Modes.ALL and args.value:
        logger.error("Cannot run mode 'all' when a value is supplied.")
        quit(code=1)

    return args


def configure_connector():
    global BATCH_SIZE, PLATFORM_NAME, STOP_ON_UNEXPECTED_ERROR, PER_DATASET_DELAY

    dot_file = Path("~/.aiod/openml/.env").expanduser()
    if dot_file.exists() and load_dotenv(dot_file):
        logger.info(f"Loaded variables from {dot_file}")
    else:
        reason = "unknown reason" if dot_file else "file does not exist"
        logger.info(f"No environment variables loaded from {dot_file}: {reason}.")

    BATCH_SIZE = os.getenv("AIOD_BATCH_SIZE", 25)
    PLATFORM_NAME = os.getenv("PLATFORM_NAME", PLATFORM_NAME)
    PER_DATASET_DELAY = float(delay) if (delay := os.getenv("PER_DATASET_DELAY")) else None
    STOP_ON_UNEXPECTED_ERROR = os.getenv("STOP_ON_UNEXPECTED_ERROR", STOP_ON_UNEXPECTED_ERROR)

    token = os.getenv("CLIENT_SECRET")
    assert token, "CLIENT_SECRET environment variable not set"

    masked_token = '*' * (len(token) + 4) + token[-4:]
    logger.info(f"{'aiondemand version:':25} {aiod.version}")
    logger.info(f"{'STOP_ON_UNEXPECTED_ERROR:':25} {STOP_ON_UNEXPECTED_ERROR}")
    logger.info(f"{'PER_DATASET_DELAY:':25} {PER_DATASET_DELAY}")
    logger.info(f"{'AI-on-Demand API server:':25} {aiod.config.api_server}")
    logger.info(f"{'Platform Name:':25} {PLATFORM_NAME}")
    logger.info(f"{'Authentication server:':25} {aiod.config.auth_server}")
    logger.info(f"{'Client ID:':25} {aiod.config.client_id}")
    logger.info(f"{'Using secret:':25} {masked_token}")

    set_token(Token(client_secret=token))
    user = aiod.get_current_user()

    required_role = f"platform_{PLATFORM_NAME}"
    wrong_platform_msg = (
        f"Client roles {user.roles} do not include required {required_role!r} role."
        "Please make sure the `PLATFORM_NAME` environment variable is configured correctly, "
        "or contact your Keycloak administrator."
    )
    assert required_role in user.roles, wrong_platform_msg

    logger.info("Successfully authenticated and connected to AI-on-Demand.")


def main():
    args = parse_args()
    logging.basicConfig(level=args.root_log_level.upper())
    logger.setLevel(args.app_log_level.upper())
    configure_connector()

    match (args.mode, args.value):
        case Modes.ID, id_:
            if not id_.isdigit():
                logger.error(f"Identifier specified should be an integer, is {id_!r}")
                quit(1)
            dataset = fetch_openml_dataset(int(id_))
            upsert_dataset(dataset)
        case Modes.SINCE, id_:
            if not id_.isdigit():
                logger.error(f"Identifier specified should be an integer, is {id_!r}")
                quit(1)
            for dataset in list_datasets(from_=int(id_)):
                upsert_dataset(dataset)
                if PER_DATASET_DELAY:
                    time.sleep(PER_DATASET_DELAY)
        case Modes.ALL, None:
            for dataset in list_datasets():
                upsert_dataset(dataset)
                if PER_DATASET_DELAY:
                    time.sleep(PER_DATASET_DELAY)
        case _:
            raise NotImplemented(f"Unexpected arguments: {args}")


if __name__ == '__main__':
    main()
