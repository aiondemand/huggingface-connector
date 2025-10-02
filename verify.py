""" Verifies data uploaded by this connector is similar to the data already on production.

It's not super effective but should provide confidence.
Expecting exact equality is kind of hard since datasets on Hugging Face may have been updated
whereas the indexed data on AI-on-Demand is not. This means that a recently synchronized local
database likely has different, more up-to-date, metadata.

The difference for the specific default repositories here confirm this. Awesome-gpt-prompts has
had its keywords and distributions updated.
"""
from http import HTTPStatus
from typing import Iterable

from huggingface_hub import dataset_info
import requests

repositories = [
    "fka/awesome-chatgpt-prompts",
    "facebook/seamless-interaction",
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    # "wikimedia/wikipedia",  # too many distributions
    # "nvidia/Nemotron-Personas-Japan",  # these and below are not on prod
    # "t-tech/T-ECD",
    # "zai-org/CC-Bench-trajectories",
    # "smolagents/aguvis-stage-2",
]

LOCAL_API = "http://localhost/"
PRODUCTION_API = "https://api.aiod.eu/"


def diff(local: dict, prod: dict, ignore: Iterable | None = None):
    if only_local := set(local) - set(prod):
        print("Local-only attributes:")
    for key in only_local:
        print(f"{key:10} {local[key]}")

    if only_prod := set(prod) - set(local):
        print("Prod-only attributes:")
    for key in only_prod:
        print(f"{key:10} {prod[key]}")

    if ignore is None:
        ignore = ["identifier", "platform", "ai_resource_identifier", "ai_asset_identifier", "aiod_entry"]
    for key in set(local) & set(prod):
        if key in ignore:
            continue
        local_value, prod_value = local[key], prod[key]
        if local_value != prod_value:
            try:
                if isinstance(local_value, list) and isinstance(prod_value, list):
                    assert set(local_value) == set(prod_value)
                    continue
            except:
                pass
            print(f"{key} in both instances, but differs:")
            print(f"local  {local_value}")
            print(f"prod   {prod_value}")



for repo in repositories:
    dataset = dataset_info(repo)
    local = requests.get(f"{LOCAL_API}platforms/example/datasets/{dataset._id}")
    assert local.status_code == HTTPStatus.OK, (local.reason, local.content, repo)
    prod = requests.get(f"{PRODUCTION_API}platforms/huggingface/datasets/{dataset._id}")
    assert prod.status_code == HTTPStatus.OK, (prod.reason, prod.content, repo)
    print(f"Comparing {repo}")
    diff(local.json(), prod.json())


