from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

from lawgpt.data.constant import METADATA, PROCESSED_JSON_NAME
from lawgpt.data.util import compute_sha256
from lawgpt.utils import dump_json, load_json

if TYPE_CHECKING:
    from _typeshed import StrPath


def download_raw_dataset(metadata: dict, download_data_dir: StrPath) -> None:
    download_data_dir = Path(download_data_dir)
    download_data_dir.mkdir(parents=True, exist_ok=True)

    filepath = download_data_dir.joinpath(metadata["filename"])
    print(f"Downloading raw dataset from {metadata['url']} to {filepath}...")
    urlretrieve(metadata["url"], filepath)
    print("Computing SHA-256...")
    sha256 = compute_sha256(filepath)
    if sha256 != metadata["sha256"]:
        raise ValueError(f"Downloaded {filepath} SHA-256 is incorrect.")


def convert_judicial_yuan_qa(json_data, list_instruction: list | None = None) -> list:
    list_instruction = [] if list_instruction is None else list_instruction
    for qa in json_data:
        list_instruction.append(
            {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
            }
        )
    return list_instruction


def convert_moex(json_data, list_instruction: list | None = None) -> list:
    list_instruction = [] if list_instruction is None else list_instruction
    for exam in json_data:
        for qa in exam["qa"]:
            list_instruction.append(
                {
                    "instruction": qa["question"],
                    "input": qa["choices"],
                    "output": qa["answer"],
                }
            )
    return list_instruction


def process_raw_dataset(download_data_dir: StrPath, processed_dir: StrPath) -> None:
    download_data_dir = Path(download_data_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = processed_dir.joinpath(PROCESSED_JSON_NAME)

    list_instruction = []
    for metadata in METADATA:
        filepath = download_data_dir.joinpath(metadata["filename"])
        if metadata["filename"] == "judicial_yuan_qa.json":
            convert_judicial_yuan_qa(load_json(filepath), list_instruction)
        elif metadata["filename"] == "moex.json":
            convert_moex(load_json(filepath), list_instruction)

    print(f"Save dataset to {out_json_path}...")
    dump_json(out_json_path, list_instruction)
    print(f"{out_json_path} has {len(list_instruction)} instructions")


def download_and_process_dataset(download_data_dir: StrPath, processed_dir: StrPath) -> None:
    download_data_dir = Path(download_data_dir)
    for metadata in METADATA:
        if not (download_data_dir / metadata["filename"]).is_file():
            download_raw_dataset(metadata, download_data_dir)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    process_raw_dataset(download_data_dir, processed_dir)
