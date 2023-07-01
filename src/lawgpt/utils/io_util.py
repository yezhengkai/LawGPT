from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from _typeshed import FileDescriptorOrPath


def load_json(json_path: FileDescriptorOrPath) -> Any:
    with open(json_path, "rb") as f:
        json_data = orjson.loads(f.read())
    return json_data


def dump_json(
    json_path: FileDescriptorOrPath, obj: Any, option: int | None = orjson.OPT_INDENT_2
) -> None:
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(obj, option=option))
