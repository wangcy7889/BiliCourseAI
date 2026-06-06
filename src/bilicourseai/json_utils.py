from __future__ import annotations

import json
import re
from typing import Any


def extract_json_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise


def extract_json_object(text: str) -> dict[str, Any]:
    value = extract_json_value(text)
    if isinstance(value, dict):
        return value
    raise json.JSONDecodeError("Expected JSON object", text, 0)


def extract_json_object_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return extract_json_object(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def json_object_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def salvage_json_object(text: str) -> dict[str, Any]:
    raw = json_object_text(text)
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass

    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    raw = escape_invalid_json_string_backslashes(raw)
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        return {}


def escape_invalid_json_string_backslashes(text: str) -> str:
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            output.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\" and in_string:
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char in valid_escapes:
                output.append(char)
                output.append(next_char)
                index += 2
            else:
                output.append("\\\\")
                index += 1
            continue

        output.append(char)
        index += 1
    return "".join(output)


def best_effort_json_object(text: str) -> dict[str, Any]:
    try:
        return extract_json_object_from_text(text)
    except json.JSONDecodeError:
        return salvage_json_object(text)


def best_effort_json_value(text: str) -> Any:
    try:
        return extract_json_value(text)
    except json.JSONDecodeError:
        return best_effort_json_object(text)
