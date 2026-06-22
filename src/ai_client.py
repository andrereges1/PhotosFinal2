"""Cliente configuravel para a IA de decisao por texto."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from typing import Any

import requests

from src.config import (
    AI_ALLOWED_DECISIONS,
    AI_DEFAULT_BASE_URL,
    AI_DEFAULT_ENDPOINT_PATH,
    AI_DEFAULT_MODEL,
    AI_MAX_RETRIES,
    AI_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AIClientConfig:
    api_key: str | None
    base_url: str
    model: str
    endpoint_path: str
    timeout_seconds: int
    max_retries: int

    @property
    def full_url(self) -> str:
        endpoint = self.endpoint_path if self.endpoint_path.startswith("/") else f"/{self.endpoint_path}"
        return self.base_url.rstrip("/") + endpoint


@dataclass(frozen=True, slots=True)
class AIClientResponse:
    response_text: str | None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        logger.info("python-dotenv nao disponivel; usando variaveis de ambiente do sistema")
        return
    load_dotenv()


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def load_ai_config() -> AIClientConfig:
    _load_dotenv_if_available()
    api_key = _env_first("AI_DECISION_API_KEY", "OPENCODE_API_KEY")
    base_url = _env_first("AI_DECISION_BASE_URL", "OPENCODE_BASE_URL", default=AI_DEFAULT_BASE_URL)
    model = _env_first("AI_DECISION_MODEL", "OPENCODE_MODEL", default=AI_DEFAULT_MODEL)
    endpoint_path = _env_first("AI_DECISION_ENDPOINT_PATH", default=AI_DEFAULT_ENDPOINT_PATH)

    return AIClientConfig(
        api_key=api_key,
        base_url=base_url or AI_DEFAULT_BASE_URL,
        model=model or AI_DEFAULT_MODEL,
        endpoint_path=endpoint_path or AI_DEFAULT_ENDPOINT_PATH,
        timeout_seconds=AI_REQUEST_TIMEOUT_SECONDS,
        max_retries=AI_MAX_RETRIES,
    )


def _extract_content_from_openai_response(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    content = data.get("content")
    if isinstance(content, str):
        return content
    return None


def call_ai_text_model(
    prompt: str,
    system_content: str,
    config: AIClientConfig | None = None,
) -> AIClientResponse:
    config = config or load_ai_config()
    if not config.api_key:
        return AIClientResponse(None, error="API key nao configurada")

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    logger.info("IA ativada: modelo=%s base_url=%s", config.model, config.base_url)
    attempts = max(1, config.max_retries + 1)
    last_error = "Falha desconhecida"
    for _ in range(attempts):
        try:
            response = requests.post(
                config.full_url,
                headers=headers,
                json=payload,
                timeout=config.timeout_seconds,
            )
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}"
                continue
            data = response.json()
            content = _extract_content_from_openai_response(data)
            if not content:
                return AIClientResponse(None, raw_response=data, error="Resposta sem conteudo JSON")
            return AIClientResponse(content, raw_response=data)
        except requests.Timeout:
            last_error = "timeout"
        except requests.RequestException as exc:
            last_error = exc.__class__.__name__
        except ValueError:
            last_error = "resposta HTTP nao era JSON"

    logger.info("IA nao respondeu de forma valida: %s", last_error)
    return AIClientResponse(None, error=last_error)


def call_ai_decision_model(prompt: str, config: AIClientConfig | None = None) -> AIClientResponse:
    return call_ai_text_model(
        prompt,
        "Voce decide estrategias de corte de fotos com base em relatorio tecnico. Responda apenas JSON.",
        config,
    )


def call_ai_batch_planner(prompt: str, config: AIClientConfig | None = None) -> AIClientResponse:
    return call_ai_text_model(
        prompt,
        "Voce planeja lotes de fotos 10x15 usando apenas relatorios tecnicos. Responda apenas JSON.",
        config,
    )


def extract_json_from_ai_response(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _has_valid_optional_bool(data: dict[str, Any], field_name: str) -> bool:
    return field_name not in data or isinstance(data[field_name], bool)


def parse_ai_json_response(response_text: str) -> dict[str, Any] | None:
    data = extract_json_from_ai_response(response_text)
    if not data:
        return None
    required_fields = {"decision", "confidence", "reason", "risk_level"}
    if not required_fields.issubset(data):
        return None

    decision = data.get("decision")
    if not isinstance(decision, str) or decision.strip() not in AI_ALLOWED_DECISIONS:
        return None

    confidence = data.get("confidence")
    if not _is_finite_number(confidence):
        return None
    confidence_value = float(confidence)
    if confidence_value < 0.0 or confidence_value > 1.0:
        return None

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None

    risk_level = data.get("risk_level")
    if not isinstance(risk_level, str) or risk_level.strip() not in {"low", "medium", "high"}:
        return None

    bool_fields = {
        "use_borders",
        "allow_crop",
        "protect_faces",
        "protect_people",
        "protect_text",
        "rotate_on_pdf",
        "create_extra_page",
    }
    if any(not _has_valid_optional_bool(data, field_name) for field_name in bool_fields):
        return None

    max_crop_percent = data.get("max_crop_percent")
    if max_crop_percent is not None:
        if not _is_finite_number(max_crop_percent):
            return None
        max_crop_value = float(max_crop_percent)
        if max_crop_value < 0.0 or max_crop_value > 100.0:
            return None

    warnings = data.get("warnings")
    if warnings is not None and not isinstance(warnings, list):
        return None

    return data


def test_ai_connection() -> bool:
    result = call_ai_decision_model('{"ping": true}')
    return result.response_text is not None
