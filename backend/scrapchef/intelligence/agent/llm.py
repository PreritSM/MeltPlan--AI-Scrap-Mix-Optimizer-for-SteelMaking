"""LLM adapter: one OpenAI-compatible chat client (default GLM 5.3 via NVIDIA, decision D18). The agent loop and the narrator
depend only on `LLM.chat`, which returns a normalised reply, so a provider swap touches this file only.

Settings (environment or a gitignored `backend/.env`):
  NVIDIA_API_KEY            key for the default endpoint (SCRAPCHEF_LLM_API_KEY overrides it)
  SCRAPCHEF_LLM_BASE_URL    default https://integrate.api.nvidia.com/v1
  SCRAPCHEF_LLM_MODEL       default z-ai/glm-5.3
  SCRAPCHEF_LLM_EXTRA_BODY  optional JSON merged into each request (provider-specific switches)
  SCRAPCHEF_LLM=off         disable the LLM entirely (tests, offline demos)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"   # backend/.env


def _load_env_file(path: Path = ENV_FILE) -> None:
    """Minimal KEY=VALUE loader; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON text from the model; validated by the caller


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""


class LLM:
    def __init__(self):
        _load_env_file()
        self.base_url = os.environ.get("SCRAPCHEF_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.environ.get("SCRAPCHEF_LLM_MODEL", "z-ai/glm-5.3")
        self.api_key = os.environ.get("SCRAPCHEF_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.extra_body = json.loads(os.environ.get("SCRAPCHEF_LLM_EXTRA_BODY", "{}") or "{}")
        self.disabled = os.environ.get("SCRAPCHEF_LLM", "").lower() == "off"
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self.disabled

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}" if self.available else "LLM unavailable (no key or disabled): deterministic fallback"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, timeout: float = 30.0,
             temperature: float = 0.2, max_tokens: int = 4096) -> Reply:
        """One chat completion. Raises openai errors (the caller falls back); never returns partial tool calls silently."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=1)
        kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        if tools:
            kw.update(tools=tools, tool_choice="auto")
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        r = self._client.chat.completions.create(**kw)
        ch = r.choices[0]
        m = ch.message
        extra = m.model_extra or {}
        return Reply(content=m.content or "", reasoning=extra.get("reasoning_content") or extra.get("reasoning") or "",
                     tool_calls=[ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (m.tool_calls or [])],
                     finish_reason=ch.finish_reason or "")
