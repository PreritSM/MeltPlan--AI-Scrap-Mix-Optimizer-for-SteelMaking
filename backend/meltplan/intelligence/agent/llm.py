"""LLM adapter: one OpenAI-compatible chat client (default GLM 5.3 via NVIDIA, decision D18). The agent loop and the narrator
depend only on `LLM.chat`, which returns a normalised reply, so a provider swap touches this file only.

Settings (environment or a gitignored `backend/.env`):
  NVIDIA_API_KEY            key for the default endpoint (MELTPLAN_LLM_API_KEY overrides it)
  MELTPLAN_LLM_BASE_URL    default https://integrate.api.nvidia.com/v1
  MELTPLAN_LLM_MODEL       default z-ai/glm-5.3
  MELTPLAN_LLM_EXTRA_BODY  optional JSON merged into each request (provider-specific switches)
  MELTPLAN_LLM=off         disable the LLM entirely (tests, offline demos)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"   # backend/.env


# Loads KEY=VALUE pairs from backend/.env into the environment.
# Skips blanks and comments, strips quotes and an optional 'export ' prefix.
# Existing environment variables are never overwritten.
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
    # Configures the LLM client from environment variables and .env.
    # Reads base URL, model, API key, extra request body and the disable switch.
    # The underlying OpenAI client is created lazily on first use.
    def __init__(self):
        _load_env_file()
        self.base_url = os.environ.get("MELTPLAN_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.environ.get("MELTPLAN_LLM_MODEL", "z-ai/glm-5.3")
        self.api_key = os.environ.get("MELTPLAN_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.extra_body = json.loads(os.environ.get("MELTPLAN_LLM_EXTRA_BODY", "{}") or "{}")
        self.disabled = os.environ.get("MELTPLAN_LLM", "").lower() == "off"
        self._client = None

    # Reports whether the LLM can be used.
    # True only when an API key is set and the LLM is not disabled.
    # Callers fall back to deterministic templates when this is False.
    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self.disabled

    # Returns a short human-readable description of the LLM setup.
    # Shows model and base URL when available.
    # Otherwise states that the deterministic fallback is in use.
    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}" if self.available else "LLM unavailable (no key or disabled): deterministic fallback"

    # Sends one chat completion request, optionally with tools.
    # Rebuilds the client once if a pooled connection went stale.
    # Returns a normalised Reply with content, reasoning and tool calls.
    def chat(self, messages: list[dict], tools: list[dict] | None = None, timeout: float = 30.0,
             temperature: float = 0.2, max_tokens: int = 4096) -> Reply:
        """One chat completion. Raises openai errors (the caller falls back); never returns partial tool calls silently."""
        from openai import APIConnectionError, OpenAI
        kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        if tools:
            kw.update(tools=tools, tool_choice="auto")
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        for attempt in (1, 2):
            if self._client is None:
                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=1)
            try:
                r = self._client.chat.completions.create(**kw)
                break
            except APIConnectionError:   # a pooled connection went stale while the demo sat idle: rebuild the client once
                self._client = None
                if attempt == 2:
                    raise
        ch = r.choices[0]
        m = ch.message
        extra = m.model_extra or {}
        return Reply(content=m.content or "", reasoning=extra.get("reasoning_content") or extra.get("reasoning") or "",
                     tool_calls=[ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (m.tool_calls or [])],
                     finish_reason=ch.finish_reason or "")
