"""
llm_backends.py -- pluggable LLM backend for the optional --llm escalation
=============================================================
Lets the concise LLM-summary feature (collect.py --llm) call Claude, OpenAI
(GPT / Codex models), or Gemini instead of a single hardcoded provider.

Config resolution order (first file found wins), all relative to the CWD the
Snakemake process runs from (a pipeline's root dir, not this repo dir):
  1. ./.snakemake_ai_debugger.yaml                            (project-local)
  2. ./snakemake-ai-debugger/.snakemake_ai_debugger.yaml      (this repo nested
     one level under the pipeline root, which is how it's checked out today)
  3. ~/.config/snakemake-ai-debugger/config.yaml              (user-wide)

Config schema -- backend and model must both be set explicitly (CLI flag or
config file); there is no assumed default, since a guessed model name for a
fast-moving provider is more likely to be wrong than helpful:

    llm:
      backend: claude        # claude | openai | gemini
      model: claude-opus-5   # backend-specific model id
                              #   openai:  e.g. gpt-5, gpt-5-codex
                              #   gemini:  e.g. gemini-2.5-pro
    api_keys:
      claude: sk-ant-...      # optional -- literal keys here work, but keep
      openai: sk-...          # this file out of version control if you use
      gemini: ...             # them. Omit a key to fall back to its env var.

API key resolution per backend, in order: api_keys.<backend> in the config
file, then the provider's standard environment variable
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

KNOWN_BACKENDS = ("claude", "openai", "gemini")

_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_CONFIG_LOCATIONS = [
    Path(".snakemake_ai_debugger.yaml"),
    Path("snakemake-ai-debugger") / ".snakemake_ai_debugger.yaml",
    Path.home() / ".config" / "snakemake-ai-debugger" / "config.yaml",
]


def _load_config() -> dict:
    for path in _CONFIG_LOCATIONS:
        if path.exists():
            try:
                return yaml.safe_load(path.read_text()) or {}
            except Exception:
                return {}
    return {}


def resolve(backend: Optional[str] = None, model: Optional[str] = None) -> tuple[str, str, str]:
    """
    Work out (backend, model, api_key) from CLI overrides + config file.
    backend/model have no built-in default -- both must come from --backend/
    --model or from llm.backend/llm.model in the config file.
    """
    cfg = _load_config()
    llm_cfg = cfg.get("llm") or {}

    backend = backend or llm_cfg.get("backend")
    if not backend:
        raise RuntimeError(
            "No LLM backend configured. Pass --backend {claude,openai,gemini} "
            "or set llm.backend in .snakemake_ai_debugger.yaml / "
            "~/.config/snakemake-ai-debugger/config.yaml"
        )
    if backend not in KNOWN_BACKENDS:
        raise RuntimeError(
            f"Unknown LLM backend '{backend}'. Choose from: {', '.join(KNOWN_BACKENDS)}"
        )

    model = model or llm_cfg.get("model")
    if not model:
        raise RuntimeError(
            f"No model configured for backend '{backend}'. Pass --model <name> "
            f"or set llm.model in .snakemake_ai_debugger.yaml / "
            f"~/.config/snakemake-ai-debugger/config.yaml"
        )

    api_key = (cfg.get("api_keys") or {}).get(backend) or os.environ.get(_ENV_VARS[backend], "")
    if not api_key:
        raise RuntimeError(
            f"No API key for backend '{backend}'. Set ${_ENV_VARS[backend]} or add "
            f"api_keys.{backend} to .snakemake_ai_debugger.yaml"
        )
    return backend, model, api_key


def call_llm(
    prompt: str,
    system: str,
    backend: str,
    model: str,
    api_key: str,
    max_tokens: int = 400,
) -> str:
    if backend == "claude":
        return _call_claude(prompt, system, model, api_key, max_tokens)
    if backend == "openai":
        return _call_openai(prompt, system, model, api_key, max_tokens)
    if backend == "gemini":
        return _call_gemini(prompt, system, model, api_key, max_tokens)
    raise RuntimeError(f"Unknown backend: {backend}")


def _call_claude(prompt: str, system: str, model: str, api_key: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Run: pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, system: str, model: str, api_key: str, max_tokens: int) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("Run: pip install openai")
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_gemini(prompt: str, system: str, model: str, api_key: str, max_tokens: int) -> str:
    # google-generativeai is deprecated (end-of-support); google-genai is its
    # replacement SDK -- see https://github.com/googleapis/python-genai
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Run: pip install google-genai")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            # Gemini 2.5+ models spend max_output_tokens on hidden "thinking"
            # tokens before the visible answer, and thinking_budget=0 to
            # disable it 400s on some model aliases (e.g. gemini-flash-latest)
            # -- so give the budget enough headroom for thinking + the answer
            # instead of fighting per-model thinking support.
            max_output_tokens=max(max_tokens, 500),
        ),
    )
    return resp.text
