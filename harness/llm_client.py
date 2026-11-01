"""
LLM API client module.

Extracted from run_experiment.py for independent testing and swappability.
Supports two backends:
  1. OpenAI-compatible API (GPT, Qwen cloud, DeepSeek cloud, etc.)
  2. vLLM-served local models (via OpenAI-compatible endpoint)

Backend is auto-detected from config["api"]["backend"]:
  - "openai" (default): real OpenAI API or compatible cloud providers
  - "vllm":  locally served model via `vllm serve`

Key differences handled:
  - vLLM: api_key optional (defaults to "EMPTY"), no reasoning_effort,
    uses max_tokens instead of max_completion_tokens
  - OpenAI: requires real api_key, supports reasoning_effort
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("cuda_debugger_exp")

_MISSING = object()
_SERVICE_LIMIT_ALIASES = {
    "service_limit",
    "model_service_limit",
    "model_limit",
    "context_limit",
    "max_model_len",
}
_FINALIZATION_OUTPUT_DISCIPLINE = (
    "\n\nOUTPUT REQUIREMENT:\n"
    "- do not overthink.\n"
    "- You must generate the required final deliverable now before stopping.\n"
    "- Keep any hidden reasoning brief; do not spend the response budget on private analysis.\n"
    "- Do not emit prose, markdown, comments, or analysis unless the requested deliverable explicitly requires it."
)
_JSON_FINALIZATION_OUTPUT_DISCIPLINE = (
    "\n\nOUTPUT REQUIREMENT:\n"
    "- do not overthink.\n"
    "- You must generate exactly one compact JSON action object now.\n"
    "- The JSON object must contain a tool name and an args object.\n"
    "- Start with `{` and end with `}`. No prose. No markdown. No analysis."
)


def is_context_length_error(exc: BaseException) -> bool:
    """Best-effort detector for provider context-window failures."""
    parts = [str(exc)]
    for attr in ("code", "type", "param"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    body = getattr(exc, "body", None)
    if body:
        parts.append(str(body))
    text = " ".join(parts).lower()
    return any(
        needle in text
        for needle in (
            "context_length_exceeded",
            "maximum context length",
            "max context length",
            "context window",
            "prompt is too long",
            "input is too long",
            "too many tokens",
            "reduce the length",
        )
    )


def detect_backend(config: dict) -> str:
    """Detect backend from config, with auto-detection heuristic."""
    explicit = config.get("api", {}).get("backend")
    if explicit:
        return explicit.lower()

    # Auto-detect: if base_url points to localhost / 127.0.0.1, likely vLLM
    base_url = config.get("api", {}).get("base_url") or ""
    if any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return "vllm"

    return "openai"


def create_client(config: dict):
    """
    Create an OpenAI-compatible client from config.

    For vLLM backend: api_key defaults to "EMPTY" if not set.
    For OpenAI backend: api_key is required from environment.
    """
    import os
    from openai import OpenAI

    backend = detect_backend(config)
    api_cfg = config.get("api", {})

    # Resolve API key
    api_key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")

    if backend == "vllm":
        # vLLM doesn't need a real key
        if not api_key:
            api_key = api_cfg.get("api_key", "EMPTY")
        base_url = api_cfg.get("base_url")
        if not base_url:
            # Default vLLM endpoint
            host = api_cfg.get("host", "localhost")
            port = api_cfg.get("port", 8000)
            base_url = f"http://{host}:{port}/v1"
        logger.info(f"Using vLLM backend: {base_url}, model={api_cfg.get('model', '?')}")
    else:
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set {api_key_env} environment variable, "
                f"or use backend: vllm for local models."
            )
        base_url = api_cfg.get("base_url")
        logger.info(f"Using OpenAI backend: model={api_cfg.get('model', '?')}")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if backend == "vllm":
        # Disable httpx connection-pool keepalive for vLLM. vLLM occasionally
        # closes idle conns under load; pooled clients miss the FIN and
        # accumulate CLOSE_WAIT until the server stops accepting new requests.
        # Local TCP reconnect is sub-ms so the throughput cost is negligible.
        import httpx
        kwargs["http_client"] = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=0)
        )
    return OpenAI(**kwargs)


def _configured_token_value(api_cfg: dict, primary: str, secondary: str) -> Any:
    value = api_cfg.get(primary, _MISSING)
    if value is _MISSING:
        value = api_cfg.get(secondary, 4096)
    return value


def _is_service_limit(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _SERVICE_LIMIT_ALIASES


def _model_context_limit(client, config: dict) -> int | None:
    api_cfg = config.get("api", {})
    for key in (
        "max_model_len",
        "model_max_len",
        "model_context_tokens",
        "max_context_tokens",
        "context_window",
    ):
        value = api_cfg.get(key)
        if value is not None:
            return int(value)

    cached = api_cfg.get("_resolved_max_model_len")
    if cached is not None:
        return int(cached)

    try:
        models = client.models.list()
    except Exception as e:
        logger.warning(f"Could not query model context limit from service: {e}")
        return None

    target = api_cfg.get("model")
    data = getattr(models, "data", None) or []
    chosen = None
    for model in data:
        mid = getattr(model, "id", None)
        root = getattr(model, "root", None)
        if target is None or mid == target or root == target:
            chosen = model
            break
    if chosen is None and data:
        chosen = data[0]

    limit = None
    if chosen is not None:
        limit = getattr(chosen, "max_model_len", None)
        if limit is None and isinstance(chosen, dict):
            limit = chosen.get("max_model_len")
    if limit is None:
        return None

    api_cfg["_resolved_max_model_len"] = int(limit)
    return int(limit)


def _estimate_prompt_tokens(messages: list[dict], chars_per_token: float) -> int:
    # Matches harness.context_manager's coarse accounting without importing it
    # into this low-level client module.
    import json

    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("role", ""))) + 4
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(str(content))
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            total_chars += len(str(fn.get("name", "")))
            args = fn.get("arguments", "")
            if isinstance(args, str):
                total_chars += len(args)
            elif args:
                total_chars += len(json.dumps(args))
        if m.get("tool_call_id"):
            total_chars += len(str(m["tool_call_id"]))
    return int(total_chars / max(chars_per_token, 1.0))


def _resolve_max_tokens(
    client,
    config: dict,
    messages: list[dict],
    value: Any,
    *,
    token_field: str,
) -> int | None:
    """Resolve an int/string token setting into the value sent to the API.

    `service_limit` means: query/read the model's total context window and
    allocate nearly all currently remaining room to output tokens. The
    reserve/estimator are conservative because vLLM validates
    prompt_tokens + max_tokens <= max_model_len before generation.
    """
    if value is None:
        return None
    if _is_service_limit(value):
        limit = _model_context_limit(client, config)
        if limit is None:
            raise RuntimeError(
                f"{token_field}=service_limit requires api.max_model_len "
                "or a /v1/models response with max_model_len"
            )
        api_cfg = config.get("api", {})
        chars_per_token = float(api_cfg.get("service_limit_chars_per_token", 2.0))
        reserve = int(api_cfg.get("service_limit_reserve_tokens", 8192))
        min_output = int(api_cfg.get("service_limit_min_output_tokens", 1024))
        prompt_tokens = _estimate_prompt_tokens(messages, chars_per_token)
        output_tokens = limit - prompt_tokens - reserve
        if output_tokens < min_output:
            output_tokens = max(1, limit - prompt_tokens)
        return int(output_tokens)
    return int(value)


def _set_token_limit(kwargs: dict, field: str, value: int | None) -> None:
    if value is not None:
        kwargs[field] = value


def _message_reasoning(message) -> str | None:
    return getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)


def _finalization_config(api_cfg: dict, section: str | None = None) -> dict:
    base = api_cfg.get("finalization")
    cfg = dict(base) if isinstance(base, dict) else {}
    if section:
        override = api_cfg.get(section)
        if isinstance(override, dict):
            cfg.update(override)
    return cfg


def _should_finalize_content(
    content: str | None,
    reasoning: str | None,
    cfg: dict,
) -> str | None:
    if not cfg.get("enabled", False):
        return None
    if not content:
        return "empty_content"
    threshold = cfg.get("reasoning_char_threshold")
    if threshold is not None and reasoning and len(reasoning) > int(threshold):
        return "reasoning_too_long"
    return None


def _reasoning_excerpt(reasoning: str | None, cfg: dict) -> str:
    if not reasoning:
        return ""
    max_chars = int(cfg.get("reasoning_excerpt_chars", 12000))
    if max_chars <= 0:
        return ""
    if len(reasoning) <= max_chars:
        return reasoning
    return reasoning[-max_chars:]


def _set_finalization_token_limit(
    client,
    config: dict,
    kwargs: dict,
    messages: list[dict],
    cfg: dict,
    *,
    backend: str,
) -> None:
    if "max_tokens" not in cfg:
        return
    value = cfg.get("max_tokens")
    token_field = "max_tokens" if backend == "vllm" else "max_completion_tokens"
    _set_token_limit(
        kwargs,
        token_field,
        _resolve_max_tokens(
            client,
            config,
            messages,
            value,
            token_field="api.finalization.max_tokens",
        ),
    )


def _call_chat_once(client, kwargs: dict) -> tuple[str | None, str | None, str | None, Any]:
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message
    u = getattr(response, "usage", None)
    if u is not None:
        # API-spend visibility (completion includes billed reasoning tokens)
        logger.info(
            f"  API usage: prompt={getattr(u, 'prompt_tokens', '?')} "
            f"completion={getattr(u, 'completion_tokens', '?')}"
        )
    return (
        getattr(msg, "content", None),
        _message_reasoning(msg),
        getattr(choice, "finish_reason", None),
        getattr(response, "usage", None),
    )


def _finalize_plain_content(
    client,
    config: dict,
    base_kwargs: dict,
    messages: list[dict],
    *,
    backend: str,
    reason: str,
    reasoning: str | None,
) -> tuple[str | None, str | None, str | None, Any]:
    api_cfg = config["api"]
    cfg = _finalization_config(api_cfg)
    excerpt = _reasoning_excerpt(reasoning, cfg)
    prompt = (
        "FINALIZATION REQUEST\n"
        f"The previous response ended with {reason} and did not provide a usable final answer.\n"
        "Keep thinking enabled, but do not continue private analysis in the final answer. "
        "Emit the final answer now in exactly the format requested by the immediately "
        "preceding user message. No extra explanation."
    )
    if excerpt:
        prompt += (
            "\n\nTail of the previous hidden reasoning, for continuity only:\n"
            f"```\n{excerpt}\n```"
        )
    prompt += _FINALIZATION_OUTPUT_DISCIPLINE
    final_messages = messages + [{"role": "user", "content": prompt}]
    final_kwargs = dict(base_kwargs)
    final_kwargs["messages"] = final_messages
    final_kwargs.pop("stream", None)
    _set_finalization_token_limit(
        client, config, final_kwargs, final_messages, cfg, backend=backend
    )
    logger.warning(
        "LLM finalization continuation triggered: "
        f"{reason}, reasoning={len(reasoning) if reasoning else 0} chars"
    )
    return _call_chat_once(client, final_kwargs)


def _finalize_json_action(
    client,
    config: dict,
    base_kwargs: dict,
    messages: list[dict],
    *,
    backend: str,
    reason: str,
    reasoning: str | None,
) -> tuple[str | None, str | None, str | None, Any]:
    api_cfg = config["api"]
    cfg = _finalization_config(api_cfg, "json_action_finalization")
    excerpt = _reasoning_excerpt(reasoning, cfg)
    prompt = (
        "FINALIZATION REQUEST FOR JSON ACTION\n"
        f"The previous response ended with {reason} and did not provide a usable final JSON action.\n"
        "Keep thinking enabled, but stop private analysis now and emit exactly one compact JSON object "
        "for the same ACTION REQUEST. Use only the available tool schema already shown. "
        "No prose. No markdown. Start with `{` and end with `}`."
    )
    if excerpt:
        prompt += (
            "\n\nTail of the previous hidden reasoning, for continuity only:\n"
            f"```\n{excerpt}\n```"
        )
    prompt += _JSON_FINALIZATION_OUTPUT_DISCIPLINE
    final_messages = messages + [{"role": "user", "content": prompt}]
    final_kwargs = dict(base_kwargs)
    final_kwargs["messages"] = final_messages
    final_kwargs.pop("stream", None)
    _set_finalization_token_limit(
        client, config, final_kwargs, final_messages, cfg, backend=backend
    )
    logger.warning(
        "JSON action finalization continuation triggered: "
        f"{reason}, reasoning={len(reasoning) if reasoning else 0} chars"
    )
    return _call_chat_once(client, final_kwargs)


def call_llm(
    client,
    config: dict,
    messages: list[dict],
    max_tokens: int | str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Call the LLM and return the assistant message content.

    Adapts parameters based on backend:
      - openai: max_completion_tokens, reasoning_effort
      - vllm:   max_tokens, temperature (no reasoning_effort)
    """
    api_cfg = config["api"]
    backend = detect_backend(config)

    kwargs = dict(
        model=api_cfg["model"],
        messages=messages,
        # No HTTP timeout on the LLM call. The 300s in configs is for the
        # generated code's build/test (build.timeout / build.test_timeout),
        # not for model generation, which can legitimately run >5min on slow
        # local vLLM (Qwen3.6-27B single-GPU @ 16K tokens). Opt back in by
        # setting api.http_timeout if a hard cap is ever needed.
        timeout=api_cfg.get("http_timeout"),
    )

    if backend == "vllm":
        # vLLM uses max_tokens, not max_completion_tokens
        _set_token_limit(
            kwargs,
            "max_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_tokens", "max_completion_tokens")
                if max_tokens is None
                else max_tokens,
                token_field="api.max_tokens" if max_tokens is None else "agent.tool_decision_max_tokens",
            ),
        )
        kwargs["temperature"] = api_cfg.get("temperature", 0.7) if temperature is None else temperature
        # vLLM-specific params
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]
        if api_cfg.get("repetition_penalty") is not None:
            kwargs["extra_body"] = {"repetition_penalty": api_cfg["repetition_penalty"]}
        if api_cfg.get("stop"):
            kwargs["stop"] = api_cfg["stop"]
    else:
        # OpenAI / compatible cloud API
        _set_token_limit(
            kwargs,
            "max_completion_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_completion_tokens", "max_tokens")
                if max_tokens is None
                else max_tokens,
                token_field=(
                    "api.max_completion_tokens"
                    if max_tokens is None
                    else "agent.tool_decision_max_tokens"
                ),
            ),
        )
        reasoning_effort = api_cfg.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = api_cfg.get("temperature", 0.7) if temperature is None else temperature
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]

    # Generic extra_body passthrough (e.g. NVIDIA-hosted models'
    # chat_template_kwargs for thinking control). Merges with anything
    # already set above (vLLM repetition_penalty).
    if api_cfg.get("extra_body"):
        kwargs.setdefault("extra_body", {}).update(api_cfg["extra_body"])

    # Per-call seed (used by repeated sampling — Exp 2). vLLM honours this;
    # most cloud providers ignore it (we record actual reproducibility in
    # _meta.seed_supported elsewhere).
    if api_cfg.get("seed") is not None:
        kwargs["seed"] = int(api_cfg["seed"])

    # Streaming path: needed for endpoints with strict gateway timeouts
    # (e.g. NVIDIA NIM 5-min cap on long thinking-mode generations).
    # Accumulates content + reasoning_content chunks separately.
    use_stream = bool(api_cfg.get("stream", False))

    def _one_call() -> tuple[str | None, str | None]:
        if use_stream:
            kwargs["stream"] = True
            cps: list[str] = []
            rps: list[str] = []
            for chunk in client.chat.completions.create(**kwargs):
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                c = getattr(delta, "content", None)
                if c:
                    cps.append(c)
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if r:
                    rps.append(r)
            return ("".join(cps) if cps else None, "".join(rps) if rps else None)
        else:
            content, reasoning, _finish_reason, _usage = _call_chat_once(client, kwargs)
            return content, reasoning

    # Retry on empty content OR transient network/stream errors. NVIDIA-hosted
    # endpoints occasionally drop streams mid-flight (RemoteProtocolError) or
    # finish reasoning without emitting any content tokens.
    # api.retry_on_empty: int, default 2 extra attempts.
    import httpx
    transient_errors = (
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.ConnectError,
    )

    def _try_once_with_catch() -> tuple[str | None, str | None, Exception | None]:
        try:
            c, r = _one_call()
            return c, r, None
        except transient_errors as e:
            return None, None, e

    max_retries = int(api_cfg.get("retry_on_empty", 2))
    content, reasoning, last_err = _try_once_with_catch()
    attempts = 1
    final_cfg = _finalization_config(api_cfg)
    final_reason = None if last_err else _should_finalize_content(content, reasoning, final_cfg)
    if final_reason:
        original_content, original_reasoning = content, reasoning
        content, reasoning, _finish_reason, _usage = _finalize_plain_content(
            client,
            config,
            kwargs,
            messages,
            backend=backend,
            reason=final_reason,
            reasoning=reasoning,
        )
        if not content and original_content:
            content = original_content
        if not content:
            reasoning = reasoning or original_reasoning

    while (not content) and (not final_reason) and attempts <= max_retries:
        why = (
            f"transient stream error: {type(last_err).__name__}"
            if last_err
            else f"empty content (reasoning={len(reasoning) if reasoning else 0} chars)"
        )
        logger.warning(f"LLM call failed/empty: {why}. Retrying ({attempts}/{max_retries})...")
        content, reasoning, last_err = _try_once_with_catch()
        attempts += 1
    if last_err and not content and not reasoning:
        # All retries exhausted — re-raise so run_experiment can record API error.
        raise last_err

    # Final fallback: if still empty, optionally use reasoning as last resort.
    if not content:
        fallback_reasoning = api_cfg.get(
            "reasoning_fallback_as_content",
            final_cfg.get("fallback_reasoning_as_content", True),
        )
        if reasoning and fallback_reasoning:
            logger.warning(
                f"LLM content still None after {attempts} attempts. "
                f"Falling back to reasoning ({len(reasoning)} chars) as content."
            )
            content = reasoning
        elif reasoning:
            content = ""
            logger.warning(
                f"LLM returned empty final content after {attempts} attempts "
                f"(reasoning={len(reasoning)} chars; reasoning fallback disabled)"
            )
        else:
            content = ""
            logger.warning(f"LLM returned empty response after {attempts} attempts (content+reasoning both empty)")

    return content


def extract_code(response_text: str) -> str:
    """Extract code from LLM response, stripping markdown fences if present."""
    text = response_text.strip()
    m = re.search(r"```(?:cuda|cpp|c\+\+|python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def call_llm_with_tools(
    client,
    config: dict,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
    max_tokens: int | str | None = None,
    temperature: float | None = None,
) -> dict:
    """Call the LLM with OpenAI-compatible tool calling enabled."""
    api_cfg = config["api"]
    backend = detect_backend(config)

    kwargs = dict(
        model=api_cfg["model"],
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        timeout=api_cfg.get("http_timeout"),
    )

    if backend == "vllm":
        _set_token_limit(
            kwargs,
            "max_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_tokens", "max_completion_tokens")
                if max_tokens is None
                else max_tokens,
                token_field="agent.tool_decision_max_tokens"
                if max_tokens is not None
                else "api.max_tokens",
            ),
        )
        kwargs["temperature"] = (
            api_cfg.get("temperature", 0.7)
            if temperature is None
            else temperature
        )
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]
        if api_cfg.get("repetition_penalty") is not None:
            kwargs["extra_body"] = {"repetition_penalty": api_cfg["repetition_penalty"]}
    else:
        _set_token_limit(
            kwargs,
            "max_completion_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_completion_tokens", "max_tokens")
                if max_tokens is None
                else max_tokens,
                token_field="agent.tool_decision_max_tokens"
                if max_tokens is not None
                else "api.max_completion_tokens",
            ),
        )
        reasoning_effort = api_cfg.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = (
                api_cfg.get("temperature", 0.7)
                if temperature is None
                else temperature
            )
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]

    if api_cfg.get("extra_body"):
        kwargs.setdefault("extra_body", {}).update(api_cfg["extra_body"])
    if api_cfg.get("seed") is not None:
        kwargs["seed"] = int(api_cfg["seed"])

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message

    tool_calls_out: list[dict] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        tool_calls_out.append({
            "id": getattr(tc, "id", None) or "",
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": getattr(fn, "name", "") if fn else "",
                "arguments": getattr(fn, "arguments", "") if fn else "",
            },
        })

    return {
        "content": getattr(msg, "content", None),
        "tool_calls": tool_calls_out,
        "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "request_max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        "usage": getattr(response, "usage", None),
    }


def call_llm_json_action(
    client,
    config: dict,
    messages: list[dict],
    max_tokens: int | str | None = None,
    temperature: float | None = None,
    response_format: bool = True,
) -> dict:
    """Call the LLM without native tools and ask for a JSON action object."""
    api_cfg = config["api"]
    backend = detect_backend(config)

    kwargs = dict(
        model=api_cfg["model"],
        messages=messages,
        timeout=api_cfg.get("http_timeout"),
    )

    if response_format:
        kwargs["response_format"] = {"type": "json_object"}

    if backend == "vllm":
        _set_token_limit(
            kwargs,
            "max_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_tokens", "max_completion_tokens")
                if max_tokens is None
                else max_tokens,
                token_field="agent.tool_decision_max_tokens"
                if max_tokens is not None
                else "api.max_tokens",
            ),
        )
        kwargs["temperature"] = (
            api_cfg.get("temperature", 0.7)
            if temperature is None
            else temperature
        )
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]
        if api_cfg.get("repetition_penalty") is not None:
            kwargs["extra_body"] = {"repetition_penalty": api_cfg["repetition_penalty"]}
    else:
        _set_token_limit(
            kwargs,
            "max_completion_tokens",
            _resolve_max_tokens(
                client,
                config,
                messages,
                _configured_token_value(api_cfg, "max_completion_tokens", "max_tokens")
                if max_tokens is None
                else max_tokens,
                token_field="agent.tool_decision_max_tokens"
                if max_tokens is not None
                else "api.max_completion_tokens",
            ),
        )
        reasoning_effort = api_cfg.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = (
                api_cfg.get("temperature", 0.7)
                if temperature is None
                else temperature
            )
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]

    if api_cfg.get("extra_body"):
        kwargs.setdefault("extra_body", {}).update(api_cfg["extra_body"])
    if api_cfg.get("seed") is not None:
        kwargs["seed"] = int(api_cfg["seed"])

    content, reasoning, finish_reason, usage = _call_chat_once(client, kwargs)
    final_cfg = _finalization_config(api_cfg, "json_action_finalization")
    final_reason = _should_finalize_content(content, reasoning, final_cfg)
    finalization_triggered = False
    original_finish_reason = finish_reason
    original_reasoning_chars = len(reasoning) if reasoning else 0
    if final_reason:
        original_content, original_reasoning = content, reasoning
        content, reasoning, finish_reason, usage = _finalize_json_action(
            client,
            config,
            kwargs,
            messages,
            backend=backend,
            reason=final_reason,
            reasoning=reasoning,
        )
        finalization_triggered = True
        if not content and original_content:
            content = original_content
        if not content:
            reasoning = reasoning or original_reasoning
    return {
        "content": content,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "request_max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        "usage": usage,
        "finalization_triggered": finalization_triggered,
        "finalization_reason": final_reason,
        "original_finish_reason": original_finish_reason,
        "original_reasoning_chars": original_reasoning_chars,
    }
