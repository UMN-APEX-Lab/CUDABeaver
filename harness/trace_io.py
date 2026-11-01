"""Stage trace artifact writer for experiment runs.

This module is intentionally passive: it records prompts, responses,
summaries, and artifact paths without affecting build/test/classification
behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


README_TEXT = """# Stage Traces

This directory records the input and output artifacts for each major harness
stage. It is meant as a navigation layer for agents and humans.

- `index.jsonl` lists every recorded stage with relative artifact paths.
- `messages.json` contains the exact chat messages sent to an LLM stage.
- `response.txt` contains the raw LLM response.
- `summary.json` contains a compact, manifest-friendly summary.
- `precheck/prompt_section.md` shows the precheck evidence injected into
  later agent prompts.
- `iter_*/generation/debug_context.json` links the first generation prompt
  back to precheck and localization trace directories.

Large raw logs remain in their stage files or in the existing run directories
such as `build_log_raw/` and `sanitizer_logs/`.
"""


class StageTraceWriter:
    """Write per-stage trace artifacts under ``outputs/<run>/stage_traces``."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        enabled: bool = True,
        inline_excerpt_chars: int = 1200,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "stage_traces"
        self.enabled = bool(enabled)
        self.inline_excerpt_chars = max(0, int(inline_excerpt_chars))
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            readme = self.root / "README.md"
            if not readme.exists():
                readme.write_text(README_TEXT, encoding="utf-8")

    @classmethod
    def from_config(cls, output_dir: Path | str, config: dict | None) -> "StageTraceWriter":
        logging_cfg = (config or {}).get("logging") or {}
        enabled = bool(logging_cfg.get("save_stage_traces", False))
        return cls(
            output_dir,
            enabled=enabled,
            inline_excerpt_chars=int(logging_cfg.get("trace_inline_excerpt_chars", 1200)),
        )

    def write_stage(
        self,
        task_stem: str,
        stage_parts: str | list[str] | tuple[str, ...],
        *,
        stage: str | None = None,
        task_id: str | None = None,
        sample_id: str | None = None,
        iteration: int | None = None,
        messages: list[dict] | None = None,
        response_text: str | None = None,
        parsed: Any | None = None,
        summary: dict | None = None,
        input_paths: dict[str, str] | None = None,
        output_paths: dict[str, str] | None = None,
        input_text_artifacts: dict[str, str] | None = None,
        output_text_artifacts: dict[str, str] | None = None,
        input_json_artifacts: dict[str, Any] | None = None,
        output_json_artifacts: dict[str, Any] | None = None,
    ) -> dict | None:
        """Write a stage directory and append one row to the trace index."""
        if not self.enabled:
            return None

        parts = [stage_parts] if isinstance(stage_parts, str) else list(stage_parts)
        safe_task = _safe_component(task_stem)
        safe_parts = [_safe_component(part) for part in parts]
        stage_dir = self.root / safe_task
        for part in safe_parts:
            stage_dir = stage_dir / part
        stage_dir.mkdir(parents=True, exist_ok=True)

        rel_inputs: dict[str, str] = dict(input_paths or {})
        rel_outputs: dict[str, str] = dict(output_paths or {})

        if messages is not None:
            path = stage_dir / "messages.json"
            path.write_text(
                json.dumps(messages, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            rel_inputs.setdefault("messages", self._rel(path))

        if response_text is not None:
            path = stage_dir / "response.txt"
            path.write_text(response_text, encoding="utf-8")
            rel_outputs.setdefault("response", self._rel(path))

        if parsed is not None:
            path = stage_dir / "parsed.json"
            path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            rel_outputs.setdefault("parsed", self._rel(path))

        self._write_artifacts(stage_dir, input_text_artifacts, text=True, paths=rel_inputs)
        self._write_artifacts(stage_dir, output_text_artifacts, text=True, paths=rel_outputs)
        self._write_artifacts(stage_dir, input_json_artifacts, text=False, paths=rel_inputs)
        self._write_artifacts(stage_dir, output_json_artifacts, text=False, paths=rel_outputs)

        summary_payload = {
            "task_id": task_id,
            "task_stem": task_stem,
            "sample_id": sample_id,
            "iteration": iteration,
            "stage": stage or "/".join(parts),
            "trace_dir": self._rel(stage_dir),
            "input_paths": rel_inputs,
            "output_paths": rel_outputs,
            "summary": _truncate_value(summary or {}, self.inline_excerpt_chars),
        }
        summary_path = stage_dir / "summary.json"
        summary_payload["output_paths"]["summary"] = self._rel(summary_path)
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        with (self.root / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_payload, ensure_ascii=False) + "\n")

        return summary_payload

    def _write_artifacts(
        self,
        stage_dir: Path,
        artifacts: dict[str, Any] | None,
        *,
        text: bool,
        paths: dict[str, str],
    ) -> None:
        for name, value in (artifacts or {}).items():
            path = stage_dir / _safe_filename(name)
            if text:
                path.write_text("" if value is None else str(value), encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(value, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            paths.setdefault(name, self._rel(path))

    def _rel(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.output_dir.resolve())
            return rel.as_posix()
        except Exception:
            return path.as_posix()


def compact_sanitizer_summary(
    sanitizer_summary: dict | None,
    *,
    output_excerpt_chars: int = 0,
    log_paths: dict[str, str] | None = None,
) -> dict | None:
    """Return a sanitizer summary without embedding full tool output."""
    if not sanitizer_summary:
        return None
    compact = {
        key: value
        for key, value in sanitizer_summary.items()
        if key != "results"
    }
    compact["results"] = {}
    for tool, info in (sanitizer_summary.get("results") or {}).items():
        out = info.get("output", "")
        tool_info = {key: value for key, value in info.items() if key != "output"}
        tool_info["output_chars"] = len(out or "")
        if log_paths and tool in log_paths:
            tool_info["log_path"] = log_paths[tool]
        compact["results"][tool] = tool_info
    return compact


def sanitizer_output_artifacts(sanitizer_summary: dict | None) -> dict[str, str]:
    """Extract full sanitizer outputs as text artifacts keyed by filename."""
    artifacts: dict[str, str] = {}
    for tool, info in ((sanitizer_summary or {}).get("results") or {}).items():
        output = info.get("output")
        if output:
            artifacts[f"sanitizer_{tool}.txt"] = output
    return artifacts


def _truncate_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    if isinstance(value, tuple):
        return [_truncate_value(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): _truncate_value(val, limit) for key, val in value.items()}
    return value


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _safe_component(value: Any) -> str:
    text = str(value or "unknown")
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "unknown"


def _safe_filename(value: Any) -> str:
    name = _safe_component(value)
    return name[:180] if len(name) > 180 else name
