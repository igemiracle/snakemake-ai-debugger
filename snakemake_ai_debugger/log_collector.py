"""
log_collector.py -- Slurm log harvester
=========================================
The real error scene is in .snakemake/slurm_logs/, not the main log.

Directory layout (Snakemake standard):
    .snakemake/slurm_logs/
        <rule_name>/
            <wildcards>/          <- may not exist for rules without wildcards
                <job_id>.log
            <job_id>.log          <- may also be directly under the rule directory

This module:
  1. Given a rule_name (from the main log), locates the slurm_logs subdirectory
  2. Finds the most recent job log (supports deeply nested wildcard paths)
  3. Returns a structured SlurmLogResult with the path and tail content
  4. Falls back to paths declared in the Snakefile log: field

Priority chain (highest to lowest):
  P1  .snakemake/slurm_logs/<rule>/**/<job_id>.log   <- latest mtime
  P2  Snakefile log: field paths
  P3  --log CLI argument
  P4  empty string (graceful degradation for Tier-1/Tier-2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Lines containing any of these tokens are treated as error signals.
_ERROR_LINE_RE = re.compile(
    r"\b(error|exception|traceback|fatal|killed|failed|abort"
    r"|segfault|core\s+dumped|oom|out\s+of\s+memory"
    r"|exit\s+code\s+[1-9]|errno\s+\d+|\[E::"
    r"|ruleexception|warning)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class SlurmLogResult:
    rule_name:   str
    log_path:    Optional[Path]   # None = not found
    wildcards:   str              # e.g. "sample=SRR123" or "" if no wildcards
    job_id:      str              # from filename, or ""
    content:     str              # full log content (up to max_lines), for pattern matching
    source:      str              # "slurm_logs" | "snakemake_log_field" | "cli_arg" | "none"

    @property
    def found(self) -> bool:
        return self.log_path is not None and bool(self.content.strip())

    @property
    def error_context(self) -> str:
        """Lines around error-signal tokens — better than a raw tail for display/LLM."""
        return _extract_error_context(self.content)

    def display_path(self) -> str:
        """Short path for terminal output."""
        if not self.log_path:
            return "(not found)"
        try:
            return str(self.log_path.relative_to(Path.cwd()))
        except ValueError:
            return str(self.log_path)


# ─────────────────────────────────────────────────────────────────
# Core: find the slurm log for a given rule
# ─────────────────────────────────────────────────────────────────

def find_slurm_log(
    rule_name:     str,
    slurm_log_dir: Path,           # typically .snakemake/slurm_logs
    tail_lines:    int = 2000,     # kept for API compat; now reads up to this many lines
    external_jobid: str = "",      # Slurm job id — used for exact filename match when set
) -> SlurmLogResult:
    """
    Locate the Slurm job log for `rule_name`.

    When `external_jobid` is given (parsed from Snakemake's main log), searches for
    <external_jobid>.log anywhere under the rule directory — exact match, no ambiguity.
    Falls back to the most-recent-mtime file when external_jobid is absent.

    Handles both layouts:
      slurm_logs/<rule>/                  <job>.log
      slurm_logs/<rule>/<wildcards>/      <job>.log
      slurm_logs/<rule>/<w1>/<w2>/...     <job>.log   (deeply nested)
    """
    # Try plain name first, then with the "rule_" prefix Snakemake sometimes adds
    rule_dir = slurm_log_dir / rule_name
    if not rule_dir.exists():
        rule_dir = slurm_log_dir / f"rule_{rule_name}"
    if not rule_dir.exists():
        return SlurmLogResult(
            rule_name=rule_name, log_path=None,
            wildcards="", job_id="", content="", source="none",
        )

    # Collect ALL .log files anywhere under this rule's directory
    all_logs = list(rule_dir.rglob("*.log"))
    if not all_logs:
        return SlurmLogResult(
            rule_name=rule_name, log_path=None,
            wildcards="", job_id="", content="", source="none",
        )

    # Prefer exact match on external_jobid (Slurm job id in the filename)
    best: Path | None = None
    if external_jobid:
        for p in all_logs:
            if _extract_job_id(p.stem) == external_jobid:
                best = p
                break

    if best is None:
        candidates = sorted(all_logs, key=lambda p: p.stat().st_mtime, reverse=True)
        best = candidates[0]

    # Derive wildcards string from the path between rule_dir and the log file
    # e.g. rule_dir=.../align_reads  best=.../align_reads/SRR123/456.log
    #   → wildcards = "SRR123"
    rel = best.relative_to(rule_dir)
    parts = rel.parts          # e.g. ("SRR123", "456.log") or ("456.log",)
    wildcard_parts = parts[:-1]
    wildcards = "/".join(wildcard_parts) if wildcard_parts else ""

    # job_id = stem of the filename (digits, possibly "slurm-12345")
    job_id = _extract_job_id(best.stem)

    content = _tail(best, tail_lines)

    return SlurmLogResult(
        rule_name=rule_name,
        log_path=best,
        wildcards=wildcards,
        job_id=job_id,
        content=content,
        source="slurm_logs",
    )


def find_all_slurm_logs(
    rule_names:   list[str],
    slurm_log_dir: Path,
    tail_lines:   int = 2000,
) -> dict[str, SlurmLogResult]:
    """Batch version (no external_jobid): returns {rule_name: SlurmLogResult}."""
    return {name: find_slurm_log(name, slurm_log_dir, tail_lines) for name in rule_names}


# ─────────────────────────────────────────────────────────────────
# Priority-chain resolver
# ─────────────────────────────────────────────────────────────────

def resolve_rule_log(
    rule_name:     str,
    slurm_log_dir: Path,
    declared_logs: list[Path],    # from Snakefile `log:` field
    cli_log:       Optional[Path],
    tail_lines:    int = 2000,
    external_jobid: str = "",     # Slurm job id for exact log file matching
) -> SlurmLogResult:
    """
    Walk the priority chain and return the first source that has content.

    P1: slurm_logs directory (most accurate for cluster jobs)
    P2: Snakefile log: field (works for local / non-cluster runs)
    P3: CLI --log argument
    """
    # P1 — Slurm logs
    result = find_slurm_log(rule_name, slurm_log_dir, tail_lines, external_jobid)
    if result.found:
        return result

    # P2 — declared log files from Snakefile
    for path in declared_logs:
        if path.exists() and path.stat().st_size > 0:
            content = _tail(path, tail_lines)
            if content.strip():
                return SlurmLogResult(
                    rule_name=rule_name, log_path=path,
                    wildcards="", job_id="",
                    content=content, source="snakemake_log_field",
                )

    # P3 — CLI override (only used when debugging a single rule)
    if cli_log and cli_log.exists():
        content = _tail(cli_log, tail_lines)
        return SlurmLogResult(
            rule_name=rule_name, log_path=cli_log,
            wildcards="", job_id="",
            content=content, source="cli_arg",
        )

    # P4 — nothing found, return empty result
    return SlurmLogResult(
        rule_name=rule_name, log_path=None,
        wildcards="", job_id="", content="", source="none",
    )


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _tail(path: Path, n: int) -> str:
    """Read up to `n` lines from path, stripping leading blank lines."""
    try:
        lines = path.read_text(errors="replace").splitlines()
        while lines and not lines[0].strip():
            lines = lines[1:]
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _extract_error_context(
    content: str,
    context_lines: int = 6,
    max_blocks: int = 5,
    fallback_lines: int = 30,
) -> str:
    """
    Scan `content` for error-signal lines and return context windows around them.

    Each matched line gets `context_lines` of surrounding context.  Overlapping
    windows are merged.  Falls back to the last `fallback_lines` lines when no
    error signals are found (e.g. a silent-failure log).

    This is intentionally better than a raw tail: in mixed stdout/stderr logs
    the actual error is often buried in the middle, not at the end.
    """
    lines = content.splitlines()
    if not lines:
        return ""

    hit_indices = [i for i, ln in enumerate(lines) if _ERROR_LINE_RE.search(ln)]

    if not hit_indices:
        return "\n".join(lines[-fallback_lines:])

    # Store (start, end) pairs so that merging preserves the original start
    ranges: list[tuple[int, int]] = []
    prev_end = -1
    for idx in hit_indices:
        start = max(0, idx - context_lines)
        end   = min(len(lines), idx + context_lines + 1)
        if start > prev_end:
            ranges.append((start, end))
            prev_end = end
            if len(ranges) >= max_blocks:
                break
        else:
            # Extend current range: keep original start, push end forward
            ranges[-1] = (ranges[-1][0], end)
            prev_end = end

    return "\n[...]\n".join("\n".join(lines[s:e]) for s, e in ranges)


def _extract_job_id(stem: str) -> str:
    """
    Extract numeric job ID from filenames like:
      '12345', 'slurm-12345', '12345_0', 'job_12345'
    """
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else stem


def list_slurm_rules(slurm_log_dir: Path) -> list[str]:
    """
    Return all rule names that have entries in slurm_logs.
    Useful for discovery / debug.
    """
    if not slurm_log_dir.exists():
        return []
    return sorted(
        p.name for p in slurm_log_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def describe_slurm_log_dir(slurm_log_dir: Path) -> str:
    """One-line summary of what's in slurm_logs — for --dry-run output."""
    rules = list_slurm_rules(slurm_log_dir)
    if not rules:
        return f"{slurm_log_dir} (empty or not found)"
    return f"{slurm_log_dir}  [{len(rules)} rule(s): {', '.join(rules[:6])}{'…' if len(rules)>6 else ''}]"
