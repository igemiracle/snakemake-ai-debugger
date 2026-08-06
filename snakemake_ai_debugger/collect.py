"""
collect.py -- Error collection and display
=========================================
Responsibilities:
  1. Parse failed rule names from the Snakemake main log
  2. Find the real error content in slurm_logs (see priority chain in log_collector.py)
  3. Run Tier-1 rule-based classifier (identification only — fast, free, no fixes)
  4. Coloured terminal output + write YAML report
  5. Optional: --llm flag makes ONE combined call to the configured LLM
     across all *distinct* error groups and prints its free-form summary of
     what's actually wrong — no rigid per-field template, no hand-maintained
     fix string for every possible tool error in rules.py.

Trigger modes:
  A. Snakefile onerror hook (recommended) -- add two lines at the top:
       from snakemake_ai_debugger.collect import on_error_hook
       onerror: on_error_hook(log)

  B. Manual CLI:
       snakemake-collect --slurm-log-dir .snakemake/slurm_logs
       snakemake-collect --llm                    # add LLM-generated fixes
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .log_collector import resolve_rule_log, describe_slurm_log_dir, SlurmLogResult
from .rules import quick_diagnose, QuickDiagnosis
from . import llm_backends

# ── ANSI colours ──────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED   = "\033[31m"; YELLOW = "\033[33m"; GREEN = "\033[32m"
CYAN  = "\033[36m"; MAGENTA = "\033[35m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


# ─────────────────────────────────────────────────────────────────
# Step 1: Parse failed rules from the Snakemake main log
#   The main log is only a failure notification — extract rule names and metadata only
# ─────────────────────────────────────────────────────────────────

import re

@dataclass
class FailedJobMeta:
    """Minimal metadata extracted from the Snakemake main log for one failed job."""
    rule_name:     str
    jobid:         str        # Snakemake internal DAG job id
    external_jobid: str       # Slurm job id (from external_jobid: line) — "" if absent
    wildcards:     str        # extracted from the wildcards: line in the main log
    input_files:   list[str]
    output_files:  list[str]
    declared_logs: list[Path]  # paths from Snakefile log: field (P2 fallback)
    block_text:    str        # raw error block, kept for Tier-2 fallback


def _parse_submission_info(text: str) -> dict[str, tuple[str, str]]:
    """
    Scan submission blocks for wildcards and slurm log paths, keyed by internal jobid.

    Snakemake prints both pieces of info in the pre-submission section:
      rule convert_coords:
          jobid: 88
          wildcards: out=out, sample=SKNAS_MYCN_4h_+Dox
      Job 88 has been submitted with SLURM jobid 14825379 (log: /path/to/14825379.log).

    The `Error in rule` blocks that follow have the external_jobid but NOT wildcards.
    Returns {jobid_str: (wildcards, external_jobid)}.
    """
    info: dict[str, tuple[str, str]] = {}
    # Each submission block starts with a timestamp + "rule <name>:"
    for block_m in re.finditer(
        r"rule \S+:\n(.*?)(?=\n\[|\Z)", text, re.DOTALL
    ):
        block = block_m.group(0)
        jm = re.search(r"\bjobid:\s*(\S+)", block)
        if not jm:
            continue
        jobid = jm.group(1)
        wm = re.search(r"\bwildcards:\s*(.+)", block)
        wildcards = wm.group(1).strip() if wm else ""
        # "Job N has been submitted with SLURM jobid XXXXX (log: ...)"
        sm = re.search(
            r"Job \d+ has been submitted with SLURM jobid (\S+)\s+\(log:\s+(\S+)\)",
            block,
        )
        ext_jobid = sm.group(1) if sm else ""
        if jobid and (wildcards or ext_jobid):
            existing = info.get(jobid, ("", ""))
            info[jobid] = (wildcards or existing[0], ext_jobid or existing[1])
    return info


def parse_main_log(text: str) -> list[FailedJobMeta]:
    """
    Scan the main log and extract one FailedJobMeta per 'Error in rule X:' block.
    Parses metadata only — no diagnosis at this stage.
    """
    # Pre-scan submission blocks to get wildcards + slurm log path per jobid
    submission_info = _parse_submission_info(text)

    jobs: list[FailedJobMeta] = []
    # Split on 'Error in rule' boundaries; process each section independently
    sections = re.split(r"(?=\nError in rule |\[.{5,30}\]\nError in rule )", text)

    for section in sections:
        m = re.search(r"Error in rule (\S+?):", section)
        if not m:
            continue
        rule_name = m.group(1)

        # Extract the error block: from 'Error in rule' to the next timestamp or end of text
        block_lines: list[str] = []
        capturing = False
        for line in section.splitlines():
            if "Error in rule" in line:
                capturing = True
            if capturing:
                if re.match(r"\[.{10,30}\]", line) and block_lines:
                    break
                block_lines.append(line)
        block = "\n".join(block_lines)

        # wildcards: line printed by Snakemake in the main log
        wc_match = re.search(r"\bwildcards:\s*(.+)", block)
        wildcards = wc_match.group(1).strip() if wc_match else ""

        # jobid (Snakemake internal) and external_jobid (Slurm) from the error block
        jm = re.search(r"\bjobid:\s*(\S+)", block)
        jobid = jm.group(1) if jm else ""
        em = re.search(r"\bexternal_jobid:\s*(\S+)", block)
        external_jobid = em.group(1) if em else ""

        # log: paths declared in the Snakefile rule
        declared_logs: list[Path] = []
        for line in block.splitlines():
            lm = re.match(r"\s+log:\s+(.+)", line)
            if lm:
                for p in lm.group(1).split(","):
                    p = p.strip()
                    if p:
                        declared_logs.append(Path(p))

        # Fill wildcards / external_jobid from submission block when missing in error block
        if jobid in submission_info:
            sub_wildcards, sub_ext = submission_info[jobid]
            wildcards     = wildcards     or sub_wildcards
            external_jobid = external_jobid or sub_ext

        jobs.append(FailedJobMeta(
            rule_name=rule_name,
            jobid=jobid,
            external_jobid=external_jobid,
            wildcards=wildcards,
            input_files=_file_list(block, "input"),
            output_files=_file_list(block, "output"),
            declared_logs=declared_logs,
            block_text=block,
        ))

    # Snakemake logs the same job multiple times in different block styles:
    #   style A: has wildcards but no external_jobid
    #   style B: has external_jobid but no wildcards
    # Merge both styles per (rule_name, jobid) so nothing is lost.
    merged: dict[tuple[str, str], FailedJobMeta] = {}
    for j in jobs:
        key = (j.rule_name, j.jobid)
        if key not in merged:
            merged[key] = j
        else:
            existing = merged[key]
            merged[key] = FailedJobMeta(
                rule_name     = existing.rule_name,
                jobid         = existing.jobid,
                external_jobid = existing.external_jobid or j.external_jobid,
                wildcards     = existing.wildcards or j.wildcards,
                input_files   = existing.input_files or j.input_files,
                output_files  = existing.output_files or j.output_files,
                declared_logs = existing.declared_logs or j.declared_logs,
                block_text    = existing.block_text,
            )
    return list(merged.values())


def _file_list(text: str, keyword: str) -> list[str]:
    files: list[str] = []
    collecting = False
    for line in text.splitlines():
        if re.match(rf"\s+{keyword}:", line):
            collecting = True
            rest = re.sub(rf"\s+{keyword}:\s*", "", line)
            files += [f.strip() for f in rest.split(",") if f.strip()]
            continue
        if collecting:
            if re.match(r"\s+(output|input|log|jobid|wildcards|conda|threads|resources):", line):
                break
            files += [f.strip() for f in line.split(",") if f.strip()]
    return files


def _find_latest_log(log_dir: Path) -> Optional[Path]:
    try:
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[0] if logs else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Step 2: Collected result structure
# ─────────────────────────────────────────────────────────────────

@dataclass
class CollectedError:
    rule_name:    str
    wildcards:    str          # from main log OR slurm path
    jobid:        str
    slurm_result: SlurmLogResult
    diagnosis:    Optional[QuickDiagnosis]   # None = unrecognised
    input_files:  list[str]
    output_files: list[str]

    @property
    def recognised(self) -> bool:
        return self.diagnosis is not None

    def to_dict(self) -> dict:
        d: dict = {
            "rule":      self.rule_name,
            "jobid":     self.jobid,
            "log_source": self.slurm_result.source,
        }
        if self.wildcards:
            d["wildcards"] = self.wildcards
        if self.slurm_result.log_path:
            d["log_path"] = self.slurm_result.display_path()
        if self.diagnosis:
            d["error_type"]      = self.diagnosis.error_type
            d["root_cause"]      = self.diagnosis.root_cause
            d["confidence"]      = self.diagnosis.confidence
            d["matched_pattern"] = self.diagnosis.matched_pattern
            if self.diagnosis.matched_text:
                d["matched_text"] = self.diagnosis.matched_text
        else:
            d["error_type"] = "Unrecognised"
            d["root_cause"] = "(no matching pattern — run with --llm to escalate)"
            ctx = self.slurm_result.error_context
            if ctx:
                d["error_context"] = ctx
        return d


# ─────────────────────────────────────────────────────────────────
# Step 3: Main collection function
# ─────────────────────────────────────────────────────────────────

def collect(
    log_dir:       Path = Path(".snakemake/log"),
    slurm_log_dir: Path = Path(".snakemake/slurm_logs"),
    cli_log:       Optional[Path] = None,
    tail_lines:    int = 2000,
) -> list[CollectedError]:
    """
    Parse failed rules from main log -> harvest slurm log -> Tier-1 classification.
    Returns a list of CollectedError. No LLM calls -- call get_llm_summary()
    separately on the result if you want the optional LLM escalation.
    """
    # 1. Find the latest Snakemake main log
    main_log = _find_latest_log(log_dir)
    if not main_log:
        print(_c(f"  ✗ No Snakemake log in {log_dir}", RED), file=sys.stderr)
        return []

    print(_c(f"\n  📋 Main log:   {main_log.name}", DIM))
    print(_c(f"  🗂  Slurm logs: {describe_slurm_log_dir(slurm_log_dir)}", DIM))

    text  = main_log.read_text(errors="replace")
    metas = parse_main_log(text)

    if not metas:
        print(_c("  ✗ No failed rules found in main log.", RED), file=sys.stderr)
        return []

    print(_c(f"\n  {len(metas)} failed job(s): "
             f"{', '.join(m.rule_name for m in metas)}\n", CYAN))

    errors: list[CollectedError] = []

    for meta in metas:
        # 2. Harvest the slurm log (priority chain P1 -> P2 -> P3)
        sl = resolve_rule_log(
            rule_name      = meta.rule_name,
            slurm_log_dir  = slurm_log_dir,
            declared_logs  = meta.declared_logs,
            cli_log        = cli_log if len(metas) == 1 else None,
            tail_lines     = tail_lines,
            external_jobid = meta.external_jobid,
        )

        # wildcards: prefer value from main log; fall back to path inferred from slurm_logs
        wildcards = meta.wildcards or sl.wildcards

        if sl.found:
            src_label = _c(f"({sl.source})", GREEN)
            wc_label  = f"  wildcards={wildcards}" if wildcards else ""
            print(f"  ✓  {_c(meta.rule_name, BOLD)}  log found {src_label}  "
                  f"{_c(sl.display_path(), DIM)}{wc_label}")
        else:
            print(_c(f"  ⚠  {meta.rule_name}  no log found "
                     f"(tried slurm_logs + declared paths)", YELLOW))

        # 3. Tier-1 rule-based classifier — match against the windowed
        # error_context (lines around error-signal tokens), not the raw
        # tail: Snakemake often appends retry/shutdown boilerplate after a
        # job's real stderr, which can push the actual error out of a plain
        # tail even when --tail is generous.
        diag = quick_diagnose(
            rule_log        = sl.error_context,
            snakemake_block = meta.block_text,
        )

        if diag:
            print(f"     {_c('→', GREEN)} {_c(diag.error_type, YELLOW)}  "
                  f"confidence {_c(f'{diag.confidence*100:.0f}%', GREEN)}  "
                  f"{_c(diag.matched_pattern, DIM)}")
        else:
            print(f"     {_c('→', YELLOW)} unrecognised error"
                  + (_c("  (no slurm log content to match against)", DIM)
                     if not sl.content.strip() else ""))

        errors.append(CollectedError(
            rule_name    = meta.rule_name,
            wildcards    = wildcards,
            jobid        = meta.jobid,
            slurm_result = sl,
            diagnosis    = diag,
            input_files  = meta.input_files,
            output_files = meta.output_files,
        ))

    return errors


# ─────────────────────────────────────────────────────────────────
# Step 4: Render and save
# ─────────────────────────────────────────────────────────────────

# Job-preamble lines that vary between jobs but carry no diagnostic signal
# (dict key ordering, resource values, the benign "no nv files" GPU warning
# that Singularity prints on every job regardless of success). Left in, these
# push the real error content past the signature window and make identical
# failures fingerprint differently per sample.
_NOISE_LINE_RE = re.compile(
    r"^\s*(threads:|resources:|Shell command:|Activating singularity image"
    r"|WARNING: Could not find any nv files)",
    re.IGNORECASE,
)


def _normalize_signature_text(text: str, wildcards: str) -> str:
    """Strip sample-specific tokens so identical failures fingerprint the
    same regardless of which sample/path they came from."""
    norm = text
    # Sample/wildcard values (e.g. "SKNAS_MYCN_4h") often appear as bare
    # tokens, not inside a path, so substitute them explicitly before the
    # generic path/digit normalization below.
    for val in re.findall(r"=\s*([^,]+)", wildcards):
        val = val.strip()
        if val:
            norm = norm.replace(val, "<wc>")
    norm = re.sub(r"/\S+", "<path>", norm)
    norm = re.sub(r"\d+", "#", norm)
    return norm


def _error_signature(err: CollectedError) -> str:
    """Fingerprint used to cluster duplicate failures (e.g. the same rule
    failing identically across many samples) so the report shows one root
    cause instead of N near-identical blocks.

    Rule-based hits are keyed on the *actual matched log line*, not just the
    pattern name — a broad pattern (e.g. a generic "SQANTI3 error") can match
    genuinely different underlying failures, and those must stay separate
    rather than being reported as one repeated error.
    """
    if err.diagnosis:
        norm = _normalize_signature_text(err.diagnosis.matched_text, err.wildcards)
        return f"{err.rule_name}::rule-based::{err.diagnosis.matched_pattern}::{norm}"
    ctx = err.slurm_result.error_context or ""
    kept = [ln for ln in ctx.splitlines() if not _NOISE_LINE_RE.match(ln)]
    norm = _normalize_signature_text("\n".join(kept), err.wildcards)
    return f"{err.rule_name}::unrecognised::{norm}"


def _group_errors(errors: list[CollectedError]) -> list[list[CollectedError]]:
    groups: dict[str, list[CollectedError]] = {}
    order: list[str] = []
    for err in errors:
        sig = _error_signature(err)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(err)
    return [groups[sig] for sig in order]


_LLM_SYSTEM_PROMPT = (
    "You are a bioinformatics/HPC engineer helping debug a failed Snakemake "
    "cluster run. Below are one or more DISTINCT failed jobs (already "
    "deduplicated -- each one is a genuinely different failure, not a repeat "
    "of another). For each one, in plain prose (no markdown, no JSON, no "
    "rigid template):\n"
    "  - say what actually went wrong, concretely\n"
    "  - give the single most likely fix\n"
    "  - ALWAYS repeat the exact log file path given for that error, so the "
    "reader can open that exact file -- this is the most important part, "
    "never omit it or paraphrase the path\n"
    "If a fast classifier's guess is given for an error, don't just restate "
    "it -- add the detail that makes the fix actionable. Be concise and "
    "concrete; skip generic advice."
)


def _build_run_prompt(groups: list[list[CollectedError]]) -> str:
    parts = []
    for i, group in enumerate(groups, 1):
        rep = group[0]
        n = len(group)
        header = f"[Error {i}] Rule: {rep.rule_name}"
        if n > 1:
            header += f" ({n} samples affected)"
        parts.append(header)
        if rep.wildcards:
            parts.append(f"Wildcards: {rep.wildcards}")
        log_path = rep.slurm_result.display_path() if rep.slurm_result.log_path else "(no log file found)"
        parts.append(f"Log file path: {log_path}")
        if rep.diagnosis:
            parts.append(
                f"A fast classifier already labelled this: {rep.diagnosis.error_type} "
                f"— {rep.diagnosis.root_cause}"
            )
        ctx = rep.slurm_result.error_context or "(no log content captured)"
        parts.append(f"Log excerpt:\n{ctx}")
    return "\n\n".join(parts)


def get_llm_summary(
    errors: list[CollectedError],
    backend: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """One combined LLM call across all distinct error groups (not once per
    sample, not once per hardcoded rules.py pattern) -- returns its free-form
    summary, or None if there's nothing to summarize or the call fails."""
    groups = _group_errors(errors)
    if not groups:
        return None
    try:
        resolved_backend, resolved_model, api_key = llm_backends.resolve(backend, model)
    except RuntimeError as e:
        print(_c(f"\n  ✗ --llm: {e}", RED), file=sys.stderr)
        return None

    print(_c(f"\n  ⚡ Calling {resolved_backend}/{resolved_model} for a summary "
             f"of {len(groups)} distinct error(s)…", CYAN))
    try:
        return llm_backends.call_llm(
            _build_run_prompt(groups), _LLM_SYSTEM_PROMPT,
            resolved_backend, resolved_model, api_key, max_tokens=200 * len(groups),
        )
    except Exception as e:
        print(_c(f"  ⚠ LLM error: {e}", YELLOW))
        return None


def render(errors: list[CollectedError], llm_summary: Optional[str] = None) -> None:
    """Full coloured terminal report, clustered by error signature. Pass the
    string from get_llm_summary() to print it as a prominent block up top."""
    n_ok  = sum(1 for e in errors if e.recognised)
    n_unk = len(errors) - n_ok
    groups = _group_errors(errors)

    print()
    print(_c("━" * 66, BOLD))
    print(_c("  🔬  snakemake-ai-debugger  —  Error Collector", BOLD, CYAN))
    print(_c("━" * 66, BOLD))
    print(f"  {len(errors)} failed job(s) in {len(groups)} distinct error(s)  ·  "
          f"{_c(str(n_ok)+' recognised', GREEN)}  ·  "
          f"{_c(str(n_unk)+' unrecognised', YELLOW if n_unk else DIM)}")

    if llm_summary:
        print()
        print(_c("─" * 66, BOLD))
        print(_c("  LLM Summary", BOLD, CYAN))
        print(_c("─" * 66, BOLD))
        for para in llm_summary.strip().splitlines():
            if para.strip():
                print(textwrap.fill(para, width=74, initial_indent="  ", subsequent_indent="  "))
            else:
                print()

    for i, group in enumerate(groups, 1):
        _render_group(group, i, len(groups))

    print()
    print(_c("━" * 66, DIM))
    if not llm_summary and any(not e.recognised for e in errors):
        print(_c("  Tip: re-run with --llm for a plain-language summary of "
                 "unrecognised errors (needs a configured backend — see "
                 "llm_backends.py)", DIM))
    print()


def _render_group(group: list[CollectedError], idx: int, total: int) -> None:
    rep = group[0]
    n = len(group)

    title = f"  [{idx}/{total}]  {rep.rule_name}"
    if n > 1:
        title += f"  {_c(f'× {n} samples', BOLD, MAGENTA)}"
    elif rep.wildcards:
        title += f"  {_c('(' + rep.wildcards + ')', DIM)}"

    print()
    print(_c("─" * 66, BOLD))
    print(_c(title, BOLD, RED if not rep.recognised else BOLD))
    if rep.slurm_result.log_path:
        print(f"  {_c('Log (representative):', DIM)} {_c(rep.slurm_result.display_path(), DIM)}")
    if n > 1:
        samples = [e.wildcards or e.jobid for e in group]
        shown = ", ".join(s for s in samples[:8] if s)
        more = f"  … +{n - 8} more" if n > 8 else ""
        print(f"  {_c('Affected:', DIM)} {shown}{more}")
    print(_c("─" * 66, BOLD))

    if rep.diagnosis:
        d = rep.diagnosis
        conf_color = GREEN if d.confidence >= 0.7 else YELLOW
        print(f"  {_c('Error type', BOLD)}   {_c(d.error_type, YELLOW)}")
        print(f"  {_c('Confidence', BOLD)}   {_c(f'{d.confidence*100:.0f}%', conf_color)}")
        if d.matched_text:
            print(f"  {_c('Evidence', BOLD)}     {_c(d.matched_text, DIM)}")
        print()
        print(f"  {_c('Root Cause', BOLD)}")
        print(textwrap.fill(d.root_cause, width=74,
                            initial_indent="    ", subsequent_indent="    "))
        print()
        print(f"  {_c('Matched:', DIM)} {_c(d.matched_pattern, DIM)}")

    else:
        print(f"  {_c('Error type', BOLD)}   {_c('Unrecognised', YELLOW)}")
        print()
        ctx = rep.slurm_result.error_context
        if ctx:
            _MAX_CTX_LINES = 15
            ctx_lines = ctx.splitlines()
            truncated = len(ctx_lines) > _MAX_CTX_LINES
            print(f"  {_c('Error context (from log):', BOLD)}")
            for line in ctx_lines[:_MAX_CTX_LINES]:
                if line == "[...]":
                    print(f"    {_c('  ·  ·  ·', DIM)}")
                else:
                    print(f"    {_c(line, DIM)}")
            if truncated:
                print(f"    {_c(f'  … ({len(ctx_lines) - _MAX_CTX_LINES} more lines — see log file)', DIM)}")
        else:
            print(_c("  (no slurm log content available)", DIM))


def save_report(errors: list[CollectedError], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "collected_at": datetime.now().isoformat(),
        "failed_jobs":  [e.to_dict() for e in errors],
    }
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    print(_c(f"  📄 Report → {path}", DIM))


# ─────────────────────────────────────────────────────────────────
# onerror hook — drop-in for Snakefile
# ─────────────────────────────────────────────────────────────────

def on_error_hook(
    snakemake_log,                            # str | Path | list[str] — Snakemake version-dependent
    slurm_log_dir: str = ".snakemake/slurm_logs",
    report_dir:    str = ".snakemake/ai_debug",
) -> None:
    """
    Usage in Snakefile (recommended):

        from snakemake_ai_debugger.collect import on_error_hook

        onerror:
            on_error_hook(log)

    `log` is the main log path passed by Snakemake to the onerror handler.
    Older Snakemake versions pass a str; newer ones may pass a list — both handled.
    The YAML report is written to .snakemake/ai_debug/YYYYMMDD_HHMMSS.yaml
    """
    # Snakemake ≥7 may pass log as a list; take the first entry
    if isinstance(snakemake_log, (list, tuple)):
        snakemake_log = snakemake_log[0] if snakemake_log else None
    log_dir      = Path(snakemake_log).parent if snakemake_log else Path(".snakemake/log")
    slurm_dir    = Path(slurm_log_dir)
    errors = collect(log_dir=log_dir, slurm_log_dir=slurm_dir)

    if errors:
        # Normal mode stays LLM-free by default even in the hook; opt in
        # explicitly (same pattern as SNAKEMAKE_AI_QUIET) since the hook has
        # no CLI flags.
        llm_summary = None
        if os.environ.get("SNAKEMAKE_AI_LLM") == "1":
            llm_summary = get_llm_summary(
                errors,
                backend=os.environ.get("SNAKEMAKE_AI_BACKEND") or None,
                model=os.environ.get("SNAKEMAKE_AI_MODEL") or None,
            )
        render(errors, llm_summary=llm_summary)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_report(errors, Path(report_dir) / f"{ts}.yaml")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="snakemake-collect",
        description=(
            "Collect and display Slurm job errors from a failed Snakemake run.\n"
            "No LLM — fast, free, works offline."
        ),
    )
    p.add_argument("--log-dir",       type=Path, default=Path(".snakemake/log"),
                   help="Snakemake run log directory  (default: .snakemake/log)")
    p.add_argument("--slurm-log-dir", type=Path, default=Path(".snakemake/slurm_logs"),
                   help="Slurm job log directory      (default: .snakemake/slurm_logs)")
    p.add_argument("--log",           type=Path, default=None,
                   help="Override: path to a specific log file (single-rule debugging)")
    p.add_argument("--output",        type=Path, default=None,
                   help="Write YAML report here  (default: auto-timestamped)")
    p.add_argument("--tail",          type=int,  default=2000,
                   help="Lines of slurm log to read  (default: 2000; matches "
                        "the collect() API default — a small tail can cut off "
                        "the real error when Snakemake appends retry/shutdown "
                        "boilerplate after a job's own stderr)")
    p.add_argument("--llm",           action="store_true",
                   help="Escalate each distinct error to an LLM for a concise, "
                        "dynamically-generated fix (off by default — normal mode "
                        "is Tier-1 only, no network calls). Requires a backend "
                        "configured via --backend/--model or "
                        ".snakemake_ai_debugger.yaml — see llm_backends.py.")
    p.add_argument("--backend",       type=str,  default=None,
                   choices=list(llm_backends.KNOWN_BACKENDS),
                   help="LLM backend for --llm (claude | openai | gemini). "
                        "No default — must be set here or in the config file.")
    p.add_argument("--model",         type=str,  default=None,
                   help="Model id for --llm (e.g. claude-opus-5, gpt-5, "
                        "gemini-2.5-pro). No default — must be set here or in "
                        "the config file.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args   = build_parser().parse_args(argv)
    errors = collect(
        log_dir       = args.log_dir,
        slurm_log_dir = args.slurm_log_dir,
        cli_log       = args.log,
        tail_lines    = args.tail,
    )
    if not errors:
        return

    llm_summary = None
    if args.llm:
        llm_summary = get_llm_summary(errors, backend=args.backend, model=args.model)

    render(errors, llm_summary=llm_summary)
    out = args.output or Path(f"ai_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    save_report(errors, out)


if __name__ == "__main__":
    main()
