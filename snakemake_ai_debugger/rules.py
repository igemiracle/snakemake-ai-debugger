"""
rules.py — Rule-based fast classifier
======================================
Tier-1 diagnosis: pattern match against known Snakemake / bioinformatics
errors.  Returns a result immediately — no LLM call needed.

If no rule matches, returns None → caller escalates to LLM (Tier-2).

Design principle:
  • Each rule matches against the *rule-level log* first (most signal,
    fewest tokens), then falls back to the Snakemake summary block.
  • `fix` is specific and actionable, not generic advice.
  • Keep this file easy to extend: just add a new RulePattern to PATTERNS.

PATTERNS here only covers tool-agnostic Snakemake/HPC/environment errors.
Pipeline- or lab-specific tool patterns belong in a separate pip-installable
rule pack registered via the "snakemake_ai_debugger.rule_packs" entry point
(see register_patterns() / _load_rule_packs() below) — that keeps internal
pipeline details out of this public package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuickDiagnosis:
    """Tier-1 result: produced without calling the LLM."""
    error_type:       str
    root_cause:       str
    fix_suggestions:  list[str]
    matched_pattern:  str          # which pattern fired (for debugging)
    confidence:       float = 1.0  # rule-based matches are high-confidence
    matched_text:     str = ""     # the actual log line that fired the regex —
                                    # lets two hits on the same pattern but with
                                    # genuinely different content stay distinguishable


@dataclass
class RulePattern:
    name:        str           # identifier shown in matched_pattern
    error_type:  str           # short type label
    # One of these must match (checked against rule log then snakemake block)
    patterns:    list[str]     # regex patterns (re.IGNORECASE)
    root_cause:  str           # human-readable explanation
    # Optional — Tier-1 is for fast, free error *identification*; actionable
    # fixes are better generated on demand by --llm than hand-maintained
    # per pattern, so new patterns don't need to write these.
    fixes:       list[str] = field(default_factory=list)
    confidence:  float = 1.0


# ─────────────────────────────────────────────────────────────────
# Pattern library — add new entries here as you encounter new errors
# ─────────────────────────────────────────────────────────────────
PATTERNS: list[RulePattern] = [

    # ── Snakemake infrastructure ────────────────────────────────
    RulePattern(
        name="missing_output",
        error_type="MissingOutputException",
        patterns=[
            r"MissingOutputException",
            r"Output files? not present after execution",
            r"output file .+ (was not created|not found)",
        ],
        root_cause=(
            "The rule completed (exit code 0) but one or more declared output "
            "files were never written. Common causes: the tool silently failed "
            "and wrote nothing, a path wildcard mismatch, or the command wrote "
            "to a different location."
        ),
        fixes=[
            "Check the rule log for silent errors: the tool may have exited 0 despite failure.",
            "Verify the output path matches exactly what the command writes (wildcards, suffixes).",
            "Add `[ -s {output} ] || (echo 'empty output' >&2; exit 1)` after the command.",
            "Run the shell command manually with the expanded paths to reproduce.",
        ],
    ),

    RulePattern(
        name="missing_input",
        error_type="MissingInputException",
        patterns=[
            r"MissingInputException",
            r"Missing input files",
            r"input file .+ (does not exist|not found|missing)",
        ],
        root_cause=(
            "A required input file does not exist when this rule tries to run. "
            "Either an upstream rule failed to produce it, the path is wrong, "
            "or the file was deleted/moved."
        ),
        fixes=[
            "Check whether the upstream rule that produces this file actually succeeded.",
            "Confirm the file path matches what the upstream rule declares as output.",
            "Run `snakemake --dryrun` to verify the DAG resolves correctly.",
        ],
    ),

    RulePattern(
        name="ambiguous_rules",
        error_type="AmbiguousRuleException",
        patterns=[r"AmbiguousRuleException", r"ambiguous rule"],
        root_cause="Two rules can both produce the same output file. Snakemake cannot decide which to use.",
        fixes=[
            "Add `ruleorder: rule_a > rule_b` to the Snakefile to break the tie.",
            "Make the output patterns more specific so only one rule matches.",
        ],
    ),

    RulePattern(
        name="wildcard_error",
        error_type="WildcardError",
        patterns=[r"WildcardError", r"wildcard .+ (not defined|cannot be determined)"],
        root_cause="A wildcard in the output/input cannot be resolved from the available targets.",
        fixes=[
            "Ensure all wildcards used in `input:` also appear in `output:` (or are constrained).",
            "Add `wildcard_constraints` to restrict wildcard values if they're ambiguous.",
            "Check rule_all inputs — the target path must contain concrete wildcard values.",
        ],
    ),

    # ── Resource / HPC ──────────────────────────────────────────
    RulePattern(
        name="oom_killed",
        error_type="OOMKilled",
        patterns=[
            r"oom.?kill",
            r"out.of.memory",
            r"Killed\b",                      # bare "Killed" from Linux OOM killer
            r"slurmstepd.*Exceeded step memory limit",
            r"DUE TO TIME LIMIT",             # time → often memory too
            r"cancelled.*due to node failure", # node death often = OOM
        ],
        root_cause=(
            "The job was killed by the OS or Slurm because it exceeded the "
            "memory (or time) limit."
        ),
        fixes=[
            "Increase `resources: mem_mb=` in the rule (check sacct MaxRSS for actual peak).",
            "If using Slurm: check `sacct -j <jobid> --format=MaxRSS,Elapsed` for actual usage.",
            "Split the input into smaller chunks and run in parallel.",
            "Check if the tool has a low-memory or streaming mode for large inputs.",
            "Reduce any batch-size / chunk-size parameter the tool exposes.",
        ],
        confidence=0.95,
    ),

    RulePattern(
        name="disk_quota",
        error_type="DiskQuotaExceeded",
        patterns=[
            r"(no space left on device|disk quota exceeded|errno 28)",
            r"OSError.*\[Errno 28\]",
            r"write.*failed.*no space",
        ],
        root_cause="The filesystem ran out of space (or hit a quota). The output file is likely incomplete.",
        fixes=[
            "Free space: remove intermediate files with `snakemake --delete-temp-output`.",
            "Check quota: `df -h <workdir>` and `du -sh .snakemake/` (logs can be large).",
            "Redirect temp files to a larger scratch partition via `resources: tmpdir=`.",
        ],
    ),

    # ── Tool / environment ──────────────────────────────────────
    RulePattern(
        name="command_not_found",
        error_type="CommandNotFound",
        patterns=[
            r"command not found",
            r"No such file or directory.*bin/",
            r"FileNotFoundError.*\.(sh|py|r|pl)\b",
            r"which: no \S+ in \(",
        ],
        root_cause="The executable called in the rule's shell command is not on PATH inside the execution environment.",
        fixes=[
            "Activate the correct conda environment or load the module: check `which <tool>`.",
            "If using Singularity/Apptainer: verify the tool is installed inside the container.",
            "Specify the full path to the executable in the shell command.",
            "Add the tool to the rule's `conda:` environment file.",
        ],
    ),

    RulePattern(
        name="permission_denied",
        error_type="PermissionError",
        patterns=[
            r"Permission denied",
            r"OSError.*\[Errno 13\]",
            r"cannot open.*for (writing|reading).*permission",
        ],
        root_cause="The process lacks read or write permission on an input/output path.",
        fixes=[
            "Check ownership: `ls -la <path>` — ensure your user owns the directory.",
            "Fix permissions: `chmod u+w <directory>`.",
            "If on a shared cluster: confirm the group permissions are set correctly.",
        ],
    ),

    RulePattern(
        name="conda_env_error",
        error_type="CondaEnvError",
        patterns=[
            r"PackagesNotFoundError",
            r"conda.*ResolvePackageNotFound",
            r"conda.*CondaError",
            r"solving environment.*failed",
            r"UnsatisfiableError",
        ],
        root_cause="Conda cannot resolve or install the packages specified in the rule's environment file.",
        fixes=[
            "Run `conda env create -f <env.yaml>` manually to see the full solver output.",
            "Check for version conflicts; try removing strict version pins.",
            "Add `--channel-priority flexible` or specify channels explicitly in the env file.",
            "Use `mamba` instead of `conda` for faster, more reliable solving.",
        ],
    ),

    RulePattern(
        name="singularity_error",
        error_type="ContainerError",
        patterns=[
            r"(singularity|apptainer).*(error|failed|cannot)",
            r"FATAL.*container",
            r"failed to (pull|build|exec).*(sif|sandbox)",
            r"container.*bind.*path.*does not exist",
        ],
        root_cause="Singularity/Apptainer failed to start or run the container. Often a bind-path or image pull issue.",
        fixes=[
            "Check bind paths: all `--bind` paths must exist on the host.",
            "Re-pull the image: `singularity pull <image>` to refresh a possibly corrupted .sif.",
            "Test interactively: `singularity shell --bind <path> <image.sif>`.",
            "Ensure SINGULARITY_CACHEDIR has enough space and write permission.",
        ],
    ),

    # ── Common bioinformatics tools ─────────────────────────────
    RulePattern(
        name="samtools_truncated_bam",
        error_type="ToolCrash:TruncatedBAM",
        patterns=[
            r"\[E::bgzf_read\]",
            r"truncated file",
            r"\[W::.*EOF marker is absent",
            r"samtools.*error.*reading",
        ],
        root_cause="samtools encountered a truncated or corrupted BAM/CRAM file. The upstream alignment likely failed mid-write.",
        fixes=[
            "Validate the BAM: `samtools quickcheck <file.bam>`.",
            "Re-run the upstream alignment rule from scratch (delete the bad BAM first).",
            "Check disk space at the time the BAM was written — truncation often = disk full.",
        ],
    ),

    RulePattern(
        name="python_import_error",
        error_type="PythonImportError",
        patterns=[
            r"ModuleNotFoundError",
            r"ImportError.*No module named",
            r"cannot import name .+ from",
        ],
        root_cause="A Python script in the rule cannot import a required module. The module is missing or installed in a different environment.",
        fixes=[
            "Check which Python is running: `which python` in the rule's environment.",
            "Install the missing package in the correct environment: `pip install <module>`.",
            "If using conda: add the package to the rule's `conda:` env file.",
        ],
    ),

    RulePattern(
        name="snakemake_syntax_error",
        error_type="SnakefileSyntaxError",
        patterns=[
            r"SyntaxError.*Snakefile",
            r"snakemake.*SyntaxError",
            r"IndentationError.*Snakefile",
        ],
        root_cause="The Snakefile itself has a Python syntax or indentation error. The pipeline cannot even be parsed.",
        fixes=[
            "Run `snakemake --lint` to get the exact line number.",
            "Check indentation around the failing rule — Snakemake is indent-sensitive.",
        ],
    ),

]


# ─────────────────────────────────────────────────────────────────
# Rule packs — extend PATTERNS without forking this file
# ─────────────────────────────────────────────────────────────────
# Anyone with tools not covered above (a lab's internal pipeline, a
# specific instrument, etc.) can ship their own patterns as a separate
# pip-installable package. Register them by exposing a
# "snakemake_ai_debugger.rule_packs" entry point that resolves to a
# zero-arg callable returning list[RulePattern] — no changes to this
# repo required. Custom patterns are checked before the built-ins so
# they can be more specific than a generic catch-all here.

def register_patterns(patterns: list[RulePattern]) -> None:
    """Prepend custom RulePatterns so they're checked before the built-ins."""
    PATTERNS[:0] = patterns


def _load_rule_packs() -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        eps = entry_points(group="snakemake_ai_debugger.rule_packs")
    except TypeError:  # Python <3.10 API shape
        eps = entry_points().get("snakemake_ai_debugger.rule_packs", [])
    for ep in eps:
        try:
            register_patterns(ep.load()())
        except Exception as e:
            import warnings
            warnings.warn(f"snakemake-ai-debugger: failed to load rule pack '{ep.name}': {e}")


_load_rule_packs()


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def quick_diagnose(rule_log: str, snakemake_block: str) -> Optional[QuickDiagnosis]:
    """
    Try each pattern against rule_log first, then snakemake_block.
    Returns the first match, or None if no pattern fires.

    Args:
        rule_log:        Contents of the rule-specific log file (stderr redirect).
                         Pass "" if not available.
        snakemake_block: The per-rule error block extracted from the Snakemake
                         run log (the lines between "Error in rule X:" and the
                         next blank line or rule).
    """
    search_texts = [t for t in (rule_log, snakemake_block) if t.strip()]

    for pat in PATTERNS:
        for text in search_texts:
            for regex in pat.patterns:
                m = re.search(regex, text, re.IGNORECASE)
                if m:
                    return QuickDiagnosis(
                        error_type=pat.error_type,
                        root_cause=pat.root_cause,
                        fix_suggestions=pat.fixes,
                        matched_pattern=f"{pat.name} · /{regex}/",
                        confidence=pat.confidence,
                        matched_text=_line_containing(text, m.start()),
                    )
    return None


def _line_containing(text: str, pos: int) -> str:
    """The single log line that contains character offset `pos`."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def list_patterns() -> None:
    """Pretty-print all known patterns (useful for `--list-patterns` CLI flag)."""
    for p in PATTERNS:
        print(f"  {p.name:30s}  {p.error_type}")
