"""
snakemake-ai-debugger · diagnose.py
=====================================
Two-tier diagnosis pipeline:

  Tier 1 — Rule-based (rules.py)
    • Extract every failed rule block from .snakemake/log  (main log = failure notification)
    • For each rule, harvest the actual error from slurm_logs (the real evidence)
    • Match against known error patterns → output immediately, no API call

  Tier 2 — LLM (Claude API)
    • Only for rules Tier 1 could not classify
    • Send minimal context: slurm log tail + Snakemake error block
    • Never send the full Snakemake run log

Log resolution priority (log_collector.py):
  P1  .snakemake/slurm_logs/<rule>/**/<job>.log   ← cluster reality
  P2  Snakefile log: field                         ← local runs
  P3  --log CLI arg                                ← manual override

Output: coloured terminal summary + timestamped YAML report.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .rules import quick_diagnose
from .log_collector import resolve_rule_log, describe_slurm_log_dir, SlurmLogResult

# ── ANSI colours ──────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED   = "\033[31m"; YELLOW = "\033[33m"; GREEN = "\033[32m"; CYAN = "\033[36m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


# ─────────────────────────────────────────────────────────────────
# Step 1: Parse the Snakemake main log -- extract rule names and metadata only
# ─────────────────────────────────────────────────────────────────

@dataclass
class FailedRuleBlock:
    """Metadata extracted from the Snakemake main log for one failed rule."""
    rule_name:    str
    block_text:   str         # the "Error in rule X:" section -- context only
    log_files:    list[Path] = field(default_factory=list)   # Snakefile log: paths
    input_files:  list[str]  = field(default_factory=list)
    output_files: list[str]  = field(default_factory=list)
    jobid:        str = ""
    # Populated later by log_collector
    slurm_log:    Optional[SlurmLogResult] = field(default=None, repr=False)


def parse_failed_blocks(snakemake_log_text: str) -> list[FailedRuleBlock]:
    """
    Extract one FailedRuleBlock per 'Error in rule X:' section.
    The main log is only used for metadata — actual error content
    comes from slurm_logs in the next step.
    """
    blocks: list[FailedRuleBlock] = []
    sections = re.split(r"(?=\[.*\]\nError in rule |\nError in rule )", snakemake_log_text)

    for section in sections:
        m = re.search(r"Error in rule (\S+?):", section)
        if not m:
            continue
        rule_name = m.group(1)

        # Capture only the lines between "Error in rule" and the next timestamp
        block_lines: list[str] = []
        in_block = False
        for line in section.splitlines():
            if "Error in rule" in line:
                in_block = True
            if in_block:
                if re.match(r"\[.{15,30}\]", line) and block_lines:
                    break
                block_lines.append(line)
        block_text = "\n".join(block_lines)

        # Snakefile log: paths (P2 fallback -- used when slurm_logs has nothing)
        log_files: list[Path] = []
        for line in block_text.splitlines():
            lm = re.match(r"\s+log:\s+(.+)", line)
            if lm:
                for p in lm.group(1).split(","):
                    p = p.strip()
                    if p:
                        log_files.append(Path(p))

        input_files  = _extract_file_list(block_text, "input")
        output_files = _extract_file_list(block_text, "output")
        jm = re.search(r"jobid:\s*(\d+)", block_text)

        blocks.append(FailedRuleBlock(
            rule_name=rule_name,
            block_text=block_text,
            log_files=log_files,
            input_files=input_files,
            output_files=output_files,
            jobid=jm.group(1) if jm else "",
        ))

    return blocks


def _extract_file_list(text: str, keyword: str) -> list[str]:
    files: list[str] = []
    collecting = False
    for line in text.splitlines():
        if re.match(rf"\s+{keyword}:", line):
            collecting = True
            rest = re.sub(rf"\s+{keyword}:\s*", "", line)
            files += [f.strip() for f in rest.split(",") if f.strip()]
            continue
        if collecting:
            if re.match(r"\s+(output|input|log|jobid|conda|singularity|threads|resources):", line):
                break
            files += [f.strip() for f in line.split(",") if f.strip()]
    return files


def _find_latest_snakemake_log(log_dir: Path) -> Optional[Path]:
    try:
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[0] if logs else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Step 2: LLM prompt — built from slurm log, not main log
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert bioinformatics engineer specialising in Snakemake,
    HPC/Slurm, Singularity containers, and long-read sequencing tools
    (ONT Dorado, IsoQuant, minimap2, samtools, gffcompare).

    A rule-based classifier already screened this error and could NOT identify it.
    Analyse the context and return ONLY a valid YAML document — no markdown
    fences, no prose outside the YAML.

    Schema:
      error_type:       <str>   # short label
      root_cause:       <str>   # 1-3 sentences, precise
      evidence:
        - source: <str>         # e.g. "slurm log line 12"
          detail: <str>
      fix_suggestions:
        - <str>
      confidence:       <float> # 0.0 – 1.0
      follow_up_checks:
        - <str>
""").strip()


def _build_llm_prompt(
    block: FailedRuleBlock,
    slurm_result: SlurmLogResult,
    snakefile_snippet: str,
) -> str:
    """
    Minimal, token-conscious prompt.

    Priority of information:
      1. Slurm log tail    ← the actual error output (most signal)
      2. Snakemake block   ← metadata (exception type, file paths)
      3. Snakefile snippet ← rule definition (shell/script command)
      4. File lists        ← input/output names only, no content
    """
    parts: list[str] = [f"## Failed rule: {block.rule_name}"]

    # The slurm log is the primary evidence — send error context, not the raw full log
    if slurm_result.found:
        src_label = slurm_result.display_path()
        if slurm_result.wildcards:
            src_label += f"  [wildcards: {slurm_result.wildcards}]"
        parts.append(f"## Slurm job log — {src_label}\n{slurm_result.error_context}")
    elif slurm_result.source == "none":
        parts.append("## Slurm log: not found (local run or log missing)")

    # Snakemake error block — adds exception type and declared paths
    if block.block_text:
        parts.append(f"## Snakemake error block\n{block.block_text}")

    # Snakefile rule definition
    if snakefile_snippet:
        parts.append(f"## Snakefile rule definition\n{snakefile_snippet}")

    # Just the file names — never file contents
    if block.input_files:
        parts.append("## Declared inputs\n" + "\n".join(block.input_files))
    if block.output_files:
        parts.append("## Declared outputs\n" + "\n".join(block.output_files))

    return "\n\n".join(parts)


def _snakefile_rule_snippet(snakefile: Optional[Path], rule_name: str, context: int = 25) -> str:
    if not snakefile or not snakefile.exists():
        return ""
    lines = snakefile.read_text(errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(rf"^rule\s+{re.escape(rule_name)}\s*:", line):
            start = i
            break
    return "\n".join(lines[max(0, start - 1): start + context])


# ─────────────────────────────────────────────────────────────────
# Step 3: LLM call
# ─────────────────────────────────────────────────────────────────

def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Run: pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _parse_yaml(text: str) -> Optional[dict]:
    text = re.sub(r"^```ya?ml\s*", "", text.strip(), flags=re.M)
    text = re.sub(r"^```\s*$", "", text, flags=re.M)
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Step 4: Result + rendering
# ─────────────────────────────────────────────────────────────────

@dataclass
class RuleDiagnosis:
    rule_name:        str
    tier:             str           # "rule-based" | "llm" | "undiagnosed"
    error_type:       str = ""
    root_cause:       str = ""
    fix_suggestions:  list[str] = field(default_factory=list)
    evidence:         list[dict] = field(default_factory=list)
    follow_up_checks: list[str] = field(default_factory=list)
    confidence:       float = 0.0
    matched_pattern:  str = ""
    wildcards:        str = ""      # from slurm log path
    slurm_log_path:   str = ""      # for reference in YAML
    raw_llm:          str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "rule_name":       self.rule_name,
            "tier":            self.tier,
            "error_type":      self.error_type,
            "root_cause":      self.root_cause,
            "fix_suggestions": self.fix_suggestions,
            "confidence":      self.confidence,
        }
        if self.wildcards:
            d["wildcards"] = self.wildcards
        if self.slurm_log_path:
            d["slurm_log"] = self.slurm_log_path
        if self.evidence:
            d["evidence"] = self.evidence
        if self.follow_up_checks:
            d["follow_up_checks"] = self.follow_up_checks
        if self.matched_pattern:
            d["matched_pattern"] = self.matched_pattern
        return d


def render_single(d: RuleDiagnosis, idx: int, total: int) -> None:
    header = f"  Rule {idx}/{total}: {d.rule_name}"
    if d.wildcards:
        header += f"  {_c('['+d.wildcards+']', DIM)}"
    tier_label = _c(f"[{d.tier}]", GREEN if d.tier == "rule-based" else CYAN)
    conf = d.confidence
    conf_str   = f"{conf * 100:.0f}%"
    conf_color = GREEN if conf >= 0.7 else YELLOW if conf >= 0.4 else RED

    print()
    print(_c("─" * 64, BOLD))
    print(f"{_c(header, BOLD, RED)}  {tier_label}  {_c(conf_str, conf_color)}")
    if d.slurm_log_path:
        print(f"  {_c('Log:', DIM)} {_c(d.slurm_log_path, DIM)}")
    print(_c("─" * 64, BOLD))

    print(f"  {_c('Error type', BOLD)}  {_c(d.error_type, YELLOW)}")

    if d.root_cause:
        print(f"\n  {_c('Root Cause', BOLD)}")
        print(textwrap.fill(d.root_cause, width=74,
                            initial_indent="    ", subsequent_indent="    "))

    if d.evidence:
        print(f"\n  {_c('Evidence', BOLD)}")
        for ev in d.evidence:
            src = ev.get("source", ""); det = ev.get("detail", "")
            print(f"    {_c('▸', CYAN)} {_c(src, DIM)}  {det}")

    if d.fix_suggestions:
        print(f"\n  {_c('Fix Suggestions', BOLD)}")
        for i, f in enumerate(d.fix_suggestions, 1):
            line = textwrap.fill(f, width=70, subsequent_indent="       ")
            print(f"    {_c(str(i)+'.', GREEN, BOLD)} {line}")

    if d.follow_up_checks:
        print(f"\n  {_c('Follow-up', BOLD)}")
        for c in d.follow_up_checks:
            print(f"    {_c('○', DIM)} {c}")

    if d.matched_pattern:
        print(f"\n  {_c('Pattern:', DIM)} {_c(d.matched_pattern, DIM)}")


def render_summary(results: list[RuleDiagnosis]) -> None:
    n_rule = sum(1 for r in results if r.tier == "rule-based")
    n_llm  = sum(1 for r in results if r.tier == "llm")
    n_unk  = sum(1 for r in results if r.tier == "undiagnosed")
    print()
    print(_c("━" * 64, BOLD))
    print(_c("  🔬  Snakemake AI Debugger", BOLD, CYAN))
    print(_c("━" * 64, BOLD))
    print(f"  {len(results)} failed rule(s)  ·  "
          f"{_c(str(n_rule)+' rule-based', GREEN)}  ·  "
          f"{_c(str(n_llm)+' LLM', CYAN)}  ·  "
          f"{_c(str(n_unk)+' undiagnosed', YELLOW if n_unk else DIM)}")
    for i, d in enumerate(results, 1):
        render_single(d, i, len(results))
    print()
    print(_c("━" * 64, DIM))
    print()


def save_yaml(results: list[RuleDiagnosis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now().isoformat(),
        "diagnoses": [r.to_dict() for r in results],
    }
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    print(_c(f"  📄 YAML → {path}", DIM))


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def diagnose(
    snakefile:      Optional[Path] = None,
    log_dir:        Path           = Path(".snakemake/log"),
    slurm_log_dir:  Path           = Path(".snakemake/slurm_logs"),
    cli_log:        Optional[Path] = None,
    model:          str            = "claude-sonnet-4-20250514",
    output:         Optional[Path] = None,
    dry_run:        bool           = False,
) -> list[RuleDiagnosis]:

    # -- 1. Parse main Snakemake log (failure notification only)
    smk_log_path = _find_latest_snakemake_log(log_dir)
    if not smk_log_path:
        print(_c(f"  ✗ No Snakemake log found in {log_dir}", RED), file=sys.stderr)
        return []

    print(_c(f"\n  📋 Main log:   {smk_log_path.name}", DIM))
    print(_c(f"  🗂  Slurm logs: {describe_slurm_log_dir(slurm_log_dir)}", DIM))

    smk_text = smk_log_path.read_text(errors="replace")
    blocks   = parse_failed_blocks(smk_text)
    if not blocks:
        print(_c("  ✗ No failed rule blocks found in main log.", RED), file=sys.stderr)
        return []

    print(_c(f"\n  Found {len(blocks)} failed rule(s): "
             f"{', '.join(b.rule_name for b in blocks)}", CYAN))

    # -- 2. Harvest slurm logs (the real error scene)
    for block in blocks:
        block.slurm_log = resolve_rule_log(
            rule_name     = block.rule_name,
            slurm_log_dir = slurm_log_dir,
            declared_logs = block.log_files,
            cli_log       = cli_log if len(blocks) == 1 else None,
        )
        src = block.slurm_log.source
        if block.slurm_log.found:
            label = _c(f"({src})", GREEN)
            path  = block.slurm_log.display_path()
            wc    = f"  wildcards={block.slurm_log.wildcards}" if block.slurm_log.wildcards else ""
            print(f"  ✓ [{block.rule_name}] log found {label}  {path}{wc}")
        else:
            print(_c(f"  ⚠ [{block.rule_name}] no log found in slurm_logs or declared paths", YELLOW))

    # ── 3. Tier-1: rule-based classification ──────────────────
    results: list[RuleDiagnosis] = []
    llm_queue: list[tuple] = []   # (block, slurm_result, snippet)

    for block in blocks:
        sl = block.slurm_log
        # Feed the slurm log content to Tier-1 first, then the main-log block
        quick = quick_diagnose(
            rule_log        = sl.content if sl else "",
            snakemake_block = block.block_text,
        )

        base = dict(
            rule_name      = block.rule_name,
            wildcards      = sl.wildcards if sl else "",
            slurm_log_path = sl.display_path() if (sl and sl.log_path) else "",
        )

        if quick:
            print(_c(f"  ✓ [{block.rule_name}] Tier-1 matched: {quick.matched_pattern}", GREEN))
            results.append(RuleDiagnosis(
                **base,
                tier            = "rule-based",
                error_type      = quick.error_type,
                root_cause      = quick.root_cause,
                fix_suggestions = quick.fix_suggestions,
                confidence      = quick.confidence,
                matched_pattern = quick.matched_pattern,
            ))
        else:
            print(_c(f"  ? [{block.rule_name}] no pattern match → queuing for LLM", YELLOW))
            snippet = _snakefile_rule_snippet(snakefile, block.rule_name)
            llm_queue.append((block, sl or SlurmLogResult(
                rule_name=block.rule_name, log_path=None,
                wildcards="", job_id="", content="", source="none",
            ), snippet, base))

    # ── 4. Tier-2: LLM for unclassified rules ─────────────────
    if llm_queue:
        if dry_run:
            for block, sl, snippet, base in llm_queue:
                prompt = _build_llm_prompt(block, sl, snippet)
                print(_c(f"\n── DRY RUN prompt for [{block.rule_name}] ──\n", BOLD))
                print(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt}")
                print(_c(f"\n  Token estimate: ~{len(prompt.split())} words", DIM))
            for _, _, _, base in llm_queue:
                results.append(RuleDiagnosis(**base, tier="undiagnosed",
                                             error_type="Unknown", confidence=0.0))
        else:
            print(_c(f"\n  ⚡ Calling Claude for {len(llm_queue)} rule(s)…", CYAN))
            for block, sl, snippet, base in llm_queue:
                prompt = _build_llm_prompt(block, sl, snippet)
                try:
                    raw    = call_claude(prompt, model=model)
                    parsed = _parse_yaml(raw)
                    if parsed:
                        results.append(RuleDiagnosis(
                            **base,
                            tier            = "llm",
                            error_type      = parsed.get("error_type", "Unknown"),
                            root_cause      = parsed.get("root_cause", ""),
                            fix_suggestions = parsed.get("fix_suggestions", []),
                            evidence        = parsed.get("evidence", []),
                            follow_up_checks= parsed.get("follow_up_checks", []),
                            confidence      = float(parsed.get("confidence", 0.5)),
                            raw_llm         = raw,
                        ))
                        print(_c(f"  ✓ [{block.rule_name}] LLM diagnosis done", GREEN))
                    else:
                        print(_c(f"  ⚠ [{block.rule_name}] LLM YAML parse failed", YELLOW))
                        results.append(RuleDiagnosis(**base, tier="undiagnosed",
                                                     error_type="Unknown (parse error)",
                                                     root_cause=raw[:300], confidence=0.0))
                except Exception as e:
                    print(_c(f"  ✗ [{block.rule_name}] LLM error: {e}", RED))
                    results.append(RuleDiagnosis(**base, tier="undiagnosed",
                                                 error_type="Unknown (LLM error)", confidence=0.0))

    # ── 5. Output ──────────────────────────────────────────────
    render_summary(results)
    out = output or Path(f"ai_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    save_yaml(results, out)
    return results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="snakemake-ai-debugger",
        description="Two-tier AI diagnosis for failed Snakemake cluster jobs.",
    )
    p.add_argument("--snakefile",     type=Path, default=Path("Snakefile"))
    p.add_argument("--log-dir",       type=Path, default=Path(".snakemake/log"),
                   help="Directory with Snakemake run logs (default: .snakemake/log)")
    p.add_argument("--slurm-log-dir", type=Path, default=Path(".snakemake/slurm_logs"),
                   help="Slurm log directory (default: .snakemake/slurm_logs)")
    p.add_argument("--log",           type=Path, default=None,
                   help="Override: path to a specific log file (single-rule debugging)")
    p.add_argument("--model",         type=str,  default="claude-sonnet-4-6")
    p.add_argument("--output",        type=Path, default=None)
    p.add_argument("--dry-run",       action="store_true",
                   help="Show collected prompts without calling the API")
    p.add_argument("--list-patterns", action="store_true",
                   help="Print all built-in Tier-1 patterns and exit")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_patterns:
        from .rules import list_patterns
        print(_c("\n  Built-in Tier-1 patterns:\n", BOLD))
        list_patterns()
        return
    diagnose(
        snakefile     = args.snakefile,
        log_dir       = args.log_dir,
        slurm_log_dir = args.slurm_log_dir,
        cli_log       = args.log,
        model         = args.model,
        output        = args.output,
        dry_run       = args.dry_run,
    )


if __name__ == "__main__":
    main()
