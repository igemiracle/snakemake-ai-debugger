"""
Tests covering the two-tier pipeline end-to-end (no LLM calls).
"""
import textwrap
from pathlib import Path
import pytest
from snakemake_ai_debugger.diagnose import (
    parse_failed_blocks,
    _extract_file_list,
    _snakefile_rule_snippet,
    _build_llm_prompt,
    render_summary,
    RuleDiagnosis,
)
from snakemake_ai_debugger.rules import quick_diagnose, QuickDiagnosis

# ── Fixtures ──────────────────────────────────────────────────────

SMK_LOG_ONE = textwrap.dedent("""\
    Building DAG of jobs...
    [Mon May 26 14:30:00 2025]
    rule extract_ids:
        input: data/sample.fastq
        output: temp/sample.bed
        log: logs/extract_ids.err
        jobid: 3

    [Mon May 26 14:30:05 2025]
    Error in rule extract_ids:
        jobid: 3
        input: data/sample.fastq
        output: temp/sample.bed
        log: logs/extract_ids.err

    MissingOutputException: Output files not present after execution:
    temp/sample.bed
    Shutting down...
""")

SMK_LOG_TWO = textwrap.dedent("""\
    [Mon May 26 14:30:00 2025]
    Error in rule align_reads:
        jobid: 1
        input: data/reads.fastq
        output: bam/reads.bam
        log: logs/align.err

    CalledProcessError: Command returned non-zero exit status 137.
    Killed

    [Mon May 26 14:32:00 2025]
    Error in rule call_variants:
        jobid: 2
        input: bam/reads.bam
        output: vcf/variants.vcf
        log: logs/call.err

    MissingInputException: Missing input files for rule call_variants:
    bam/reads.bam
""")

SNAKEFILE = textwrap.dedent("""\
    rule all:
        input: "temp/sample.bed"

    rule extract_ids:
        input: "data/sample.fastq"
        output: "temp/sample.bed"
        log: "logs/extract_ids.err"
        shell:
            "awk '{{print $1}}' {input} > {output}"

    rule align_reads:
        input: "data/reads.fastq"
        output: "bam/reads.bam"
        threads: 8
        shell:
            "minimap2 -t {threads} ref.fa {input} | samtools sort > {output}"
""")

# ── parse_failed_blocks ───────────────────────────────────────────

def test_parse_single_block():
    blocks = parse_failed_blocks(SMK_LOG_ONE)
    assert len(blocks) == 1
    assert blocks[0].rule_name == "extract_ids"
    assert "MissingOutputException" in blocks[0].block_text

def test_parse_log_files():
    blocks = parse_failed_blocks(SMK_LOG_ONE)
    assert any("extract_ids.err" in str(p) for p in blocks[0].log_files)

def test_parse_input_output():
    blocks = parse_failed_blocks(SMK_LOG_ONE)
    b = blocks[0]
    assert any("sample.fastq" in f for f in b.input_files)
    assert any("sample.bed" in f for f in b.output_files)

def test_parse_two_blocks():
    blocks = parse_failed_blocks(SMK_LOG_TWO)
    assert len(blocks) == 2
    names = [b.rule_name for b in blocks]
    assert "align_reads" in names
    assert "call_variants" in names

def test_parse_empty_log():
    assert parse_failed_blocks("No errors here.") == []

# ── quick_diagnose (rules.py) ─────────────────────────────────────

def test_tier1_missing_output():
    result = quick_diagnose("", "MissingOutputException: temp/sample.bed not present")
    assert result is not None
    assert result.error_type == "MissingOutputException"
    assert result.confidence >= 0.9

def test_tier1_oom_killed():
    result = quick_diagnose("Killed\noom-kill event", "")
    assert result is not None
    assert result.error_type == "OOMKilled"

def test_tier1_oom_in_snakemake_block():
    result = quick_diagnose("", "CalledProcessError: exit status 137.\nKilled")
    assert result is not None
    assert result.error_type == "OOMKilled"

def test_tier1_missing_input():
    result = quick_diagnose("", "MissingInputException: Missing input files for rule foo")
    assert result is not None
    assert result.error_type == "MissingInputException"

def test_tier1_command_not_found():
    result = quick_diagnose("/bin/sh: minimap2: command not found", "")
    assert result is not None
    assert result.error_type == "CommandNotFound"

def test_tier1_conda_error():
    result = quick_diagnose("PackagesNotFoundError: The following packages are not available", "")
    assert result is not None
    assert result.error_type == "CondaEnvError"

def test_tier1_no_match_returns_none():
    result = quick_diagnose("everything looks fine", "no error here")
    assert result is None

def test_tier1_truncated_bam():
    result = quick_diagnose("[E::bgzf_read] truncated file", "")
    assert result is not None
    assert "TruncatedBAM" in result.error_type

# ── Snakefile snippet ─────────────────────────────────────────────

def test_snakefile_snippet(tmp_path):
    sf = tmp_path / "Snakefile"
    sf.write_text(SNAKEFILE)
    snippet = _snakefile_rule_snippet(sf, "extract_ids")
    assert "awk" in snippet
    assert "extract_ids" in snippet

def test_snakefile_snippet_missing_rule(tmp_path):
    sf = tmp_path / "Snakefile"
    sf.write_text(SNAKEFILE)
    snippet = _snakefile_rule_snippet(sf, "no_such_rule")
    assert isinstance(snippet, str)  # doesn't crash

def test_snakefile_snippet_no_file():
    snippet = _snakefile_rule_snippet(Path("/nonexistent/Snakefile"), "rule_x")
    assert snippet == ""

# ── LLM prompt minimality ─────────────────────────────────────────

def test_llm_prompt_excludes_full_log():
    from snakemake_ai_debugger.log_collector import SlurmLogResult
    blocks = parse_failed_blocks(SMK_LOG_TWO)
    block = blocks[0]   # align_reads
    empty_sl = SlurmLogResult(rule_name=block.rule_name, log_path=None,
                               wildcards="", job_id="", content="", source="none")
    prompt = _build_llm_prompt(block, slurm_result=empty_sl, snakefile_snippet="")
    assert "call_variants" not in prompt

def test_llm_prompt_contains_block_text():
    from snakemake_ai_debugger.log_collector import SlurmLogResult
    blocks = parse_failed_blocks(SMK_LOG_ONE)
    block = blocks[0]
    sl = SlurmLogResult(rule_name=block.rule_name, log_path=Path("/fake/path.log"),
                        wildcards="", job_id="", content="some stderr here", source="slurm_logs")
    prompt = _build_llm_prompt(block, slurm_result=sl, snakefile_snippet="")
    assert "MissingOutputException" in prompt
    assert "some stderr here" in prompt

def test_llm_prompt_log_capped(tmp_path):
    """Slurm log is capped at 80 lines by log_collector — prompt stays manageable."""
    from snakemake_ai_debugger.log_collector import SlurmLogResult
    long_content = "\n".join([f"line {i}" for i in range(200)])
    blocks = parse_failed_blocks(SMK_LOG_ONE)
    sl = SlurmLogResult(rule_name="extract_ids", log_path=None,
                        wildcards="", job_id="", content=long_content, source="slurm_logs")
    prompt = _build_llm_prompt(blocks[0], slurm_result=sl, snakefile_snippet="")
    assert "extract_ids" in prompt

# ── render_summary (smoke test — just mustn't crash) ──────────────

def test_render_summary_smoke(capsys):
    results = [
        RuleDiagnosis(
            rule_name="extract_ids", tier="rule-based",
            error_type="MissingOutputException",
            root_cause="Output not written.",
            fix_suggestions=["Check the shell command."],
            confidence=0.95,
            matched_pattern="missing_output · /MissingOutputException/",
        ),
        RuleDiagnosis(
            rule_name="align_reads", tier="llm",
            error_type="OOMKilled",
            root_cause="Job killed by OOM killer.",
            fix_suggestions=["Increase mem_mb."],
            confidence=0.80,
        ),
    ]
    render_summary(results)
    out = capsys.readouterr().out
    assert "extract_ids" in out
    assert "align_reads" in out
    assert "rule-based" in out
    assert "llm" in out


# ── log_collector tests ───────────────────────────────────────────
from snakemake_ai_debugger.log_collector import (
    find_slurm_log,
    resolve_rule_log,
    list_slurm_rules,
    _extract_job_id,
)

def _make_slurm_log(tmp_path: Path, rule: str, wildcards: str, job_id: str, content: str) -> Path:
    """Helper: create a fake slurm log at the expected path."""
    if wildcards:
        log_dir = tmp_path / "slurm_logs" / rule / wildcards
    else:
        log_dir = tmp_path / "slurm_logs" / rule
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_id}.log"
    log_file.write_text(content)
    return log_file


def test_find_slurm_log_no_wildcards(tmp_path):
    _make_slurm_log(tmp_path, "build_gene_list", "", "98765", "slurmstepd: error: Exceeded step memory limit\nKilled\n")
    result = find_slurm_log("build_gene_list", tmp_path / "slurm_logs")
    assert result.found
    assert result.source == "slurm_logs"
    assert result.wildcards == ""
    assert result.job_id == "98765"
    assert "Exceeded step memory limit" in result.content


def test_find_slurm_log_with_wildcards(tmp_path):
    _make_slurm_log(tmp_path, "align_reads", "SRR123456", "11111", "CUDA error: no device\n")
    result = find_slurm_log("align_reads", tmp_path / "slurm_logs")
    assert result.found
    assert result.wildcards == "SRR123456"
    assert "CUDA error" in result.content


def test_find_slurm_log_nested_wildcards(tmp_path):
    """Deeply nested: slurm_logs/<rule>/<sample>/<condition>/<job>.log"""
    _make_slurm_log(tmp_path, "call_isoforms", "sample1/treated", "55555",
                    "IsoQuant: no alignments found\n")
    result = find_slurm_log("call_isoforms", tmp_path / "slurm_logs")
    assert result.found
    assert "sample1" in result.wildcards
    assert "treated" in result.wildcards


def test_find_slurm_log_picks_latest(tmp_path):
    """When multiple jobs exist for the same rule, picks the most recent."""
    import time
    _make_slurm_log(tmp_path, "trim_reads", "SRR001", "10001", "old error\n")
    time.sleep(0.02)
    _make_slurm_log(tmp_path, "trim_reads", "SRR001", "10002", "new error: command not found\n")
    result = find_slurm_log("trim_reads", tmp_path / "slurm_logs")
    assert "new error" in result.content
    assert "old error" not in result.content


def test_find_slurm_log_missing_rule(tmp_path):
    (tmp_path / "slurm_logs").mkdir()
    result = find_slurm_log("nonexistent_rule", tmp_path / "slurm_logs")
    assert not result.found
    assert result.source == "none"


def test_resolve_priority_p1_wins(tmp_path):
    """slurm_logs (P1) takes priority over declared log files (P2)."""
    _make_slurm_log(tmp_path, "my_rule", "", "42", "OOM Killed\n")
    # Create a P2 file that says something different
    p2 = tmp_path / "logs" / "my_rule.log"
    p2.parent.mkdir()
    p2.write_text("P2 content: this should not appear\n")
    result = resolve_rule_log("my_rule", tmp_path / "slurm_logs",
                              declared_logs=[p2], cli_log=None)
    assert result.source == "slurm_logs"
    assert "OOM" in result.content


def test_resolve_priority_p2_fallback(tmp_path):
    """Falls back to declared log file when slurm_logs has nothing."""
    (tmp_path / "slurm_logs").mkdir()
    p2 = tmp_path / "logs" / "my_rule.log"
    p2.parent.mkdir()
    p2.write_text("MissingOutputException: file.bed not found\n")
    result = resolve_rule_log("my_rule", tmp_path / "slurm_logs",
                              declared_logs=[p2], cli_log=None)
    assert result.source == "snakemake_log_field"
    assert "MissingOutputException" in result.content


def test_resolve_priority_p3_cli(tmp_path):
    """Falls back to CLI --log when both P1 and P2 are absent."""
    (tmp_path / "slurm_logs").mkdir()
    cli = tmp_path / "manual.log"
    cli.write_text("command not found: dorado\n")
    result = resolve_rule_log("my_rule", tmp_path / "slurm_logs",
                              declared_logs=[], cli_log=cli)
    assert result.source == "cli_arg"
    assert "command not found" in result.content


def test_list_slurm_rules(tmp_path):
    for rule in ["align_reads", "build_gene_list", "call_isoforms"]:
        (tmp_path / "slurm_logs" / rule).mkdir(parents=True)
    rules = list_slurm_rules(tmp_path / "slurm_logs")
    assert set(rules) == {"align_reads", "build_gene_list", "call_isoforms"}


def test_extract_job_id():
    assert _extract_job_id("12345") == "12345"
    assert _extract_job_id("slurm-98765") == "98765"
    assert _extract_job_id("job_42_0") == "42"


def test_tier1_uses_slurm_log_content(tmp_path):
    """End-to-end: Tier-1 pattern fires on content from slurm log."""
    _make_slurm_log(tmp_path, "align_reads", "SRR999", "777",
                    "slurmstepd: error: Exceeded step memory limit\nKilled\n")
    from snakemake_ai_debugger.log_collector import find_slurm_log
    from snakemake_ai_debugger.rules import quick_diagnose
    sl = find_slurm_log("align_reads", tmp_path / "slurm_logs")
    result = quick_diagnose(rule_log=sl.content, snakemake_block="")
    assert result is not None
    assert result.error_type == "OOMKilled"
