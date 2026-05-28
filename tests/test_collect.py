"""
Tests for collect.py — pure error collection, no LLM.
"""
import textwrap, time
from pathlib import Path
import pytest

from snakemake_ai_debugger.collect import (
    parse_main_log,
    collect,
    render,
    save_report,
    CollectedError,
    FailedJobMeta,
)
from snakemake_ai_debugger.log_collector import SlurmLogResult
from snakemake_ai_debugger.rules import QuickDiagnosis

# ── fixtures ──────────────────────────────────────────────────────

MAIN_LOG_OOM = textwrap.dedent("""\
    Building DAG of jobs...
    [Thu May 26 09:00:01 2026]
    rule align_reads:
        input: data/SRR123.fastq
        output: bam/SRR123.bam
        log: logs/align_SRR123.log
        jobid: 7
        wildcards: sample=SRR123
        resources: mem_mb=8000

    [Thu May 26 09:45:12 2026]
    Error in rule align_reads:
        jobid: 7
        input: data/SRR123.fastq
        output: bam/SRR123.bam
        log: logs/align_SRR123.log
        wildcards: sample=SRR123

    CalledProcessError: Command 'minimap2 ...' returned non-zero exit status 137.
    Shutting down...
""")

MAIN_LOG_TWO_FAILURES = textwrap.dedent("""\
    [Thu May 26 09:00:01 2026]
    Error in rule align_reads:
        jobid: 3
        input: data/SRR001.fastq
        output: bam/SRR001.bam
        wildcards: sample=SRR001

    CalledProcessError: exit status 137.

    Error in rule call_isoforms:
        jobid: 5
        input: bam/SRR001.bam
        output: isoforms/SRR001.gtf
        log: logs/isoquant.log
        wildcards: sample=SRR001

    MissingOutputException: isoforms/SRR001.gtf not present.
""")

SLURM_OOM_CONTENT = textwrap.dedent("""\
    srun: error: node01: task 0: Killed
    slurmstepd: error: Exceeded step memory limit at some point.
    srun: error: Bailing out, caught error 9
""")

SLURM_MISSING_OUT_CONTENT = textwrap.dedent("""\
    [IsoQuant] Loading annotation...
    [IsoQuant] Processing sample SRR001
    [IsoQuant] Writing output...
    [IsoQuant] Done.
    # (output file never actually written — silent failure)
""")


def _make_slurm(tmp_path, rule, wildcard_dir, job_id, content):
    d = tmp_path / "slurm_logs" / rule
    if wildcard_dir:
        d = d / wildcard_dir
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{job_id}.log"
    f.write_text(content)
    return f


# ── parse_main_log ────────────────────────────────────────────────

def test_parse_single_failure():
    metas = parse_main_log(MAIN_LOG_OOM)
    assert len(metas) == 1
    m = metas[0]
    assert m.rule_name == "align_reads"
    assert m.jobid == "7"
    assert m.wildcards == "sample=SRR123"
    assert any("SRR123.fastq" in f for f in m.input_files)
    assert any("align_SRR123.log" in str(p) for p in m.declared_logs)


def test_parse_two_failures():
    metas = parse_main_log(MAIN_LOG_TWO_FAILURES)
    assert len(metas) == 2
    names = [m.rule_name for m in metas]
    assert "align_reads" in names
    assert "call_isoforms" in names


def test_parse_empty():
    assert parse_main_log("Nothing went wrong today.") == []


def test_parse_wildcards_captured():
    metas = parse_main_log(MAIN_LOG_TWO_FAILURES)
    align = next(m for m in metas if m.rule_name == "align_reads")
    assert align.wildcards == "sample=SRR001"


# ── collect() — integration with fake slurm_logs ─────────────────

def test_collect_oom_from_slurm(tmp_path):
    """OOM error in slurm log → Tier-1 recognises it."""
    _make_slurm(tmp_path, "align_reads", "SRR123", "7", SLURM_OOM_CONTENT)
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "2026-05-26T09.log").write_text(MAIN_LOG_OOM)

    errors = collect(
        log_dir       = tmp_path / "log",
        slurm_log_dir = tmp_path / "slurm_logs",
    )
    assert len(errors) == 1
    e = errors[0]
    assert e.rule_name == "align_reads"
    assert e.recognised
    assert e.diagnosis.error_type == "OOMKilled"
    assert e.slurm_result.source == "slurm_logs"


def test_collect_missing_output_recognised(tmp_path):
    """MissingOutputException in main-log block → recognised even without slurm log."""
    (tmp_path / "slurm_logs").mkdir()
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "2026-05-26T09.log").write_text(MAIN_LOG_TWO_FAILURES)

    errors = collect(
        log_dir       = tmp_path / "log",
        slurm_log_dir = tmp_path / "slurm_logs",
    )
    isoform_err = next(e for e in errors if e.rule_name == "call_isoforms")
    assert isoform_err.recognised
    assert isoform_err.diagnosis.error_type == "MissingOutputException"


def test_collect_unrecognised_shows_error_context(tmp_path):
    """Unknown error → unrecognised, error context preserved in dict."""
    # Mix of normal output and an error line in the middle
    weird_log = (
        "Tool starting...\nLoading reference...\n"
        "Processing reads...\nSome completely new tool output\n"
        "FATAL: unexpected condition in processor\n"
        "DoneWeirdly\nAll finished.\n"
    )
    _make_slurm(tmp_path, "align_reads", "SRR123", "7", weird_log)
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "2026-05-26.log").write_text(MAIN_LOG_OOM)

    errors = collect(
        log_dir=tmp_path / "log",
        slurm_log_dir=tmp_path / "slurm_logs",
    )
    e = errors[0]
    assert not e.recognised
    d = e.to_dict()
    assert "error_context" in d
    # error context should include the FATAL line, not just the final lines
    assert "FATAL" in d["error_context"]


def test_collect_two_failures(tmp_path):
    """Two rules fail → two CollectedErrors with correct rule names."""
    _make_slurm(tmp_path, "align_reads",  "SRR001", "3", SLURM_OOM_CONTENT)
    _make_slurm(tmp_path, "call_isoforms","SRR001", "5", SLURM_MISSING_OUT_CONTENT)
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "2026-05-26.log").write_text(MAIN_LOG_TWO_FAILURES)

    errors = collect(
        log_dir=tmp_path / "log",
        slurm_log_dir=tmp_path / "slurm_logs",
    )
    assert len(errors) == 2
    names = {e.rule_name for e in errors}
    assert names == {"align_reads", "call_isoforms"}


def test_collect_no_log_dir(tmp_path):
    """Missing log dir → empty list, no crash."""
    result = collect(
        log_dir=tmp_path / "nonexistent",
        slurm_log_dir=tmp_path / "slurm_logs",
    )
    assert result == []


# ── render smoke test ────────────────────────────────────────────

def _fake_error(rule, recognised=True, wildcards=""):
    diag = QuickDiagnosis(
        error_type="OOMKilled",
        root_cause="Job exceeded memory.",
        fix_suggestions=["Increase mem_mb."],
        matched_pattern="oom_killed · /Killed/",
        confidence=0.95,
    ) if recognised else None
    sl = SlurmLogResult(
        rule_name=rule, log_path=Path(f".snakemake/slurm_logs/{rule}/1.log"),
        wildcards=wildcards, job_id="1",
        content="Killed\n" if recognised else "some unknown output\n",
        source="slurm_logs",
    )
    return CollectedError(
        rule_name=rule, wildcards=wildcards, jobid="42",
        slurm_result=sl, diagnosis=diag,
        input_files=[], output_files=[],
    )


def test_render_recognised(capsys):
    render([_fake_error("align_reads", recognised=True, wildcards="sample=SRR001")])
    out = capsys.readouterr().out
    assert "align_reads" in out
    assert "OOMKilled" in out
    assert "SRR001" in out


def test_render_unrecognised(capsys):
    render([_fake_error("mystery_rule", recognised=False)])
    out = capsys.readouterr().out
    assert "mystery_rule" in out
    assert "Unrecognised" in out


def test_render_mixed(capsys):
    errors = [
        _fake_error("align_reads", recognised=True),
        _fake_error("weird_rule",  recognised=False),
    ]
    render(errors)
    out = capsys.readouterr().out
    assert "1 recognised" in out
    assert "1 unrecognised" in out


# ── save_report ────────────────────────────────────────────────

def test_save_report(tmp_path):
    errors = [_fake_error("align_reads", recognised=True, wildcards="sample=X")]
    out = tmp_path / "report.yaml"
    save_report(errors, out)
    assert out.exists()
    import yaml
    data = yaml.safe_load(out.read_text())
    assert "collected_at" in data
    assert data["failed_jobs"][0]["rule"] == "align_reads"
    assert data["failed_jobs"][0]["wildcards"] == "sample=X"
    assert data["failed_jobs"][0]["error_type"] == "OOMKilled"
