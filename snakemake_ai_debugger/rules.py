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
            "memory (or time) limit. For ONT data this is common with Dorado "
            "basecalling and IsoQuant on large samples."
        ),
        fixes=[
            "Increase `resources: mem_mb=` in the rule (check sacct MaxRSS for actual peak).",
            "If using Slurm: check `sacct -j <jobid> --format=MaxRSS,Elapsed` for actual usage.",
            "Split the input into smaller chunks and run in parallel.",
            "For IsoQuant: use `--low-memory` flag if available for your version.",
            "For Dorado: reduce `--batch-size` or run on a node with more RAM.",
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
        name="isoquant_no_reads",
        error_type="ToolCrash:IsoQuantNoReads",
        patterns=[
            r"isoquant.*no (reads|alignments)",
            r"isoquant.*empty (bam|input)",
            r"ERROR.*isoquant.*0 reads",
        ],
        root_cause="IsoQuant received a BAM with zero usable reads. The upstream alignment or filtering step produced an empty file.",
        fixes=[
            "Check the BAM: `samtools flagstat <file.bam>` — confirm reads are present.",
            "Verify the reference genome and annotation match (same chromosome naming: chr1 vs 1).",
            "Check minimap2 mapping rate — very low (<5%) usually means wrong reference.",
        ],
    ),

    RulePattern(
        name="dorado_gpu_error",
        error_type="ToolCrash:DoradoGPU",
        patterns=[
            r"dorado.*CUDA",
            r"dorado.*GPU.*error",
            r"CUDA error",
            r"no CUDA-capable device",
            r"device-side assert triggered",
        ],
        root_cause="Dorado failed to access or use the GPU. Either no GPU is allocated, the CUDA driver is incompatible, or GPU memory is exhausted.",
        fixes=[
            "Confirm GPU allocation: `nvidia-smi` inside the job environment.",
            "Request a GPU node in Slurm: add `#SBATCH --gres=gpu:1` or `resources: gpu=1`.",
            "Reduce `--batch-size` to lower GPU memory usage.",
            "Check CUDA version compatibility: `nvcc --version` vs Dorado's requirements.",
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

    # ── ONT RNA-seq pipeline tools ──────────────────────────────

    RulePattern(
        name="minimap2_index_error",
        error_type="ToolCrash:Minimap2",
        patterns=[
            r"minimap2.*\[ERROR\]",
            r"minimap2.*failed to open",
            r"minimap2.*can't open file",
            r"\[map_worker\].*failed",
            r"minimap2.*segmentation fault",
        ],
        root_cause="minimap2 failed to open the reference index or input file. The reference .mmi index may be missing, corrupted, or built with an incompatible minimap2 version.",
        fixes=[
            "Rebuild the index: delete the .mmi file and re-run the indexing rule.",
            "Check the reference FASTA exists and is not truncated: `samtools faidx <ref.fa>`.",
            "Confirm minimap2 version matches the index (re-index after version upgrade).",
        ],
    ),

    RulePattern(
        name="salmon_index_error",
        error_type="ToolCrash:SalmonIndex",
        patterns=[
            r"salmon.*Error",
            r"salmon.*could not open",
            r"salmon.*index.*not found",
            r"salmon.*invalid index",
            r"salmon.*failed to parse",
            r"Error computing effective length",
            r"salmon.*no valid alignments",
        ],
        root_cause="Salmon failed — typically a missing or incompatible index, an empty transcriptome FASTA, or no valid reads mapped.",
        fixes=[
            "Rebuild the Salmon index: delete it and re-run the index rule.",
            "Verify the transcriptome FASTA is non-empty: `grep -c '>' <transcripts.fa>`.",
            "Check read count in the input: `samtools flagstat <input.bam>` — very low mapping rate needs investigation.",
            "Confirm the decoys file matches the genome used for the index.",
        ],
    ),

    RulePattern(
        name="stringtie_no_reads",
        error_type="ToolCrash:StringTieNoReads",
        patterns=[
            r"StringTie.*Warning.*no valid reads",
            r"StringTie.*no valid bundles",
            r"stringtie.*error.*no valid",
            r"Error.*StringTie.*empty",
        ],
        root_cause="StringTie received a BAM with zero usable reads in the target region. Either the BAM is empty, filtered too aggressively, or reference chromosomes don't match.",
        fixes=[
            "Check BAM read count: `samtools flagstat <sample.bam>`.",
            "Verify chromosome naming matches between BAM and GTF (chr1 vs 1).",
            "Confirm the BAM is sorted and indexed: `samtools sort` then `samtools index`.",
        ],
    ),

    RulePattern(
        name="gffcompare_error",
        error_type="ToolCrash:GFFcompare",
        patterns=[
            r"gffcompare.*Error",
            r"gffcompare.*cannot open",
            r"Error.*parsing.*GTF",
            r"gffcompare.*failed",
            r"gffread.*error",
            r"gffread.*cannot open",
        ],
        root_cause="gffcompare or gffread failed to parse the GTF/GFF file. The file may be malformed, empty, or use incompatible annotation formats.",
        fixes=[
            "Validate the GTF: `grep -v '^#' <file.gtf> | awk '$3==\"transcript\"' | wc -l` — confirm > 0.",
            "Check if upstream StringTie/IsoQuant produced an empty GTF.",
            "Ensure the GTF uses the same chromosome naming as the genome FASTA.",
            "Run gffread manually to get the full error message.",
        ],
    ),

    # More specific SQANTI3 failure modes are checked first — the generic
    # catch-all below fires only when none of these match, so genuinely
    # different SQANTI3 crashes don't collapse into the same canned advice.
    RulePattern(
        name="sqanti3_cli_arg_error",
        error_type="ToolCrash:SQANTI3ArgError",
        patterns=[
            r"sqanti3_filter\.py:\s*error:\s*unrecognized arguments",
        ],
        root_cause=(
            "sqanti3_filter.py was called with a flag its subcommand does not "
            "accept (e.g. --gtf/--isoforms passed to 'rules' mode, which only "
            "the 'ml' mode takes). This is a wrong shell command in the "
            "Snakemake rule itself, not a data or environment problem."
        ),
        fixes=[
            "Compare the flags in the rule's shell command against "
            "`sqanti3_filter.py rules --help` / `sqanti3_filter.py ml --help` "
            "— the two subcommands take different arguments.",
            "Remove or move the flags that belong to the other subcommand.",
        ],
        confidence=0.97,
    ),

    RulePattern(
        name="sqanti3_report_error",
        error_type="ToolCrash:SQANTI3ReportGeneration",
        patterns=[
            r"Something went wrong during (Rules|Machine [Ll]earning) filtering report",
        ],
        root_cause=(
            "The SQANTI3 filter step ran but crashed while generating its own "
            "summary report. The actual exception is in SQANTI3's own "
            "filter_report.log (path printed on the line above this error in "
            "the Slurm log), not in this job's stderr."
        ),
        fixes=[
            "Open the filter_report.log path printed right above this error — "
            "that has the real traceback.",
            "This report step is usually R/rmarkdown-based — confirm R and "
            "its packages are present inside the container.",
        ],
        confidence=0.9,
    ),

    RulePattern(
        name="sqanti3_error",
        error_type="ToolCrash:SQANTI3",
        patterns=[
            r"SQANTI3.*[Ee]rror",
            r"sqanti.*[Ee]rror",
            r"sqanti.*Traceback",
            r"sqanti.*failed",
            r"Error.*sqanti.*GTF",
            r"sqanti.*no transcripts",
        ],
        root_cause="SQANTI3 QC failed — typically due to an empty or malformed input GTF, mismatched genome annotation, or missing reference files.",
        fixes=[
            "Check the input GTF from StringTie/IsoQuant: `awk '$3==\"transcript\"' <input.gtf> | wc -l`.",
            "Verify the genome annotation GTF and the sample GTF use the same chromosome format.",
            "Confirm that all SQANTI3 reference files (cage peaks, polyA sites) are present.",
            "Run SQANTI3 manually on a small subset to reproduce and see the full traceback.",
        ],
    ),

    RulePattern(
        name="jaffa_fusion_error",
        error_type="ToolCrash:JAFFAfusion",
        patterns=[
            r"JAFFA.*[Ee]rror",
            r"jaffa.*failed",
            r"bpipe.*error",
            r"jaffal.*Traceback",
            r"java.*OutOfMemoryError",
            r"JAFFA.*no results",
        ],
        root_cause="JAFFA gene-fusion detection failed. Common causes: Java heap exhaustion, missing JAFFA reference database, or too few input reads.",
        fixes=[
            "Increase Java heap in the JAFFA bpipe config: add `-Xmx<N>g` to the Java call.",
            "Verify the JAFFA reference directory is bound/accessible in the container (`--bind /path/to/jaffa_ref`).",
            "Check fastq read count: JAFFA needs a minimum number of reads to produce results.",
        ],
    ),

    RulePattern(
        name="ctat_fusion_error",
        error_type="ToolCrash:CTATfusion",
        patterns=[
            r"ctat.*(LR.)?fusion.*[Ee]rror",
            r"STAR-Fusion.*[Ee]rror",
            r"ctat.*failed",
            r"STAR.*genome.*not found",
            r"STAR.*genome.*load.*error",
        ],
        root_cause="CTAT-LR-Fusion / STAR-Fusion failed. Usually the CTAT genome library path is wrong, the STAR genome is missing, or there is insufficient memory for genome loading.",
        fixes=[
            "Verify the CTAT library path matches the `--genome_lib_dir` / bind path in the singularity call.",
            "Re-download or untar the CTAT genome library if it was interrupted.",
            "Increase `mem_mb` — STAR genome loading requires ~30 GB RAM for human genome.",
        ],
    ),

    RulePattern(
        name="dorado_modification_error",
        error_type="ToolCrash:DoradoModification",
        patterns=[
            r"dorado.*mod.*[Ee]rror",
            r"dorado.*basecall.*[Ee]rror",
            r"dorado.*model.*not found",
            r"dorado.*could not load model",
            r"dorado.*pod5.*error",
            r"pod5.*[Ee]rror",
        ],
        root_cause="Dorado modification basecalling failed. The modification model may be missing, the POD5 file is corrupt, or GPU allocation is insufficient.",
        fixes=[
            "Download the modification model: `dorado download --model <model_name>`.",
            "Verify the POD5 file is not truncated: `pod5 view <file.pod5> | head`.",
            "Confirm GPU allocation: `nvidia-smi` — Dorado modification calling requires a GPU.",
            "Check container bind paths include the POD5 data directory.",
        ],
    ),

    RulePattern(
        name="nanoplot_error",
        error_type="ToolCrash:NanoPlot",
        patterns=[
            r"NanoPlot.*[Ee]rror",
            r"nanoplot.*[Ee]rror",
            r"NanoPlot.*failed",
            r"NanoStat.*[Ee]rror",
        ],
        root_cause="NanoPlot QC failed — typically an empty input FASTQ/BAM or an incompatible input format.",
        fixes=[
            "Check that the input FASTQ/BAM exists and is non-empty.",
            "Verify the input is a valid ONT FASTQ (contains @-headers with length/quality fields).",
        ],
        confidence=0.85,
    ),
]


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
