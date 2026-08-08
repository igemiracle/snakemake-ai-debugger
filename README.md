# snakemake-ai-debugger 🔬

**AI-powered root-cause analysis for failed Snakemake bioinformatics pipelines.**

When your Snakemake job crashes, this tool automatically collects the right
context — Snakemake logs, the relevant Snakefile rule, Slurm memory/time stats,
rule-specific stderr — and asks Claude (or a local Ollama model) to diagnose
the actual root cause, not just echo the error message.

---

## Quick install

```bash
pip install snakemake-ai-debugger
# or from source:
git clone https://github.com/you/snakemake-ai-debugger
cd snakemake-ai-debugger && pip install -e .
```

---

## Three ways to use it

### 1 · Manual (after a crash)

Run this in the directory where your pipeline lives:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or skip for Ollama

snakemake-ai-debugger \
    --snakefile Snakefile \
    --log-dir   .snakemake/log \
    --log       logs/extract_ids.err   # optional: rule-specific log
    --slurm-job 894312                 # optional: pulls sacct stats
```

You'll get a coloured terminal report **and** a timestamped YAML file:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔬 Snakemake AI Debugger — Diagnosis Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Rule          extract_ids
  Error type    MissingOutputException
  Confidence    82%

  Root Cause
    The rule exited 0 but never wrote temp.bed. Slurm shows no
    crash, so the tool silently produced empty output — most
    likely a wildcard/path mismatch or a malformed input record
    the tool skipped without erroring.

  Evidence
    ▸ .snakemake/log/2025-05-26.log:134   Output file temp.bed was not created
    ▸ Slurm job 894312                     ExitCode 0 — process did not crash

  Fix Suggestions
    1. Add a post-command guard: [ -s {output} ] || (echo "empty output" && exit 1)
    2. Verify the output path matches exactly what the command writes.
    3. Run the shell command manually with the expanded paths to reproduce.

  Follow-up Checks
    ○ grep -c '' logs/extract_ids.err  — confirm zero-byte output
    ○ Check container definition for awk version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📄 YAML report saved → ai_debug_20250526_143201.yaml
```

---

### 2 · Automatic (Snakefile hook)

Add **one line** to your Snakefile — no other changes:

```python
# Snakefile
include: "/path/to/snakemake_ai_debugger/snakemake_hook.py"

rule all:
    input: ...
```

Now every job failure automatically triggers the debugger.
Control it with environment variables:

```bash
export SNAKEMAKE_AI_BACKEND=auto   # claude | ollama | auto
export SNAKEMAKE_AI_QUIET=1        # set to silence auto-diagnosis
```

---

### 3 · Slurm epilog (cluster-wide, optional)

Ask your HPC admin to add to `/etc/slurm/epilog.sh`:

```bash
if [ "$SLURM_JOB_EXIT_CODE" != "0" ]; then
    snakemake-ai-debugger \
        --log-dir "$SLURM_SUBMIT_DIR/.snakemake/log" \
        --slurm-job "$SLURM_JOB_ID" \
        --backend ollama           # local model, no API key needed
fi
```

---

## LLM backends

| Backend | How it works | Needs |
|---------|-------------|-------|
| `claude` | Anthropic API | `ANTHROPIC_API_KEY` |
| `ollama` | Local model at `localhost:11434` | [Ollama](https://ollama.ai) running |
| `auto` *(default)* | Tries Claude first, falls back to Ollama | Either |

For HPC environments with no outbound internet, use `--backend ollama` with
a locally served model (e.g. `ollama run llama3.1:8b`).

---

## YAML report schema

```yaml
failed_rule: extract_ids
error_type: MissingOutputException
root_cause: "..."
evidence:
  - source: ".snakemake/log/...log:134"
    detail: "Output file temp.bed was not created"
fix_suggestions:
  - "Add post-command guard: [ -s {output} ] || (echo 'empty output' && exit 1)"
  - "Verify the output path matches exactly what the command writes"
confidence: 0.82
follow_up_checks:
  - "grep -c '' logs/extract_ids.err"
```

The YAML is machine-readable — pipe it into your ticket system, Slack bot,
or Snakemake report generator.

---

## Extending Tier-1 with your own rules

`rules.py` only ships tool-agnostic Snakemake/HPC/environment patterns
(missing output, OOM, conda errors, ...). If you want fast, free,
no-LLM-call detection for errors from tools specific to your own
pipeline, ship them as a separate pip-installable package instead of
forking this repo:

```python
# your_package/rules.py
from snakemake_ai_debugger.rules import RulePattern

MY_PATTERNS = [
    RulePattern(
        name="my_tool_error",
        error_type="ToolCrash:MyTool",
        patterns=[r"my_tool.*fatal error"],
        root_cause="...",
        fixes=["..."],
    ),
]

def get_patterns() -> list[RulePattern]:
    return MY_PATTERNS
```

```toml
# your_package/pyproject.toml
[project.entry-points."snakemake_ai_debugger.rule_packs"]
my_pack = "your_package.rules:get_patterns"
```

`pip install your_package` alongside `snakemake-ai-debugger` and the
patterns are picked up automatically — no Snakefile or core-package
changes needed. Custom patterns are checked before the built-ins, so
they can be more specific than a generic catch-all here. Run
`snakemake-ai-debugger --list-patterns` to confirm they loaded.

---

## Roadmap

- [ ] RAG knowledge base (Snakemake/Biostars issues, tool-specific errors)
- [ ] XGBoost resource predictor (dynamic `mem_mb` / `runtime`)
- [ ] VS Code extension integration
- [ ] `--watch` mode (tail `.snakemake/log` in real time)

---

## Contributing

Issues and PRs welcome. The core logic lives in `snakemake_ai_debugger/diagnose.py`
(~250 lines). The Snakemake hook is in `snakemake_hook.py` (30 lines).

---

## License

MIT
