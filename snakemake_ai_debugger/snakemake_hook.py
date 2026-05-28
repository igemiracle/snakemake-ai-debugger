"""
snakemake_hook.py  —  drop-in Snakemake onerror hook
=====================================================
Add this to your Snakefile with one line:

    include: "path/to/snakemake_hook.py"

Or call the Python API directly (preferred — no include conflicts):

    from snakemake_ai_debugger.collect import on_error_hook
    onerror:
        on_error_hook(log)

Environment variables
---------------------
  SNAKEMAKE_AI_QUIET     set to 1 to suppress auto-diagnosis on error
"""

import os


def _run_debugger(log: str, snakefile: str) -> None:
    if os.environ.get("SNAKEMAKE_AI_QUIET") == "1":
        return
    try:
        from snakemake_ai_debugger.collect import on_error_hook
        print("\n\033[36m  🔬 snakemake-ai-debugger: analysing failure…\033[0m")
        on_error_hook(log)
    except ImportError:
        print(
            "\033[33m  snakemake-ai-debugger not found. "
            "Install with: pip install snakemake-ai-debugger\033[0m"
        )
    except Exception as e:
        print(f"\033[33m  snakemake-ai-debugger: {e}\033[0m")


onerror:
    # `log` is the Snakemake-provided path to the overall run log
    _run_debugger(log=log, snakefile=snakemake.snakefile)
