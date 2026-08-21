"""
Command-line entry point for the ArchForge prototype.

Usage:
    python -m archforge.cli ask "Write a CAN driver init function for the M_CAN peripheral"
    python -m archforge.cli ask --file query.txt
    python -m archforge.cli init-config
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ArchForgeConfig, DEFAULT_CONFIG_PATH
from .pipeline import ArchForgePipeline

LOG_DIR = Path("logs")


def cmd_ask(args: argparse.Namespace) -> None:
    config = ArchForgeConfig.load(args.config)
    pipeline = ArchForgePipeline(config)

    query = Path(args.file).read_text() if args.file else args.query
    if not query:
        print("Provide a query directly or with --file.", file=sys.stderr)
        sys.exit(1)

    print("Drafting...")
    result = pipeline.run(query)

    print("\n--- Grounded on ---")
    print(", ".join(result.citations) if result.citations else "(no matching reference found)")

    print("\n--- Final answer ---")
    print(result.final_code)

    print("\n--- Validation ---")
    if result.validation.passed:
        print(f"Passed after {result.validation.attempts} attempt(s).")
    else:
        print(f"Unresolved after {result.validation.attempts} attempt(s):")
        for v in result.validation.violations:
            print(f"  - Line {v.line}, {v.rule}: {v.message}")
    if result.validation.compile_checked and result.validation.compile_output:
        print(f"Compile check output:\n{result.validation.compile_output}")

    if args.log:
        _write_log(result)


def _write_log(result) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{timestamp}.md"
    log_path.write_text(
        f"# Query\n\n{result.query}\n\n"
        f"# Draft\n\n{result.draft}\n\n"
        f"# Grounded ({', '.join(result.citations) or 'no citations'})\n\n"
        f"{result.grounded_answer}\n\n"
        f"# Final (validated)\n\n{result.final_code}\n\n"
        f"# Validation\n\npassed={result.validation.passed}, "
        f"attempts={result.validation.attempts}\n"
    )
    print(f"\nLogged to {log_path}")


def cmd_init_config(args: argparse.Namespace) -> None:
    path = Path(args.config)
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    ArchForgeConfig().save(path)
    print(f"Wrote default config to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="archforge", description="ArchForge prototype CLI")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run a query through draft / ground / validate")
    ask_parser.add_argument("query", nargs="?", help="The question or request")
    ask_parser.add_argument("--file", help="Read the query from a file instead")
    ask_parser.add_argument("--log", action="store_true", help="Save the full trace under logs/")
    ask_parser.set_defaults(func=cmd_ask)

    init_parser = subparsers.add_parser("init-config", help="Write a default config file")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init_config)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
