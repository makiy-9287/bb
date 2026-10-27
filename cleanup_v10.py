#!/usr/bin/env python3
"""
V10 cleanup — run ONCE:   python3 cleanup_v10.py        (add --dry-run to only preview)

Deletes the old (v6–v9: scalp / ICT-SMC / manual-payload) files that V10 no longer uses,
then DELETES ITSELF. It only touches the explicit list below, inside this folder.
Never touched: .env, active_trades.json, closed_trades.json, data/, agent.log, V10 files.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

OBSOLETE_FILES = [
    # engine / scalp / manual workflow (v7–v9)
    "ict_engine.py", "backtest.py", "manual_payload.py", "signal_filter.py", "indicators.py",
    "SCALP_PROMPT.md", "TEST_REPORT.md",
    # old tests of removed modules
    "tests/test_engine.py", "tests/test_backtest_and_flow.py", "tests/test_manual_payload.py",
    "tests/test_agent_main.py", "tests/engine_helpers.py",
]
OBSOLETE_DIRS = ["legacy", "__pycache__", "tests/__pycache__", ".pytest_cache", "debug_reasoning"]
OBSOLETE_GLOBS_IN = {"outputs": ("scalp_payload_", ".json")}        # old manual payload files only


def _looks_like_v10() -> bool:
    try:
        v = open(os.path.join(HERE, "VERSION"), encoding="utf-8").read().strip()
        return v in ("V10", "V11") and ("V10" in open(os.path.join(HERE, "main.py"), encoding="utf-8").read() or
                                        "V11" in open(os.path.join(HERE, "main.py"), encoding="utf-8").read())
    except OSError:
        return False


def main(argv=None) -> int:
    dry = "--dry-run" in (argv if argv is not None else sys.argv[1:])
    if not _looks_like_v10():
        print("✖ This does not look like the V10 folder (VERSION/main.py missing) — nothing deleted.")
        return 1
    targets = []
    for f in OBSOLETE_FILES:
        p = os.path.join(HERE, f)
        if os.path.isfile(p):
            targets.append(("file", p))
    for d in OBSOLETE_DIRS:
        p = os.path.join(HERE, d)
        if os.path.isdir(p):
            targets.append(("dir", p))
    for d, (prefix, suffix) in OBSOLETE_GLOBS_IN.items():
        folder = os.path.join(HERE, d)
        if os.path.isdir(folder):
            targets += [("file", os.path.join(folder, n)) for n in sorted(os.listdir(folder))
                        if n.startswith(prefix) and n.endswith(suffix)]
            if not [n for n in os.listdir(folder) if not (n.startswith(prefix) and n.endswith(suffix))]:
                targets.append(("dir", folder))                     # only if it would be left empty
    if not targets:
        print("✔ Nothing to clean — already V10-only.")
    for kind, p in targets:
        rel = os.path.relpath(p, HERE)
        if dry:
            print(f"[dry-run] would delete {kind}: {rel}")
            continue
        try:
            shutil.rmtree(p) if kind == "dir" else os.remove(p)
            print(f"🗑  deleted {kind}: {rel}")
        except OSError as exc:
            print(f"⚠  could not delete {rel}: {exc}")
    if dry:
        print("\n(dry-run: nothing deleted, script kept)")
        return 0
    try:
        os.remove(os.path.abspath(__file__))
        print("\n✔ Cleanup done — this script deleted itself.")
    except OSError as exc:
        print(f"\n✔ Cleanup done (could not self-delete: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
