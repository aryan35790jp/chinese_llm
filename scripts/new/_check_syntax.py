"""
Tiny utility: compile-check every new file. Run it after editing.

    python scripts/new/_check_syntax.py
"""
from __future__ import annotations
import glob
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    files = (
        glob.glob(str(ROOT / "radical_lib" / "*.py"))
        + glob.glob(str(ROOT / "scripts" / "new" / "*.py"))
    )
    n_ok = 0
    n_fail = 0
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
            n_ok += 1
        except py_compile.PyCompileError as e:
            n_fail += 1
            print(f"[FAIL] {f}\n  {e.msg}")
    print(f"\n{n_ok} OK, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
