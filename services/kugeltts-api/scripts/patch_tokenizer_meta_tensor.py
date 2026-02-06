#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PATTERN = "torch.linspace(0, drop_path_rate, sum(self.depths))"
REPLACEMENT = 'torch.linspace(0, drop_path_rate, sum(self.depths), device="cpu")'


def patch_file(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    count_old = content.count(PATTERN)
    count_new = content.count(REPLACEMENT)

    if count_old == 0 and count_new == 0:
        raise RuntimeError(
            "Expected tokenizer pattern not found; refusing to continue without explicit patch target."
        )

    if count_old == 0:
        print(f"No unpatched occurrences found in {path}.")
        return 0

    updated = content.replace(PATTERN, REPLACEMENT)
    path.write_text(updated, encoding="utf-8")
    print(f"Patched {count_old} occurrence(s) in {path}.")
    return count_old


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "src/kugelaudio_open/models/tokenizer.py"
    )
    if not target.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {target}.")
    patch_file(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
