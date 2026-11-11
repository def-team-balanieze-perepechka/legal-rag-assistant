from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> list[str]:
    files = [ROOT / "solution.py", ROOT / "requirements.txt", *sorted((ROOT / "core").glob("*.py"))]
    names = [f.name for f in files]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"одинаковые имена файлов в корне архива: {dupes}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "submission.zip"))
    args = ap.parse_args()
    names = build(Path(args.out))
    size = Path(args.out).stat().st_size
    print(f"{args.out}: {len(names)} файлов, {size / 1024:.1f} КБ", file=sys.stderr)
    print("\n".join(names))


if __name__ == "__main__":
    main()
