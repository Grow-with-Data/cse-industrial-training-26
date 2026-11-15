"""Write the API's OpenAPI document to docs/openapi.json, or check that it is current.

    uv run python scripts/export_openapi.py            # regenerate docs/openapi.json
    uv run python scripts/export_openapi.py --check    # exit 1 if it is out of date

WHY COMMIT A GENERATED FILE
    /openapi.json is built from the code on every request, so why keep a copy?
    Because the copy is REVIEWED. Rename a field and `git diff docs/openapi.json`
    shows exactly what every client will see change, in the same pull request.
    Session 15's frontend (and the Claude Code session that writes it) reads
    this file, not the running server.

    --check is the guard: run it before you commit (session 14 runs it in CI).
    If a route or a schema changed and nobody regenerated the file, it fails
    and says so.

Building the schema needs no database, no key and no model: it imports the app
and asks FastAPI for app.openapi(). The lifespan does not run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shopwise.main import app

OUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if docs/openapi.json is stale")
    args = parser.parse_args()
    current = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != current:
            print(f"{OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT} is out of date. "
                  "Run: uv run python scripts/export_openapi.py")
            return 1
        print("docs/openapi.json is up to date.")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(current, encoding="utf-8")
    print(f"wrote {OUT} ({len(app.openapi()['paths'])} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
