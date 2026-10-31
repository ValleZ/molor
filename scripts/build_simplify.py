"""Inject the simplifier payload into its template, making one portable file.

Usage: build_simplify.py [data.json] [out.html] [--mixpanel TOKEN]

Same contract as build_viewer.py, and for the same reasons: the analytics
snippet appears only when a token is passed, so a local build contacts nothing,
and the token is given on the command line rather than committed.
"""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_viewer import snippet  # noqa: E402  the identical loader stub


def main() -> int:
    argv = list(sys.argv[1:])
    token = ""
    if "--mixpanel" in argv:
        k = argv.index("--mixpanel")
        if k + 1 >= len(argv):
            raise SystemExit("--mixpanel needs a project token")
        token = argv[k + 1].strip()
        del argv[k:k + 2]
    data_name = argv[0] if argv else "artifacts/simplify_data.json"
    out_name = argv[1] if len(argv) > 1 else "artifacts/simplify.html"

    tpl = (ROOT / "viewer" / "simplify.html").read_text(encoding="utf-8")
    data = (ROOT / data_name).read_text(encoding="utf-8")
    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that could break out of it is a literal "</script>".
    data = data.replace("</", "<\\/")
    html = tpl.replace("__SIMPLIFY_DATA__", data)
    html = html.replace("__MIXPANEL_SNIPPET__", snippet(token) if token else "")
    out = ROOT / out_name
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1048576:.2f} MB)"
          + (f"  [mixpanel {token[:6]}...]" if token else "  [no analytics]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
