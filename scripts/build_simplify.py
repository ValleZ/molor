"""Inject the simplifier payload into its template, making one portable file.

Usage: build_simplify.py [data.json] [out.html]
                         [--template PATH] [--mixpanel TOKEN]

Same contract as build_viewer.py, and for the same reasons: the analytics
snippet appears only when a token is passed, so a local build contacts nothing,
and the token is given on the command line rather than committed.

--template exists so an experimental variant of the page can be built the same
way the real one is: patch viewer/simplify.html, hand the result here, and the
payload is injected exactly as it is for the published build. Building a
variant by copying the template straight out is how one ships a page with the
__SIMPLIFY_DATA__ placeholder still in it, which parses as JSON on line 1 and
kills the whole script before any handler is attached.
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
    tpl_name = "viewer/simplify.html"
    if "--template" in argv:
        k = argv.index("--template")
        if k + 1 >= len(argv):
            raise SystemExit("--template needs a path")
        tpl_name = argv[k + 1].strip()
        del argv[k:k + 2]
    data_name = argv[0] if argv else "artifacts/simplify_data.json"
    out_name = argv[1] if len(argv) > 1 else "artifacts/simplify.html"

    tpl = (ROOT / tpl_name).read_text(encoding="utf-8")
    if "__SIMPLIFY_DATA__" not in tpl:
        raise SystemExit(
            f"{tpl_name} has no __SIMPLIFY_DATA__ placeholder, so the payload "
            f"would never reach the page. Is this a template or a built file?")
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
