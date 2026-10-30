"""Inject the exported data into the viewer template to make one portable file.

Usage: build_viewer.py [data.json] [out.html] [--mixpanel TOKEN]

The analytics snippet is injected only when a token is passed, so a build run
locally contains no tracking code at all and contacts nothing. The deployed
build is the only one that carries it.

What the snippet does and does not do is a deliberate choice. It counts visits
via ``track_pageview`` and nothing else: no session recording, no heatmaps, no
``identify`` call, so no visitor is given a persistent identity by this page
beyond the anonymous device id Mixpanel's own library keeps in localStorage.
Session replay at 100% is a different thing from counting visitors, and the
page is public, so it is left off and switched on deliberately if ever wanted.

The project token is a client-side write key and is meant to be public, so
embedding it in the page is normal. It is passed on the command line rather
than committed, because a token in the repository is a token that outlives the
reason it was added.
"""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]

# The loader stub, verbatim from Mixpanel's own documentation. It is minified
# upstream and left exactly as shipped: reformatting someone else's loader is
# how a subtle difference gets introduced between what was tested and what runs.
STUB = (
    '(function (f, b) { if (!b.__SV) { var e, g, i, h; window.mixpanel = b; '
    'b._i = []; b.init = function (e, f, c) { function g(a, d) { var b = '
    'd.split("."); 2 == b.length && ((a = a[b[0]]), (d = b[1])); a[d] = '
    'function () { a.push([d].concat(Array.prototype.slice.call(arguments, '
    '0))); }; } var a = b; "undefined" !== typeof c ? (a = b[c] = []) : (c = '
    '"mixpanel"); a.people = a.people || []; a.toString = function (a) { var d '
    '= "mixpanel"; "mixpanel" !== c && (d += "." + c); a || (d += " (stub)"); '
    'return d; }; a.people.toString = function () { return a.toString(1) + '
    '".people (stub)"; }; i = "disable time_event track track_pageview '
    'track_links track_forms track_with_groups add_group set_group '
    'remove_group register register_once alias unregister identify name_tag '
    'set_config reset opt_in_tracking opt_out_tracking has_opted_in_tracking '
    'has_opted_out_tracking clear_opt_in_out_tracking start_batch_senders '
    'people.set people.set_once people.unset people.increment people.append '
    'people.union people.track_charge people.clear_charges people.delete_user '
    'people.remove".split( " "); for (h = 0; h < i.length; h++) g(a, i[h]); '
    'var j = "set set_once union unset remove delete".split(" "); '
    'a.get_group = function () { function b(c) { d[c] = function () { '
    'call2_args = arguments; call2 = [c].concat(Array.prototype.slice.call('
    'call2_args, 0)); a.push([e, call2]); }; } for ( var d = {}, e = '
    '["get_group"].concat( Array.prototype.slice.call(arguments, 0)), c = 0; '
    'c < j.length; c++) b(j[c]); return d; }; b._i.push([e, f, c]); }; '
    'b.__SV = 1.2; e = f.createElement("script"); e.type = "text/javascript"; '
    'e.async = !0; e.src = "undefined" !== typeof MIXPANEL_CUSTOM_LIB_URL ? '
    'MIXPANEL_CUSTOM_LIB_URL : "file:" === f.location.protocol && '
    '"//cdn.mxpnl.com/libs/mixpanel-2-latest.min.js".match(/^\\/\\//) ? '
    '"https://cdn.mxpnl.com/libs/mixpanel-2-latest.min.js" : '
    '"//cdn.mxpnl.com/libs/mixpanel-2-latest.min.js"; g = '
    'f.getElementsByTagName("script")[0]; g.parentNode.insertBefore(e, g); } '
    '})(document, window.mixpanel || []);'
)


def snippet(token: str) -> str:
    """The loader plus an init that counts visits and does nothing else."""
    return (
        '<script type="text/javascript">\n'
        + STUB
        + "\n</script>\n"
        '<script type="text/javascript">\n'
        "  // Counting visits only. record_sessions_percent and\n"
        "  // record_heatmap_data are deliberately absent: this page is public,\n"
        "  // and replaying a stranger's session is not the same question as\n"
        "  // how many strangers there were. No identify() call either, so\n"
        "  // visitors stay anonymous to this page.\n"
        '  if (window.mixpanel && window.mixpanel.init) {\n'
        '    try {\n'
        '      window.mixpanel.init("' + token + '", {\n'
        "        track_pageview: true,\n"
        '        persistence: "localStorage",\n'
        "        ip: true\n"
        "      });\n"
        "    } catch (e) { /* analytics must never break the page */ }\n"
        "  }\n"
        "</script>"
    )


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    token = ""
    if "--mixpanel" in argv:
        k = argv.index("--mixpanel")
        if k + 1 >= len(argv):
            raise SystemExit("--mixpanel needs a project token")
        token = argv[k + 1].strip()
        del argv[k:k + 2]
    data_name = argv[0] if len(argv) > 0 else "artifacts/viewer_data.json"
    out_name = argv[1] if len(argv) > 1 else "artifacts/receptor_atlas.html"

    tpl = (ROOT / "viewer" / "template.html").read_text(encoding="utf-8")
    data = (ROOT / data_name).read_text(encoding="utf-8")
    # The payload sits inside a <script type="application/json"> block, so the
    # only sequence that could break out of it is a literal "</script>".
    data = data.replace("</", "<\\/")
    out = ROOT / out_name
    html = tpl.replace("__ORVECTOR_DATA__", data)
    html = html.replace("__MIXPANEL_SNIPPET__", snippet(token) if token else "")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1048576:.2f} MB)"
          + (f"  [mixpanel {token[:6]}...]" if token else "  [no analytics]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
