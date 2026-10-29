"""Inject the exported data into the viewer template to make one portable file."""
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
data_name = sys.argv[1] if len(sys.argv) > 1 else "artifacts/viewer_data.json"
out_name = sys.argv[2] if len(sys.argv) > 2 else "artifacts/receptor_atlas.html"
tpl = (ROOT / "viewer" / "template.html").read_text(encoding="utf-8")
data = (ROOT / data_name).read_text(encoding="utf-8")
# The payload sits inside a <script type="application/json"> block, so the only
# sequence that could break out of it is a literal "</script>".
data = data.replace("</", "<\/")
out = ROOT / out_name
out.write_text(tpl.replace("__ORVECTOR_DATA__", data), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1048576:.2f} MB)")
